
"""
Train a tokenizer using the HuggingFace Tokenizers library.
In the style of GPT-4 tokenizer.

This script is identical to karparthy/nanochat/scripts/tok_train.py
but uses Zarr instead of torch to serialize token_bytes.py
"""
import argparse
import random
import time

import numpy as np
import zarr

from nanocode.common import get_model_dir, init_distributed, print0, setup_logging
from data.pretrain import parquets_iter_batched
from nanocode.tokenizer import RustBPETokenizer

init_distributed()
# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)') 
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=32768, help='Vocabulary size (default: 32768, GPT2-small)')
parser.add_argument('--code_ratio', type=float, default=0.2, help='Fraction of code data (default: 0.2)')
args = parser.parse_args()
print0(f"max_chars: {args.max_chars:,}")
print0(f"doc_cap: {args.doc_cap:,}")
print0(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    fineweb = parquets_iter_batched("fineweb-edu", split="train")
    stack_v2 = parquets_iter_batched("the-stack-v2-dedup", split="train")
    batch_iter = lambda: stack_v2 if random.random() < args.code_ratio else fineweb
    nchars = 0
    while nchars <= args.max_chars:
        try:
            batch = next(batch_iter())
        except StopIteration:
            break
        for doc in batch:
            doc_text = doc[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
model_dir = get_model_dir()
tokenizer_dir = model_dir / "tokenizer"
tokenizer_dir.mkdir(parents=True, exist_ok=True)
setup_logging(model_dir / "tok_train.txt")
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print0(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = np.array(token_bytes, dtype=np.int32)
token_bytes_path = tokenizer_dir / "token_bytes.zarr" 
zarr.save(token_bytes_path, token_bytes)
print0(f"Saved token_bytes to {token_bytes_path}")



