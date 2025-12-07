from collections import deque

import jax.numpy as jnp
import numpy as np
from nanojax.dataset import parquets_iter_batched


def tokenizing_data_loader(B, T, split, tokenizer, tokenizer_threads=4, tokenizer_batch_size=128):
    """Stream pretraining text from parquet files, tokenize, yield training batches."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(split=split):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()
    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        # Move tokens from the deque into the scratch buffer
        # note: JAX does not natively support int64 (see JAX gotchas), but this isn't really an issue
        # as torch's cross entropy requires int64 targets for only historical(?) reasons
        tokens = np.array([token_buffer.popleft() for _ in range(needed_tokens)], dtype=np.int32)
        # Create the inputs/targets and yield
        inputs = jnp.asarray(tokens[:-1]).reshape(B, T)
        targets = jnp.asarray(tokens[1:]).reshape(B, T)
        yield inputs, targets

