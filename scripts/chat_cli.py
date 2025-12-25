import os
import sys

import jax
import jax.numpy as jnp

from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.common import get_base_dir
from nanojax.gpt import GPT, KVCache
from nanojax.tokenizer import get_tokenizer

checkpoint = "mid"
compute_dtype = jnp.bfloat16
max_tokens = 16
seed = 42

exec(open(os.path.join('nanojax', 'configurator.py')).read()) # overrides from command line 
command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print(command)

tokenizer = get_tokenizer()
base_dir = get_base_dir()
checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
model_cfg = load_model_config(checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

model = GPT.init(
    model_cfg,
    rng
)
model = load_checkpoint(checkpoint_dir / "model.zarr", model)

max_seq_len = model_cfg.sequence_len * 2
def generate(idx):
    # setup KV-caches for a single sample and up to 2x model context length
    kv_cache = KVCache.init(
        batch_size=1,
        max_seq_len=max_seq_len,
        n_layer=model_cfg.n_layer,
        embed_dim=model_cfg.n_embed,
        n_head=model_cfg.n_head,
        n_kv_head=model_cfg.n_kv_head,
        compute_dtype=compute_dtype
    )
    idx = jnp.asarray(idx, dtype=jnp.int32)[None, :]
    # prefill
    logits, kv_cache = model.forward(idx, compute_dtype=compute_dtype, kv_cache=kv_cache)
    logits = logits[:, -1, :] # bsv -> bv
    
    pred = jnp.argmax(logits, axis=-1, keepdims=True)
    idx = jnp.concat((idx, pred), axis=1)
    
    @jax.jit
    def generate_next_token(idx, mask, kv_cache):
        logits, kv_cache = model.forward(idx, mask=mask, compute_dtype=compute_dtype, kv_cache=kv_cache)
        logits = logits[:, -1, :] # bsv -> bv
        pred = jnp.argmax(logits, axis=-1, keepdims=True)
        return pred, kv_cache

    for _ in range(max_tokens):
        # async kick off generation for the next token
        # our cached k,v are 0s for all positions we haven't filled yet up to our
        # pre-defined cache max_seq_len, so we need to mask these out.
        s = idx.shape[1]
        mask = jnp.arange(kv_cache.k.shape[1]) < kv_cache.pos + s
        next_token, kv_cache = generate_next_token(pred, mask, kv_cache)
        yield pred[0]
        pred = next_token
        idx = jnp.concat((idx, pred), axis=1)
        

user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

tokens = []
while True:
    try:
        user_input = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    if not user_input:
        continue

    tokens.append(user_start)
    tokens.extend(tokenizer.encode(user_input))
    tokens.append(user_end)
    tokens.append(assistant_start)
    print("\nAssistant: ", end="", flush=True)

    for token in generate(tokens):
        tokens.append(token[0])
        if token[0] == assistant_end:
            break
        print(tokenizer.decode(token), end="", flush=True)

    if token[0] != assistant_end:
        tokens.append(assistant_end)
    if len(tokens) > max_seq_len:
        print(f"Max sequence len {max_seq_len} exceeded. Goodbye!")
        break

    print()
    
    

    
    
