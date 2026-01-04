import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.common import get_base_dir, print0
from nanojax.gpt import GPT, KVCache
from nanojax.tokenizer import get_tokenizer

checkpoint = "mid"
compute_dtype = jnp.bfloat16
max_tokens = 128
seed = 42
temperature = 0.6

exec(open(os.path.join("nanojax", "configurator.py")).read()) # overrides from command line

tokenizer = get_tokenizer()
base_dir = get_base_dir()
checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
model_cfg = load_model_config(checkpoint_dir / "model.zarr")

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

rng = jax.random.key(seed)

model = GPT.init(model_cfg, rng)
model = load_checkpoint(checkpoint_dir / "model.zarr", model)

max_seq_len = model_cfg.sequence_len * 2
pad_token_id = tokenizer.encode_special("<|assistant_end|>")


@jax.jit
def prefill(model, idx, actual_len, kv_cache):
    logits, kv_cache = model.forward(idx, compute_dtype=compute_dtype, kv_cache=kv_cache)
    # fix cache position to actual length (not padded length)
    kv_cache = replace(kv_cache, pos=actual_len)
    # take logit at actual_len - 1 (last non-padded position)
    logits = jax.lax.dynamic_slice(logits, (0, actual_len - 1, 0), (1, 1, logits.shape[-1]))
    return logits, kv_cache

@jax.jit
def generate_next_token(model, idx, mask, kv_cache, key):
    logits, kv_cache = model.forward(idx, mask=mask, compute_dtype=compute_dtype, kv_cache=kv_cache)
    logits = logits[:, -1:, :]
    if temperature is not None:
        pred = jax.random.categorical(key, logits / temperature)
    else:
        pred = jnp.argmax(logits, axis=-1)
    return pred, kv_cache

def generate(idx: list, rng):
    kv_cache = KVCache.init(
        batch_size=1,
        max_seq_len=max_seq_len,
        n_layer=model_cfg.n_layer,
        embed_dim=model_cfg.n_embed,
        n_head=model_cfg.n_head,
        n_kv_head=model_cfg.n_kv_head,
        compute_dtype=compute_dtype,
    )
    actual_len = len(idx)
    # pad to max_seq_len for fixed shape compilation
    idx = idx + [pad_token_id] * (max_seq_len - len(idx))
    idx = jnp.asarray(idx, dtype=jnp.int32)[None, :]
    logits, kv_cache = prefill(model, idx, actual_len, kv_cache)
    if temperature is not None:
        rng, key = jax.random.split(rng)
        pred = jax.random.categorical(key, logits / temperature)
    else:
        pred = jnp.argmax(logits, axis=-1)

    for i in range(max_tokens):
        # our cached k,v are 0s for all positions we haven't filled yet up to our
        # pre-defined cache max_seq_len, so we need to mask these out.
        mask = jnp.arange(kv_cache.k.shape[2]) < kv_cache.pos + 1
        rng, key = jax.random.split(rng)
        next_token, kv_cache = generate_next_token(model, pred, mask, kv_cache, key)
        yield pred[0]
        pred = next_token


user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")


tokens = []
while True:
    try:
        user_input = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print0("\nGoodbye!")
        break
    if not user_input:
        continue

    tokens.append(user_start)
    tokens.extend(tokenizer.encode(user_input))
    tokens.append(user_end)
    tokens.append(assistant_start)
    print0("\nAssistant: ", end="", flush=True)
    for token in generate(tokens, rng):
        tokens.append(token[0])
        if token[0] == assistant_end:
            break
        print0(tokenizer.decode(token), end="", flush=True)

    if token[0] != assistant_end:
        tokens.append(assistant_end)
    if len(tokens) > max_seq_len:
        print0(f"Max sequence len {max_seq_len} exceeded. Goodbye!")
        break

    rng, _ = jax.random.split(rng)
    print0()
