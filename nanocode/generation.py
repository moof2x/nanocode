"""
Autoregressive text generation with KV caching.

Generation happens in two phases: a prefill pass processes the full prompt
in parallel and populates the KV cache, then a decode loop generates one
token at a time, reading from and appending to the cache. The prompt is
padded to a fixed length so we get a single compiled trace regardless of
input length.
"""
import jax
import jax.numpy as jnp

from nanocode.gpt import GPT, KVCache


@jax.jit(static_argnums=(4,))
def prefill(
    idx: jax.Array,
    model: GPT,
    seq_len: int,
    kv_cache: KVCache,
    compute_dtype: jnp.dtype
):
    logits, kv_cache = model.forward(idx, compute_dtype=compute_dtype, kv_cache=kv_cache)
    # take logit at actual_len - 1 (last non-padded position)
    logits = jax.lax.dynamic_slice(logits, (0, seq_len - 1, 0), (1, 1, logits.shape[-1]))
    return logits, kv_cache

@jax.jit(static_argnums=(5,))
def generate_next_token(
    idx: jax.Array,
    mask: jax.Array,
    model: GPT,
    kv_cache: KVCache,
    temperature: float,
    compute_dtype: jnp.dtype,
    key
):
    logits, kv_cache = model.forward(idx, mask=mask, compute_dtype=compute_dtype, kv_cache=kv_cache)
    logits = logits[:, -1:, :] # bsv -> bv
    # bv -> b
    if temperature is not None:
        pred = jax.random.categorical(key, logits / temperature)
    else:
        pred = jnp.argmax(logits, axis=-1)
    return pred, kv_cache

def generate(
    idx: list,
    model: GPT,
    max_tokens: int,
    temperature: float,
    compute_dtype: jnp.dtype,
    pad_token_id: int,
    rng,
    assistant_end_id: int | None = None
) -> list[jax.Array]:
    # TODO: this currently creates a new cache from scratch for every call
    # this is fine for single-turn generation but multi-turn interactions become slow
    model_cfg = model.cfg
    max_seq_len = model_cfg.sequence_len * 2
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
    idx = idx + [pad_token_id] * (max_seq_len- len(idx))
    idx = jnp.asarray(idx, dtype=jnp.int32)[None, :]
    logits, kv_cache = prefill(idx, model,  actual_len, kv_cache, compute_dtype)
    kv_cache = kv_cache.forward_pos(actual_len)
    if temperature is not None:
        rng, key = jax.random.split(rng)
        pred = jax.random.categorical(key, logits / temperature)
    else:
        pred = jnp.argmax(logits, axis=-1)
    new_tokens = []
    for i in range(max_tokens):
        # our cached k,v are 0s for all positions we haven't filled yet up to our
        # pre-defined cache max_seq_len, so we need to mask these out.
        mask = jnp.arange(kv_cache.k.shape[2]) < kv_cache.pos + 1
        rng, key = jax.random.split(rng)
        next_token, kv_cache = generate_next_token(pred, mask, model, kv_cache, temperature, compute_dtype, key)
        kv_cache = kv_cache.forward_pos(1)
        new_tokens.append(pred[0])
        pred = next_token
        if assistant_end_id is not None and int(pred[0][0]) == assistant_end_id:
            break
    return new_tokens
