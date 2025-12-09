import math

import jax
import jax.numpy as jnp

from nanojax.gpt import calculate_loss


def evaluate_bpb(model, dataloader, steps, token_bytes, compute_dtype):
    @jax.jit
    def eval_step(model, x, y, token_bytes):
        loss = calculate_loss(x, y, model, compute_dtype, reduce=False).flatten()
        y = y.flatten()
        num_bytes = token_bytes[y]
        return (loss * (num_bytes > 0)).sum(), num_bytes.sum()
    
    total_nats = jnp.asarray(0)
    total_bytes = jnp.asarray(0, dtype=jnp.int32)
    for _ in range(steps):
        x, y = next(dataloader)
        cur_nats, cur_bytes = eval_step(model, x, y, token_bytes)
        total_nats += cur_nats 
        total_bytes += cur_bytes 
    if total_bytes == 0:
        return float("inf")
    return float(total_nats / (math.log(2) * total_bytes))
