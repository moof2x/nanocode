import math

import jax
import jax.numpy as jnp

from nanojax.gpt import calculate_loss


def evaluate_bpb(model, dataloader, steps, token_bytes, compute_dtype, mesh):
    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            jax.tree.map(lambda _: jax.P(), model),
            jax.P("b", None),  
            jax.P("b", None),    
            jax.P(),  
        ),
        out_specs=(jax.P(), jax.P()),
        check_vma=False
    )
    def eval_step(model, x, y, token_bytes):
        loss = calculate_loss(x, y, model, compute_dtype=compute_dtype, reduce=False).flatten()
        y = y.flatten()
        valid_targets = y >= 0
        safe_y = jnp.where(valid_targets, y, 0)
        num_bytes = jnp.where(valid_targets, token_bytes[safe_y], 0)

        local_nats = (loss * (num_bytes > 0)).sum()
        local_bytes = num_bytes.sum()
        
        total_nats = jax.lax.psum(local_nats, "b")
        total_bytes = jax.lax.psum(local_bytes, "b")
        
        return total_nats, total_bytes
    
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
