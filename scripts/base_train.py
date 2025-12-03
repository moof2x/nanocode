from nanojax.dataloader import tokenizing_data_loader
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from nanojax.gpt import GPT, calculate_loss, GPTConfig,estimate_flops
from nanojax.adamw import AdamW
from nanojax.muon import Muon
from nanojax import configs
from dataclasses import asdict
import operator
import time
import jax
import jax.numpy as jnp
import math
import trackio
import sys
import os
from nanojax.checkpointing import save_checkpoint
from nanojax.common import get_base_dir

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
base_dir = get_base_dir()
checkpoint_dir = os.path.join()
print(f"Vocab size: {vocab_size}")
command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print(command)
rng = jax.random.key(42)
config = configs.d3_4m
lr = 3e-4
batch_size = 64
minibatch_size = 64
grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."

train_loader = tokenizing_data_loader(batch_size, config.sequence_len, "train", tokenizer)
x, y = next(train_loader)

assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
model = GPT.init(
    config,
    rng
)
print(config)
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
total_tokens = num_params * 20
num_steps = math.ceil(total_tokens / config.sequence_len / batch_size) + 1
expected_loss = 1.8172 + 482.01/(num_params)**0.3478 + 2085.43/(total_tokens)**0.3658
print(f"{num_params} model parameters")
print(f"Training on {total_tokens} tokens over {num_steps} steps")
print(f"Expected final loss: {expected_loss:.4f}")
print("="*20)

num_flops_per_token = estimate_flops(model)
print(f"Estimated FLOPs per token: {num_flops_per_token}")
compute_dtype = jnp.float32
state = Muon.init(model)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

trackio.init(
    project="nanojax",
    config=asdict(config)
)

@jax.jit
def train_step(idx, targets, model, state):
    
    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)

        loss, grads = grad_fun(idx_, targets_, model, compute_dtype)
        loss_accm, grads_accm = carry
        return (loss_accm + loss, jax.tree.map(jnp.add, grads_accm, grads)), None

    (loss, grads), _ = jax.lax.scan(inner_step, (0.0, jax.tree.map(jnp.zeros_like, model)), jnp.arange(grad_accm_steps))
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps
    
    updates, state = state.update(model, grads, lr, step)
    model = jax.tree.map(lambda p, u: p - u, model, updates)
    return model, state, loss

prompts = [
    ["The capital of France is"],
    ["Einstein's special theory of relatively states that energy"],
    ["The closest planet to the Sun is"]
]
prompt_idx = [tokenizer.encode(p, prepend=tokenizer.get_bos_token_id()) for p in prompts]
prompt_idx = [jnp.asarray(p, dtype=jnp.int32) for p in prompt_idx]
step = 1    
while True:
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    log_dict = {"loss": float(loss)}
    print(f"Step {step}/{num_steps} | Loss: {loss:.3f} / {expected_loss:.3f} ")
    if step % 10 == 0:
        # log profiling every now and then
        jax.block_until_ready(loss)
        dt = time.perf_counter() - d0
        flops_per_sec = num_flops_per_token * x.size / dt
        mfu = 100 * flops_per_sec / 11.15e12# 2080 super FLOPs/sec
        print(f"\tdt: {dt:.3f}s | tkps: {int(x.size // dt)} | mfu: {mfu:.2f}")
        print(f"\tTokens seen: {x.size * step} / {total_tokens} ({((x.size * step / total_tokens) * 100):.2f}%)")
        print(f"\tEstimated time remaining: {(((num_steps - step) * dt)/60):.1f} min")
        log_dict["tkps"] = int(x.size // dt)

    if step % 50== 0:
        for idx in prompt_idx:
            for i in range(10):
                logits = model.forward(idx, compute_dtype)[:, -1, :] # bsv -> bv
                pred = jnp.argmax(logits, axis=-1, keepdims=True)
                idx = jnp.concat([idx, pred], axis=1)
            print(tokenizer.decode(idx[0]))

    step += 1 
    trackio.log(log_dict)
    if step == num_steps:
        break


trackio.finish()
