import math
import operator
import os
import sys
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import trackio

from nanojax import configs
from nanojax.adamw import AdamW
from nanojax.checkpointing import save_checkpoint
from nanojax.common import get_base_dir
from nanojax.dataloader import tokenizing_data_loader
from nanojax.eval import evaluate_bpb
from nanojax.gpt import GPT, GPTConfig, calculate_loss, estimate_flops
from nanojax.muon import Muon
from nanojax.tokenizer import get_token_bytes, get_tokenizer

config = configs.d3
### optimization hparams
lr = 3e-4
batch_size = 32
minibatch_size = 32
num_steps = -1
warmup_ratio = 0.0
warmdown_ratio = 0.2
compute_dtype = jnp.float32

### misc
seed = 42
accelerator_flops = 11.15e12 # 2080 super FLOPs/sec

### training loop control
profile_every = 100
sample_every = 50
eval_every = 50

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))] + ["config", "compute_dtype"]
exec(open(os.path.join('nanojax', 'configurator.py')).read()) # overrides from command line 
user_config = {k: globals()[k] for k in config_keys} 
command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print(command)
for k, v in user_config.items():
    print(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."
max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len* 20 # magic number from nanochat

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"

base_dir = get_base_dir()
checkpoint_dir = base_dir / "base_checkpoints"
print(f"Vocab size: {vocab_size}")
rng = jax.random.key(seed)

train_loader = tokenizing_data_loader(batch_size, max_seq_len, "train", tokenizer)
get_val_dataloader = lambda: tokenizing_data_loader(batch_size, max_seq_len, "val", tokenizer)
x, y = next(train_loader)

model = GPT.init(
    config,
    rng
)
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
if num_steps < 0:
    total_tokens = num_params * 20
    num_steps = math.ceil(total_tokens / max_seq_len / batch_size) + 1
else:
    total_tokens = num_steps * max_seq_len * batch_size
print(f"{num_params} model parameters")
print(f"Training on {total_tokens} tokens over {num_steps} steps")
print("="*20)

num_flops_per_token = estimate_flops(model)
print(f"Estimated FLOPs per token: {num_flops_per_token}")
state = Muon.init(model)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

trackio.init(
    project="nanojax",
    config=asdict(config)
)

def get_lr_multiplier(step):
    warmup_iters = round(warmup_ratio * num_steps)
    warmdown_iters = round(warmdown_ratio * num_steps)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= num_steps - warmdown_iters:
        return 1.0
    else:
        progress = (num_steps - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

    
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
    
    updates, state = state.update(model, grads, lr_multiplier, step + 1)
    model = jax.tree.map(jnp.subtract, model, updates)
    return model, state, loss

prompts = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "The closest planet to the Sun is",
    "The opposite of hot is",
    "The second-last day of the week is"
]
prompt_idx = [tokenizer.encode(p, prepend=tokenizer.get_bos_token_id()) for p in prompts]
prompt_idx = [jnp.asarray(p, dtype=jnp.int32)[None, :] for p in prompt_idx]
eval_forward = jax.jit(model.forward, static_argnums=1)

step = 0
while True:
    lr_multiplier = get_lr_multiplier(step)
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    log_dict = {"loss": float(loss)}
    print(f"Step {step}/{num_steps} | Loss: {loss:.3f}")
    if step % profile_every == 0:
        # log profiling every now and then
        jax.block_until_ready(loss)
        dt = time.perf_counter() - d0
        flops_per_sec = num_flops_per_token * x.size / dt
        mfu = 100 * flops_per_sec  /accelerator_flops
        print(f"\tdt: {dt:.3f}s | tkps: {int(x.size // dt)} | mfu: {mfu:.2f}")
        print(f"\tTokens seen: {x.size * step} / {total_tokens} ({((x.size * step / total_tokens) * 100):.2f}%)")
        print(f"\tEstimated time remaining: {(((num_steps - step) * dt)/60):.1f} min")
        log_dict["tkps"] = int(x.size // dt)

    if step % sample_every == 0:
        for idx in prompt_idx:
            for i in range(16):
                logits = eval_forward(idx, compute_dtype)[:, -1, :] # bsv -> bv
                pred = jnp.argmax(logits, axis=-1, keepdims=True)
                idx = jnp.concat([idx, pred], axis=1)
            print(tokenizer.decode(idx[0]))

    if step % eval_every == 0:
        d0 = time.perf_counter()
        eval_steps = eval_tokens // (minibatch_size * max_seq_len) 
        val_bpb = evaluate_bpb(model, iter(get_val_dataloader()), eval_steps, token_bytes, compute_dtype)
        print(f"bpb: {val_bpb:.2f} | dt: {(time.perf_counter() - d0):.2f}s")
    step += 1 
    trackio.log(log_dict)
    if step == num_steps:
        break

save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
trackio.finish()
