import math
import operator
import os
import sys
import time
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
import trackio

from nanojax import configs
from nanojax.adamw import AdamW
from nanojax.checkpointing import save_checkpoint
from nanojax.common import get_base_dir, print0, setup_logging, init_distributed
from nanojax.dataloader import get_distributed_dataloader
from nanojax.eval import evaluate_bpb
from nanojax.gpt import GPT, GPTConfig, calculate_loss, estimate_flops
from nanojax.muon import Muon
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from nanojax.generation import generate

# distributed setup
world_size, mesh = init_distributed()

config = configs.d3
### optimization hparams
batch_size = 32
minibatch_size = 32
num_steps = -1
grad_clip = 1.0

# learning rates/scheduling
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0
eps = 1e-10
wd = 0.0
wte_lr = 0.2
lm_head_lr = 0.004
lr = 0.02

### misc
seed = 42
accelerator_flops = 11.15e12 # 2080 super FLOPs/sec
compute_dtype = jnp.bfloat16

### training loop control
sample_every = 50
eval_every = 50
profile_every = 500

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))] + ["config", "compute_dtype"]
exec(open(os.path.join('nanojax', 'configurator.py')).read()) # overrides from command line
base_dir = get_base_dir()
setup_logging(base_dir / "base_log.txt")
user_config = {k: globals()[k] for k in config_keys}
for k, v in user_config.items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."
max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len* 20 # magic number from nanochat
checkpoint_dir = base_dir / "base_checkpoints"
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
print0(f"Vocab size: {vocab_size}")

# distributed setup
accelerator_flops *= world_size
print(f"World size {world_size}")

train_loader = get_distributed_dataloader(batch_size, max_seq_len, "train", tokenizer, mesh)
get_val_dataloader = lambda: get_distributed_dataloader(minibatch_size, max_seq_len, "val", tokenizer, mesh)

model = GPT.init(
    config,
    rng
)
    
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print0(f"{num_params} model parameters")
if num_steps < 0:
    total_tokens = num_params * 20
    num_steps = math.ceil(total_tokens / max_seq_len / (batch_size * world_size)) + 1
else:
    total_tokens = num_steps * max_seq_len * (batch_size * world_size)

print0(f"Training on {total_tokens} tokens over {num_steps} steps")
print0("="*20)

num_flops_per_token = estimate_flops(model)
print0(f"Estimated FLOPs per token: {num_flops_per_token}")

state = Muon.init(model, eps=eps,  wd=wd, wte_lr=wte_lr, lm_head_lr=lm_head_lr, lr=lr)
grad_fn = jax.value_and_grad(calculate_loss, argnums=2)

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

# we construct a PartitionSpec with default behaviour indicating to replicate for our model and optimizer states
model_spec = jax.tree.map(lambda _: jax.P(), model)
state_spec = jax.tree.map(lambda _: jax.P(), state)

in_specs = (jax.P("b", None), jax.P("b", None), model_spec, state_spec)
out_specs = (model_spec, state_spec, jax.P())

@jax.jit(donate_argnums=(2, 3))
@jax.shard_map(in_specs=in_specs, out_specs=out_specs, mesh=mesh)
def train_step(idx, targets, model, state):
    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)

        loss, grads = grad_fn(idx_, targets_, model, compute_dtype=compute_dtype)
        loss_accm, grads_accm = carry
        return (loss_accm + loss, jax.tree.map(jnp.add, grads_accm, grads)), None

    initial_loss = jax.lax.pcast(0.0, ("b",), to="varying")
    (loss, grads), _ = jax.lax.scan(inner_step, (initial_loss, jax.tree.map(jnp.zeros_like, model)), jnp.arange(grad_accm_steps))
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps
    
    loss = jax.lax.pmean(loss, "b")
    grads = jax.lax.pmean(grads, "b")

    # grad norm clipping
    global_norm = jnp.sqrt(jax.tree.reduce(operator.add, jax.tree.map(lambda g: jnp.sum(jax.lax.square(g)), grads)))
    grad_scale_value = jnp.minimum(1.0, grad_clip / (global_norm + 1e-6))
    grads = jax.tree.map(lambda g: g * grad_scale_value, grads)
        
    updates, state = state.update(model, grads, lr_multiplier)
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

total_training_time = 0
step = 0
x, y = next(train_loader)
while True:
    last_step = (step + 1) == num_steps
    lr_multiplier = get_lr_multiplier(step)
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    loss = float(loss) # synchronise
    dt = time.perf_counter() - d0

    # profiling info
    flops_per_sec = num_flops_per_token * x.size / dt
    mfu = 100 * flops_per_sec / accelerator_flops
    tkps = int(x.size // dt)
    eta = ((num_steps - step) * dt) / 60
    total_training_time += dt

    print0(f"Step: {step}/{num_steps} | Loss: {loss:.3f} | dt: {dt:.2f}s | | tkps: {tkps} | mfu: {mfu:.2f} | min ETA: {eta:.1f} min | lr_multiplier: {lr_multiplier:.3f}")
    log_dict = {"loss": loss, "tkps": tkps, "mfu": mfu,  "lr_multiplier": lr_multiplier}

    if (step % profile_every == 0) or last_step:
        memory_stats = jax.devices()[0].memory_stats() or {}
        used, available = memory_stats.get("peak_bytes_reserved", 0) / 1e9, memory_stats.get("bytes_reservable_limit", 0) / 1e9
        log_dict["peak_bytes_reserved"] = used
        print0(f"\tPeak bytes reserved/limit: {used:.2f}/{available:.2f}")

    if (step % sample_every == 0) or last_step:
        for idx in prompt_idx:
            new_tokens = generate(
              idx,
              model,
              max_tokens=16,
              temperature=None,
              compute_dtype=compute_dtype,
              pad_token_id=tokenizer.encode_special("<|assistant_end|>"),
              rng=rng
            )
            print0("\t" + tokenizer.decode(idx + [int(t[0]) for t in new_tokens]))

    if (step % eval_every == 0) or last_step:
        d0 = time.perf_counter()
        eval_steps = eval_tokens // (minibatch_size * max_seq_len * world_size) 
        val_bpb = evaluate_bpb(model, iter(get_val_dataloader()), eval_steps, token_bytes, compute_dtype, mesh)
        print0(f"\tbpb: {float(val_bpb):.4f} | dt: {(time.perf_counter() - d0):.2f}s")
        log_dict["val/bpb"] = val_bpb

    step += 1 
    trackio.log(log_dict)
    if step == num_steps:
        break

print0(f"Total training time: {(total_training_time/60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
trackio.finish()
