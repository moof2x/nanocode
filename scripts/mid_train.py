import math
import operator
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import trackio

from nanojax import configs
from nanojax.checkpointing import load_checkpoint, load_model_config, save_checkpoint
from nanojax.common import get_base_dir, print0, setup_logging
from nanojax.dataloader import tokenizing_data_loader
from nanojax.eval import evaluate_bpb
from nanojax.gpt import GPT, GPTConfig, calculate_loss, estimate_flops
from nanojax.muon import Muon
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from tasks.dolly import Dolly
from tasks.hhrlhf import HHRLHF
from tasks.mixture import TaskMixture
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

### optimization hparams
batch_size = 32
minibatch_size = 32
num_steps = -1

# learning rates
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

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))] + ["compute_dtype"]
exec(open(os.path.join('nanojax', 'configurator.py')).read()) # overrides from command line
base_dir = get_base_dir()
setup_logging(base_dir / "mid_log.txt")
user_config = {k: globals()[k] for k in config_keys}
for k, v in user_config.items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."
base_checkpoint_dir = base_dir / "base_checkpoints"
checkpoint_dir = base_dir / "mid_checkpoints"
config = load_model_config(base_checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()

max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len* 20 # magic number from nanochat
# distributed setup
world_size = jax.device_count()
accelerator_flops *= world_size
mesh = jax.make_mesh((world_size,), ("b",), axis_types=(jax.sharding.AxisType.Explicit))
jax.set_mesh(mesh)

assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
model = GPT.init(
    config,
    rng
)
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print0(f"{num_params} model parameters")
print0("="*20)

num_flops_per_token = estimate_flops(model)
print0(f"Estimated FLOPs per token: {num_flops_per_token}")

state = Muon.init(model, eps=eps,  wd=wd, wte_lr=wte_lr, lm_head_lr=lm_head_lr, lr=lr)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

model = load_checkpoint(base_checkpoint_dir / "model.zarr", model)

last_step = False
train_ds = TaskMixture([
    SmolTalk("train", seed), # 460K rows
    Dolly("train", seed), # 10K rows
    HHRLHF("train", seed), # 160K rows
    MMLU("train", seed) # 100K rows
], seed)

val_ds = TaskMixture([
  SmolTalk("test", seed),
  HHRLHF("test", seed)                  
], seed)
approx_progress = 0.0
def dataloader(dataset, B, T, split, tokenizer):
    global last_step, approx_progress

    ds_size = len(dataset)
    cursor = 0
    token_buffer = deque()
    B *= jax.local_device_count() # each process collects data for all of it's local accelerators
    needed_tokens = B * T + 1
    while True:
        while len(token_buffer) < needed_tokens:
            sample = dataset[cursor]
            ids, _ = tokenizer.render_conversation(sample, max_tokens=T)
            token_buffer.extend(ids)
            cursor += jax.process_count() 
            if cursor >= ds_size:
                if split == "train":
                    last_step = True
                cursor -= ds_size # we may need to wrap around to fulfill the remaining needed_tokens for the last step
        if split == "train":
            approx_progress = cursor / len(dataset)
        scratch = np.array([token_buffer.popleft() for _ in range(needed_tokens)], dtype=np.int32)
        inputs = jnp.asarray(scratch[:-1]).reshape(B, T)
        targets = jnp.asarray(scratch[1:]).reshape(B, T)
        yield inputs, targets

def dist_dataloader(dataset, batch_size, seq_len, split, tokenizer, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * world_size
    loader = dataloader(dataset, batch_size, seq_len, split,  tokenizer)
    return map(
        partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size, seq_len)),
        loader
    )
    
    
train_loader = dist_dataloader(train_ds, batch_size, max_seq_len, "train", tokenizer, mesh)
get_val_dataloader = lambda:  dist_dataloader(val_ds, minibatch_size, max_seq_len, "test", tokenizer, mesh)
trackio.init(
    project="nanojax",
    config=asdict(config)
)

def get_lr_multiplier(progress):
    # first 80% of training: no decay, then linearly ramp down to 0.
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

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

        loss, grads = grad_fun(idx_, targets_, model, compute_dtype=compute_dtype)
        loss_accm, grads_accm = carry
        return (loss_accm + loss, jax.tree.map(jnp.add, grads_accm, grads)), None
    
    initial_loss = jax.lax.pcast(0.0, ("b",), to="varying")
    (loss, grads), _ = jax.lax.scan(inner_step, (initial_loss, jax.tree.map(jnp.zeros_like, model)), jnp.arange(grad_accm_steps))
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps

    loss = jax.lax.pmean(loss, "b")
    grads = jax.lax.pmean(grads, "b")

    updates, state = state.update(model, grads, lr_multiplier)
    model = jax.tree.map(jnp.subtract, model, updates)
    return model, state, loss

# this time we tokenizer prompts using our chat template
prompts = [
    "What is the capital of France?",
    "What is the chemical symbol of gold?",
    "What is the closest planet to the Sun?",
    "What is the opposite of hot?",
]
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

prompts = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
prompt_idx = [tokenizer.render_conversation(p)[0] for p in prompts]
prompt_idx = [p + [assistant_start] for p in prompt_idx]
prompt_idx = [jnp.asarray(p, dtype=jnp.int32)[None, :] for p in prompt_idx]

total_training_time = 0
step = 0
progress = 0.0
x, y = next(train_loader)
while True:
    lr_multiplier = get_lr_multiplier(progress)

    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    loss = float(loss) # synchronize
    dt = time.perf_counter() - d0
    
    if num_steps > 0:
        approx_progress = step / num_steps
    progress = max(progress, approx_progress)
    pct_done = progress * 100
    
    flops_per_sec = num_flops_per_token * x.size / dt
    mfu = 100 * flops_per_sec / accelerator_flops
    tkps = int(x.size // dt)
    total_training_time += dt

    print0(f"Step: {step} ({pct_done:.2f}%)| Loss: {loss:.3f} | dt: {dt:.2f}s | tkps: {tkps} | mfu: {mfu:.2f} | lr_multiplier: {lr_multiplier:.3f}")
    log_dict = {"loss": loss, "tkps": tkps, "mfu": mfu, "lr_multiplier": lr_multiplier}
    
    if (step % profile_every == 0) or last_step:
        memory_stats = jax.devices()[0].memory_stats()
        used, available = memory_stats.get("peak_bytes_reserved", 0) / 1e9, memory_stats.get("bytes_reservable_limit", 0) / 1e9
        log_dict["peak_bytes_reserved"] = used
        print0(f"\tPeak bytes reserved/limit: {used:.2f}/{available:.2f}")

    if (step % sample_every == 0) or last_step:
        for idx in prompt_idx:
            for i in range(16):
                logits, _ = model.forward(idx, compute_dtype=compute_dtype)
                logits = logits[:, -1, :] # bsv -> bv
                pred = jnp.argmax(logits, axis=-1, keepdims=True)
                idx = jnp.concat([idx, pred], axis=1)
            print0(tokenizer.decode(idx[0]))

    if (step % eval_every == 0) or last_step:
        d0 = time.perf_counter()
        eval_steps = eval_tokens // (minibatch_size * max_seq_len * world_size) 
        val_bpb = evaluate_bpb(model, get_val_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
        print0(f"\tbpb: {float(val_bpb):.4f} | dt: {(time.perf_counter() - d0):.2f}s")
        log_dict["val/bpb"] = val_bpb

    step += 1 
    trackio.log(log_dict)
    if (num_steps > 0 and step >= num_steps) or last_step:
        break

print0(f"Total training time: {(total_training_time/60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state) 
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
trackio.finish()

