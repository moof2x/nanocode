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
from nanojax.common import get_base_dir, print0
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

checkpoint = "mid"

### optimization hparams
batch_size = 32
minibatch_size = 32
num_steps = -1
num_epochs = 1

# learning rates
eps = 1e-10
wd = 0.0
wte_lr = 0.2
lm_head_lr = 0.004
lr = 0.02
init_lr_frac = 0.02

### misc
seed = 42
accelerator_flops = 11.15e12  # 2080 super FLOPs/sec
compute_dtype = jnp.bfloat16

### training loop control
sample_every = 50
eval_every = 50
profile_every = 500

config_keys = [k for k,v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))] + ["compute_dtype"]
exec(open(os.path.join("nanojax", "configurator.py")).read()) # overrides from command line
user_config = {k: globals()[k] for k in config_keys}
for k, v in user_config.items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
base_dir = get_base_dir()
base_checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
checkpoint_dir = base_dir / "sft_checkpoints"
config = load_model_config(base_checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len * 20  # magic number from nanochat
# distributed setup
world_size = jax.device_count()
accelerator_flops *= world_size
mesh = jax.make_mesh((world_size,), ("b",), axis_types=(jax.sharding.AxisType.Explicit))
jax.set_mesh(mesh)

assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
model = GPT.init(config, rng)
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print0(f"{num_params} model parameters")
print0("=" * 20)

num_flops_per_token = estimate_flops(model)
print0(f"Estimated FLOPs per token: {num_flops_per_token}")

state = Muon.init(
    model,
    eps=eps,
    wd=wd,
    wte_lr=wte_lr * init_lr_frac,
    lm_head_lr=lm_head_lr * init_lr_frac,
    lr=lr * init_lr_frac,
)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

model = load_checkpoint(base_checkpoint_dir / "model.zarr", model)

train_ds = TaskMixture([
    SmolTalk("train[:5%]", seed),  # 460*0.05=23K rows
    Dolly("train[:10%]", seed),  # 10*0.1=1K rows
    HHRLHF("train[:5%]", seed),  # 160*0.05=8K rows
    MMLU("train[:5%]", seed),  # 100*0.05=5K rows
], seed)

val_ds = TaskMixture([SmolTalk("test", seed), HHRLHF("test", seed)], seed)


def dataloader(dataset, B, T, tokenizer):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    B *= jax.local_device_count() # each process collects data for all of it's local accelerators

    def collate(batch):
        # we always pad or truncate to max seq len
        inputs = np.full((B, T), pad_token_id, dtype=np.int32)
        targets = np.full_like(inputs, -1)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            inputs[i, : n - 1] = ids[:-1]
            row_targets = ids[1:n]
            row_targets[mask[1:n] == 0] = -1
            targets[i, : n - 1] = row_targets
        return jnp.asarray(inputs), jnp.asarray(targets)

    batch = []
    while True:
        for i in range(jax.process_index(), len(dataset), jax.process_count()):
            batch.append(tokenizer.render_conversation(dataset[i], max_tokens=T + 1))
            if len(batch) == batch_size:
                yield collate(batch)
                batch = []


def dist_dataloader(dataset, batch_size, seq_len, tokenizer, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * world_size
    loader = dataloader(dataset, batch_size, seq_len, tokenizer)
    return map(partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size, seq_len)), loader)


train_loader = dist_dataloader(train_ds, batch_size, max_seq_len, tokenizer, mesh)
get_val_dataloader = lambda: dist_dataloader(val_ds, minibatch_size, max_seq_len, tokenizer, mesh)
trackio.init(project="nanojax", config=asdict(config))

steps_per_epoch = len(train_ds) // (batch_size * world_size)
if num_steps < 0:
    num_steps = steps_per_epoch * num_epochs


def get_lr_multiplier(step):
    # linear lr decay
    return 1 - step / num_steps


# we construct a PartitionSpec with default behaviour indicating to replicate for our model and optimizer states
model_spec = jax.tree.map(lambda _: jax.P(), model)
state_spec = jax.tree.map(lambda _: jax.P(), state)

in_specs = (jax.P("b", None), jax.P("b", None), model_spec, state_spec)
out_specs = (model_spec, state_spec, jax.P(), jax.P())


@jax.jit(donate_argnums=(2, 3))
@jax.shard_map(in_specs=in_specs, out_specs=out_specs, mesh=mesh)
def train_step(idx, targets, model, state):
    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)

        loss, grads = grad_fun(idx_, targets_, model, ignore_idx=-1, compute_dtype=compute_dtype)
        loss_accm, grads_accm = carry
        return (loss_accm + loss, jax.tree.map(jnp.add, grads_accm, grads)), None

    initial_loss = jax.lax.pcast(0.0, ("b",), to="varying")
    (loss, grads), _ = jax.lax.scan(inner_step, (initial_loss, jax.tree.map(jnp.zeros_like, model)), jnp.arange(grad_accm_steps))
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps

    loss = jax.lax.pmean(loss, "b")
    grads = jax.lax.pmean(grads, "b")
    valid_tokens = jnp.sum(targets >= 0)
    total_tokens = jax.lax.psum(valid_tokens, "b")
    # step is accessed globally as it would trigger recompiles if passed to our JIT-ed step
    updates, state = state.update(model, grads, lr_multiplier, step + 1)
    model = jax.tree.map(jnp.subtract, model, updates)
    return model, state, loss, total_tokens


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
x, y = next(train_loader)
for step in range(num_steps):
    last_step = (step + 1) == num_steps
    lr_multiplier = get_lr_multiplier(step)

    d0 = time.perf_counter()
    model, state, loss, total_tokens = train_step(x, y, model, state)
    x, y = next(train_loader)
    loss = float(loss)  # synchronize
    total_tokens = int(total_tokens)
    dt = time.perf_counter() - d0

    flops_per_sec = num_flops_per_token * total_tokens / dt
    mfu = 100 * flops_per_sec / accelerator_flops
    tkps = int(total_tokens // dt)
    eta = ((num_steps - step) * dt) / 60
    total_training_time += dt

    print0(f"Step: {step}/{num_steps} | Loss: {loss:.3f} | dt: {dt:.2f}s | tkps: {tkps} | mfu: {mfu:.2f} | min ETA {eta:.1f} | lr_multiplier: {lr_multiplier:.3f}")
    log_dict = {"loss": loss, "tkps": tkps, "mfu": mfu, "lr_multiplier": lr_multiplier}

    if step % profile_every == 0:
        memory_stats = jax.devices()[0].memory_stats() or {}
        used, available = memory_stats.get("peak_bytes_reserved", 0) / 1e9, memory_stats.get("bytes_reservable_limit", 0) / 1e9
        log_dict["peak_bytes_reserved"] = used
        print0(f"\tPeak bytes reserved/limit: {used:.2f}/{available:.2f}")

    if (step % sample_every == 0) or last_step:
        for idx in prompt_idx:
            for i in range(16):
                logits, _ = model.forward(idx, compute_dtype=compute_dtype)
                logits = logits[:, -1, :]  # bsv -> bv
                pred = jnp.argmax(logits, axis=-1, keepdims=True)
                idx = jnp.concat([idx, pred], axis=1)
            print0(tokenizer.decode(idx[0]))

    if (step % eval_every == 0) or last_step:
        d0 = time.perf_counter()
        eval_steps = eval_tokens // (minibatch_size * max_seq_len * world_size)
        val_bpb = evaluate_bpb(model, get_val_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
        print0(f"\tbpb: {float(val_bpb):.4f} | dt: {(time.perf_counter() - d0):.2f}s")
        log_dict["val/bpb"] = val_bpb

    trackio.log(log_dict)

print0(f"Total training time: {(total_training_time / 60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
trackio.finish()
