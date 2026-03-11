"""Pre-training loop. Adapted from nanochat's base_train.py but rewritten for JAX."""
import argparse
import math
import operator
import os
import sys
import time

import jax
import jax.numpy as jnp
from nanocode.checkpointing import save_checkpoint
from nanocode.common import (
    get_base_dir,
    get_model_dir,
    init_distributed,
    print0,
    setup_logging,
)
from nanocode.dataloader import get_distributed_dataloader
from nanocode.eval import evaluate_bpb
from nanocode.generation import generate
from nanocode.gpt import GPT, calculate_loss, estimate_flops
from nanocode.muon import Muon
from nanocode.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_model

from nanocode import configs

# distributed setup
world_size, mesh = init_distributed()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='d3', choices=configs.CONFIGS)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--minibatch-size', type=int, default=32)
parser.add_argument('--num-steps', type=int, default=-1)
parser.add_argument('--warmup-ratio', type=float, default=0.0)
parser.add_argument('--warmdown-ratio', type=float, default=0.4)
parser.add_argument('--final-lr-frac', type=float, default=0.0)
parser.add_argument('--eps', type=float, default=1e-10)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--wte-lr', type=float, default=0.3)
parser.add_argument('--lm-head-lr', type=float, default=0.004)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--code-ratio', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--param-data-ratio', type=int, default=8)
parser.add_argument('--accelerator-flops', type=float, default=918e12)
parser.add_argument('--compute-dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--attn-impl', type=str, default='splash', choices=['splash', 'eager'])
parser.add_argument('--sample-every', type=int, default=50)
parser.add_argument('--eval-every', type=int, default=50)
parser.add_argument('--core-metric-every', type=int, default=2000)
parser.add_argument('--core-metric-max-per-task', type=int, default=500)
parser.add_argument('--profile-every', type=int, default=500)
args = parser.parse_args()

config = configs.CONFIGS[args.config]
### optimization hparams
batch_size = args.batch_size
minibatch_size = args.minibatch_size
num_steps = args.num_steps
# learning rates/scheduling
warmup_ratio = args.warmup_ratio
warmdown_ratio = args.warmdown_ratio
final_lr_frac = args.final_lr_frac
eps = args.eps
wd = args.wd
wte_lr = args.wte_lr
lm_head_lr = args.lm_head_lr
lr = args.lr
### misc
code_ratio = args.code_ratio
seed = args.seed
param_data_ratio = args.param_data_ratio
accelerator_flops = args.accelerator_flops # TPU v6e
compute_dtype = jnp.bfloat16 if args.compute_dtype == 'bfloat16' else jnp.float32
attn_impl = args.attn_impl
### training loop control
sample_every = args.sample_every
eval_every = args.eval_every
core_metric_every = args.core_metric_every
core_metric_max_per_task = args.core_metric_max_per_task
profile_every = args.profile_every

base_dir = get_base_dir()
model_dir = get_model_dir()
setup_logging(model_dir / "base_log.txt")
for k, v in vars(args).items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."
max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len * 40 * world_size
checkpoint_dir = model_dir / "base_checkpoints"
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOCODE_BASE_DIR={base_dir} MODEL_TAG={os.environ.get('MODEL_TAG', '')} {command}")

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
print0(f"Vocab size: {vocab_size}")

accelerator_flops *= world_size
print0(f"World size: {world_size}")

train_loader = get_distributed_dataloader(batch_size, max_seq_len, "train", tokenizer, code_ratio, mesh)
get_fwe_val_dataloader = lambda: get_distributed_dataloader(minibatch_size, max_seq_len, "val", tokenizer, 0.0, mesh)
get_sv2_val_dataloader = lambda: get_distributed_dataloader(minibatch_size, max_seq_len, "val", tokenizer, 1.0, mesh)

model = GPT.init(
    config,
    rng,
    attn_impl=attn_impl
)
    
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print0(f"{num_params/1e6}M model parameters")
for name, layer in [("wte", model.wte), ("h", model.h),("lm_head", model.lm_head)]:
    layer_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, layer))
    print0(f"  {layer_params/1e6}M {name} parameters")


if num_steps < 0:
    total_tokens = num_params * param_data_ratio
    num_steps = math.ceil(total_tokens / max_seq_len / (batch_size * world_size)) + 1
else:
    total_tokens = num_steps * max_seq_len * (batch_size * world_size)

print0(f"Training on {total_tokens} tokens over {num_steps} steps")
print0("="*20)

num_flops_per_token = estimate_flops(model)
print0(f"Estimated FLOPs per token: {num_flops_per_token}")

state = Muon.init(model, eps=eps,  wd=wd, wte_lr=wte_lr, lm_head_lr=lm_head_lr, lr=lr)
grad_fn = jax.value_and_grad(calculate_loss, argnums=2)

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

in_specs = (jax.P("b", None), jax.P("b", None), model_spec, state_spec, jax.P())
out_specs = (model_spec, state_spec, jax.P())

@jax.jit(donate_argnums=(2, 3))
@jax.shard_map(in_specs=in_specs, out_specs=out_specs, mesh=mesh, check_vma=False)
def train_step(idx, targets, model, state, lr_multiplier):
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
    lr_multiplier = jnp.array(get_lr_multiplier(step))
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state, lr_multiplier)
    x, y = next(train_loader)
    loss = float(loss) # synchronise
    dt = time.perf_counter() - d0

    # profiling info
    eta = -1
    if step > 1:
        total_training_time += dt
        average_time_per_step = total_training_time / step
        eta = ((num_steps - step) * average_time_per_step) / 60
    flops_per_sec = num_flops_per_token * x.size / dt
    mfu = 100 * flops_per_sec / accelerator_flops
    tkps = int(x.size // dt)

    print0(f"Step: {step}/{num_steps} | Loss: {loss:.3f} | dt: {dt:.2f}s | | tkps: {tkps} | mfu: {mfu:.2f} | ETA: {eta:.1f} min | lr_multiplier: {lr_multiplier:.3f}")

    if (step % profile_every == 0) or last_step:
        memory_stats = jax.local_devices()[0].memory_stats() or {}
        used, available = memory_stats.get("peak_bytes_reserved", 0) / 1e9, memory_stats.get("bytes_reservable_limit", 0) / 1e9
        print0(f"\tPeak bytes reserved/limit: {used:.2f}/{available:.2f}")
    
    if step > 0:
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
            fwe_bpb = evaluate_bpb(model, get_fwe_val_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
            sv2_bpb = evaluate_bpb(model, get_sv2_val_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
            print0(f"\tfwe_bpb: {float(fwe_bpb):.4f} | sv2_bpb: {float(sv2_bpb):.4f} | avg_bpb: {float((fwe_bpb + sv2_bpb) / 2):.4f} | dt: {(time.perf_counter() - d0):.2f}s")

        # only run CORE on rank 0 in multi-node
        if ((step % core_metric_every == 0) or last_step) and jax.process_index() == 0:
            d0 = time.perf_counter()
            core_results = evaluate_model(model, tokenizer, minibatch_size * 2, compute_dtype, mesh, max_per_task=core_metric_max_per_task)
            core_metric = core_results['core_metric']
            dt = time.perf_counter() - d0
            print0(f"  CORE metric: {core_metric:.4f} | dt: {dt:.2f}s")
        jax.experimental.multihost_utils.sync_global_devices("CORE")

    step += 1
    if step == num_steps:
        break

print0(f"Total training time: {(total_training_time/60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
