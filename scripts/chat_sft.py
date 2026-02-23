import operator
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from nanojax.checkpointing import load_checkpoint, load_model_config, save_checkpoint
from nanojax.common import get_base_dir, init_distributed, print0, setup_logging
from nanojax.eval import evaluate_bpb
from nanojax.generation import generate
from nanojax.gpt import GPT, calculate_loss, estimate_flops
from nanojax.muon import Muon
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from scripts.chat_eval import run_chat_eval
from tasks.dolly import Dolly
from tasks.sequence import TaskSequence
from tasks.mixture import TaskMixture
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.json_dataset import JSONDataset
from tasks.gsm8k import GSM8K
from tasks.dataset import Dataset

# distributed setup
world_size, mesh = init_distributed()

checkpoint = "base"

### optimization hparams
batch_size = 32
minibatch_size = 32
num_steps = -1
num_epochs = 1

# learning rates
eps = 1e-10
wd = 0.0
wte_lr = 0.3
lm_head_lr = 0.004
lr = 0.02
init_lr_frac = 1

### misc
seed = 42
accelerator_flops = 11.15e12  # 2080 super FLOPs/sec
compute_dtype = jnp.bfloat16
attn_impl = "splash"

### training loop control
sample_every = 50
eval_every = 50
profile_every = 500

config_keys = [k for k,v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))] + ["compute_dtype"]
exec(open(os.path.join("nanojax", "configurator.py")).read()) # overrides from command line
base_dir = get_base_dir()
setup_logging(base_dir / "chat_sft_log.txt")
user_config = {k: globals()[k] for k in config_keys}
for k, v in user_config.items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
base_checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
checkpoint_dir = base_dir / "sft_checkpoints"
config = load_model_config(base_checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len * 20  # magic number from nanochat
accelerator_flops *= world_size
print0(f"World size: {world_size}")

assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
model = GPT.init(config, rng, attn_impl)

num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print0(f"{num_params/1e6}M model parameters")
for name, layer in [("wte", model.wte), ("h", model.h),("lm_head", model.lm_head)]:
    num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, layer))
    print0(f"  {num_params/1e6}M {name} parameters")

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
grad_fn = jax.value_and_grad(calculate_loss, argnums=2)

model = load_checkpoint(base_checkpoint_dir / "model.zarr", model)

# our curriculum will begin with general chat/instruction following and graduate to code+agentic tasks
train_ds = TaskMixture(
    [
        ### general chat templating and instruction following
        Dataset("QuixiAI/SystemChat-2.0", "messages", "train[:20%]", seed), # teaches the model to follow system prompts
        Dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", "messages", "train_sft", seed), # 2 epochs of regular conversations
        Dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", "messages", "train_sft", seed),
        Dataset("HuggingFaceH4/no_robots", "messages", "train", seed), # 2 epochs of instruction following
        Dataset("HuggingFaceH4/no_robots", "messages", "train", seed), 
        JSONDataset("rollouts/all_train.jsonl"),  # 2 epochs of simple-ish tool calling rollouts (~100k)
        JSONDataset("rollouts/all_train.jsonl"),  # 2 epochs of simple-ish tool calling rollouts (~100k)
        JSONDataset("rollouts/rollouts_train.jsonl"),  # 5 epochs of long-context rollouts at 2K each
        JSONDataset("rollouts/rollouts_train.jsonl"),
        JSONDataset("rollouts/rollouts_train.jsonl"),
        JSONDataset("rollouts/rollouts_train.jsonl"),
        JSONDataset("rollouts/rollouts_train.jsonl"),
    ],
    seed
)

val_ds = TaskMixture([
    MMLU("all", "test", seed),
    GSM8K(subset="main", split="test", seed=seed),
    JSONDataset("rollouts/all_test.jsonl"),
    JSONDataset("rollouts/rollouts_test.jsonl"),
    Dataset("HuggingFaceH4/no_robots", "messages", "test", seed),
    Dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", "messages", "test_sft", seed),
], seed)


def dataloader(dataset, B, T, tokenizer, buffer_size=100):
    bos_token = tokenizer.get_bos_token_id()
    B *= jax.local_device_count() # each process collects data for all of its local accelerators
    row_capacity = T + 1

    conv_buffer = []
    cursor = jax.process_index()

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            if cursor >= len(dataset):
                cursor = jax.process_index()
            conversation = dataset[cursor]
            try:
                ids, mask = tokenizer.render_conversation(conversation, max_tokens=T + 1)
            except:
                import ipdb
                ipdb.set_trace()
                x = 10
            conv_buffer.append((ids, mask))
            cursor += jax.process_count()

    def collate(batch):
        # we always pack/pad/truncate to max seq len
        inputs = np.zeros((B, T), dtype=np.int32)
        targets = np.full((B, T), -1, dtype=np.int32)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            inputs[i, :n-1] = ids[:-1]
            row_targets = ids[1:n]
            row_mask = mask[1:n]
            row_targets[row_mask == 0] = -1
            targets[i, :n-1] = row_targets

        return jnp.asarray(inputs), jnp.asarray(targets)

    while True:
        batch = []

        for _ in range(B):
            row_ids = []
            row_mask = []

            while len(row_ids) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                if not conv_buffer:
                    remaining = row_capacity - len(row_ids)
                    row_ids.extend([bos_token] * remaining)
                    row_mask.extend([0] * remaining)
                    break

                remaining = row_capacity - len(row_ids)

                best_idx = -1
                best_len = 0
                for i, (conv_ids, conv_mask) in enumerate(conv_buffer):
                    conv_len = len(conv_ids)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv_ids, conv_mask = conv_buffer.pop(best_idx)
                    row_ids.extend(conv_ids)
                    row_mask.extend(conv_mask)
                else:
                    row_ids.extend([bos_token] * remaining)
                    row_mask.extend([0] * remaining)
                    break

            batch.append((row_ids[:row_capacity], row_mask[:row_capacity]))

        yield collate(batch)


def dist_dataloader(dataset, batch_size, seq_len, tokenizer, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * world_size
    loader = dataloader(dataset, batch_size, seq_len, tokenizer)
    return map(partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size, seq_len)), loader)


train_loader = dist_dataloader(train_ds, batch_size, max_seq_len, tokenizer, mesh)
get_val_dataloader = lambda: dist_dataloader(val_ds, minibatch_size, max_seq_len, tokenizer, mesh)

steps_per_epoch = len(train_ds) // (batch_size * world_size)
if num_steps < 0:
    num_steps = steps_per_epoch * num_epochs


def get_lr_multiplier(step):
    # linear lr decay - flat for first 98% then ramp down
    return 1
    progress = step / num_steps
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2


# we construct a PartitionSpec with default behaviour indicating to replicate for our model and optimizer states
model_spec = jax.tree.map(lambda _: jax.P(), model)
state_spec = jax.tree.map(lambda _: jax.P(), state)

in_specs = (jax.P("b", None), jax.P("b", None), model_spec, state_spec, jax.P())
out_specs = (model_spec, state_spec, jax.P(), jax.P())


@jax.jit(donate_argnums=(2, 3))
@jax.shard_map(in_specs=in_specs, out_specs=out_specs, mesh=mesh, check_vma=False)
def train_step(idx, targets, model, state, lr_multiplier):
    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)
        # loss at grad accm step i is L_i = (1/n_i) * sum_j(l_i,j), grads are grad(L_i)
        # for the jth token of each n_i total non-valid tokens in the current microbatch
        loss, grads = grad_fn(idx_, targets_, model, ignore_idx=-1, compute_dtype=compute_dtype)
        # to correctly normalize our loss over the total number of non-padding tokens over multiple grad accm steps, we multiply by the total valid token count n_i to correctly accumulate the normalised loss later
        # L_i = (1/n_i) * sum_j(l_i,j)
        # n_i * L_i = sum_j(l_i,j)
        n_i = jnp.sum(targets_ >= 0)
        loss *= n_i
        grads = jax.tree.map(lambda g: g * n_i, grads)
        
        loss_accm, grads_accm = carry
        return (loss_accm + loss, jax.tree.map(jnp.add, grads_accm, grads)), None

    initial_loss = jax.lax.pcast(0.0, ("b",), to="varying")
    # initial_grads = jax.tree.map(lambda p: jax.lax.pcast(jnp.zeros_like(p), "b", to="varying"), model)
    initial_grads = jax.tree.map(jnp.zeros_like, model)
    # loss_accm = sum_i(n_i * L_i) = sum_i(sum_j(l_i,j))
    # grads_accm = sum_i(n_i * grad(L_i)) = sum_i(sum_j(grad(l_i,j)))
    (loss_accm, grads_accm), _ = jax.lax.scan(inner_step, (initial_loss, initial_grads), jnp.arange(grad_accm_steps))

    # N = sum_i(n_i), total valid tokens across microbatches and ranks
    total_tokens = jax.lax.psum(jnp.sum(targets >= 0), "b")

    # recall that at each grad accm step we calculate the mean loss L_i = (1 / n_i) * sum_j(l_i,j)
    # and that at each grad accm step we undo this mean by multiplying by n_i, so
    # loss_accm = sum_i(n_i * L_i) = sum_i(sum_j(l_i,j)) - the unnormalized per-token losses
    # to now obtain a loss normalized over all valid tokens, we simply divide by N
    # L = loss_accm / N
    # (we also perform an all-reduce here to obtain the correct loss over all ranks)
    loss = jax.lax.psum(loss_accm, "b") / total_tokens

    # our gradient normalisation follows the same above procedure
    # G = (1 / N) * sum_i(n_i * grad(L_i)) = (1/ N) * sum_i(sum_j(grad(l_i,j)))
    # G = (1 / N) * grads_accm
    # note: we sum vs. average gradients across ranks here as we manually normalize grads by global valid token count
    grads = jax.lax.psum(grads_accm, "b")
    grads = jax.tree.map(lambda g: g / total_tokens, grads)

    updates, state = state.update(model, grads, lr_multiplier)
    model = jax.tree.map(jnp.subtract, model, updates)
    return model, state, loss, total_tokens


# this time we tokenize prompts using our chat template
prompts = [
    "Can you fix the bug in hello.py?",
    "Can you implement a MLP layer for me?",
    "Where is the utils.py file located?",
]
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

prompts = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
prompt_idx = [tokenizer.render_conversation(p)[0] for p in prompts]
prompt_idx = [p + [assistant_start] for p in prompt_idx]

total_training_time = 0
x, y = next(train_loader)
for step in range(num_steps):
    last_step = (step + 1) == num_steps
    lr_multiplier = jnp.array(get_lr_multiplier(step))

    d0 = time.perf_counter()
    model, state, loss, total_tokens = train_step(x, y, model, state, lr_multiplier)
    x, y = next(train_loader)
    loss = float(loss)  # synchronize
    total_tokens = int(total_tokens)
    dt = time.perf_counter() - d0

    flops_per_sec = num_flops_per_token * total_tokens / dt
    mfu = 100 * flops_per_sec / accelerator_flops
    tkps = int(total_tokens // dt)
    eta = ((num_steps - step) * dt) / 60
    total_training_time += dt

    print0(f"Step: {step}/{num_steps} | Loss: {loss:.3f} | dt: {dt:.2f}s | tkps: {tkps} | mfu: {mfu:.2f} | min ETA {eta:.1f}m | lr_multiplier: {lr_multiplier:.3f}")

    if (step % profile_every == 0) or last_step:
        memory_stats = jax.local_devices()[0].memory_stats() or {}
        used, available = memory_stats.get("peak_bytes_reserved", 0) / 1e9, memory_stats.get("bytes_reservable_limit", 0) / 1e9
        print0(f"\tPeak bytes reserved/limit: {used:.2f}/{available:.2f}")

    if (step % sample_every == 0) or last_step:
        for idx in prompt_idx:
            new_tokens = generate(
              idx,
              model,
              max_tokens=64,
              temperature=None,
              compute_dtype=compute_dtype,
              pad_token_id=assistant_end,
              rng=rng,
              assistant_end_id=assistant_end
            )
            print0("\t" + tokenizer.decode(idx + [int(t[0]) for t in new_tokens]))

    if (step % eval_every == 0) or last_step:
        d0 = time.perf_counter()
        eval_steps = eval_tokens // (minibatch_size * max_seq_len * world_size)
        val_bpb = evaluate_bpb(model, get_val_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
        print0(f"\tbpb: {float(val_bpb):.4f} | dt: {(time.perf_counter() - d0):.2f}s")

print0(f"Total training time: {(total_training_time / 60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
