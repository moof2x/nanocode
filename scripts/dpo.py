"""DPO fine-tuning with IPO loss. Loads an SFT checkpoint and trains on paired preference data."""
import argparse
import operator
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from nanocode.checkpointing import load_checkpoint, load_model_config, save_checkpoint
from nanocode.common import get_base_dir, get_model_dir, init_distributed, print0, setup_logging
from nanocode.generation import generate
from nanocode.eval import evaluate_bpb
from nanocode.gpt import GPT, estimate_flops
from nanocode.muon import Muon
from nanocode.tokenizer import get_token_bytes, get_tokenizer
from data.json_dataset import JSONDataset, JSONPreferenceDataset, PairedJSONPreferenceDataset
from data.common import SYSTEM_PROMPT
from data.mixture import TaskMixture

# distributed setup
world_size, mesh = init_distributed()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='sft')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--minibatch-size', type=int, default=32)
parser.add_argument('--num-steps', type=int, default=-1)
parser.add_argument('--num-epochs', type=int, default=1)
parser.add_argument('--beta', type=float, default=0.5, help='IPO hparam')
parser.add_argument('--eps', type=float, default=1e-10)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--wte-lr', type=float, default=0.003)
parser.add_argument('--lm-head-lr', type=float, default=0.0001)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--attn-impl', type=str, default='splash', choices=['splash', 'eager'])
parser.add_argument('--accelerator-flops', type=float, default=918e12)
parser.add_argument('--compute-dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--sample-every', type=int, default=50)
parser.add_argument('--eval-every', type=int, default=50)
parser.add_argument('--profile-every', type=int, default=500)
args = parser.parse_args()

checkpoint = args.checkpoint
### optimization hparams
batch_size = args.batch_size
minibatch_size = args.minibatch_size
num_steps = args.num_steps
num_epochs = args.num_epochs
# IPO hparam
beta = args.beta
# learning rates
eps = args.eps
wd = args.wd
wte_lr = args.wte_lr
lm_head_lr = args.lm_head_lr
lr = args.lr
### misc
seed = args.seed
attn_impl = args.attn_impl
accelerator_flops = args.accelerator_flops # TPU v6e
compute_dtype = jnp.bfloat16 if args.compute_dtype == 'bfloat16' else jnp.float32
### training loop control
sample_every = args.sample_every
eval_every = args.eval_every
profile_every = args.profile_every

base_dir = get_base_dir()
rollouts_dir = base_dir / "rollouts"
model_dir = get_model_dir()
setup_logging(model_dir / "dpo_log.txt")
for k, v in vars(args).items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()

base_checkpoint_dir = model_dir / f"{checkpoint}_checkpoints"
checkpoint_dir = model_dir / "dpo_checkpoints"
config = load_model_config(base_checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOCODE_BASE_DIR={base_dir} MODEL_TAG={os.environ.get('MODEL_TAG', '')} {command}")

max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len * 20  # magic number from nanochat
accelerator_flops *= world_size
print0(f"World size: {world_size}")

assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"

model = GPT.init(config, rng, attn_impl)
ref_model = GPT.init(config, rng, attn_impl)

model = load_checkpoint(base_checkpoint_dir / "model.zarr", model)
ref_model = load_checkpoint(base_checkpoint_dir / "model.zarr", ref_model)

num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print0(f"{num_params} model parameters")
print0("=" * 20)

num_flops_per_token = estimate_flops(model)
print0(f"Estimated FLOPs per token: {num_flops_per_token}")

state = Muon.init(
    model,
    eps=eps,
    wd=wd,
    wte_lr=wte_lr,
    lm_head_lr=lm_head_lr,
    lr=lr,
)

# we use PairedJSONPreferenceDataset over JSONPreferenceDataset to filter out
# pairs where either chosen or rejected has no assistant content
# I found this improved performance a bit in my specific synthetic datasets
train_ds = TaskMixture([
    PairedJSONPreferenceDataset(rollouts_dir / "nanocode-tulu-selfoss-evol-preference/all.pref_train.jsonl", seed),
    PairedJSONPreferenceDataset(rollouts_dir / "nanocode-long-context-preference/preference_rollouts_train.jsonl", seed),
], seed)

val_pref_ds = TaskMixture([
    PairedJSONPreferenceDataset(rollouts_dir / "nanocode-tulu-selfoss-evol-preference/all.pref_test.jsonl", seed),
    PairedJSONPreferenceDataset(rollouts_dir / "nanocode-long-context-preference/preference_rollouts_test.jsonl", seed),
], seed)

val_bpb_ds = TaskMixture([
    JSONDataset(rollouts_dir / "nanocode-tulu-selfoss-evol/all_test.jsonl", seed),
    JSONDataset(rollouts_dir / "nanocode-long-context/rollouts_test.jsonl", seed),
], seed)


def bpb_dataloader(dataset, B, T, tokenizer):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    B *= jax.local_device_count()

    def collate(batch):
        inputs = np.full((B, T), pad_token_id, dtype=np.int32)
        targets = np.full_like(inputs, -1)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids, mask = np.array(ids), np.array(mask)
            inputs[i, :n-1] = ids[:-1]
            row_targets = ids[1:n]
            row_targets[mask[1:n] == 0] = -1
            targets[i, :n-1] = row_targets
        return inputs, targets

    while True:
        batch = []
        for i in range(jax.process_index(), B * jax.process_count(), jax.process_count()):
            conv = dataset[i % len(dataset)]
            ids, mask = tokenizer.render_conversation(conv, max_tokens=T + 1)
            batch.append((ids, mask))
        yield collate(batch)


def dist_bpb_dataloader(dataset, batch_size, seq_len, tokenizer, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * world_size
    loader = bpb_dataloader(dataset, batch_size, seq_len, tokenizer)
    return map(partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size, seq_len)), loader)


def dataloader(dataset, B, b, T, tokenizer):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    B *= jax.local_device_count() # each process collects data for all of it's local accelerators
    # we'll create minibatch sizes of concatenated chosen-rejected pairs
    minibatch_size = b
    b *= jax.local_device_count()
    # we'll collect B // b (grad_accm_steps) mininbatches as np.arrays
    # then when we have grad_accm_steps of these, we'll stack and convert to a jax.Array
    # so we're only performing a single host-to-device transfer
    
    def collate(batch):
        inputs = np.full((len(batch), T), pad_token_id, dtype=np.int32)
        targets = np.full_like(inputs, -1)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids, mask = np.array(ids), np.array(mask)
            inputs[i, : n - 1] = ids[:-1]
            row_targets = ids[1:n]
            row_targets[mask[1:n] == 0] = -1
            targets[i, : n - 1] = row_targets
        return inputs, targets

    batch = []
    minibatch = []
    while True:
        for i in range(jax.process_index(), len(dataset), jax.process_count()):
            # append a tuple of ((chosen_ids, chosen_mask), (rejected_ids, rejected_mask))
            chosen, rejected = dataset[i]
            chosen = tokenizer.render_conversation(chosen, max_tokens=T + 1)
            rejected = tokenizer.render_conversation(rejected, max_tokens= T + 1)
            minibatch.append((chosen, rejected))
            if len(minibatch) == b:
                # we have accumulated enough samples for a single forward pass; collate
                chosen_batch, rejected_batch = zip(*minibatch)
                # we need to ensure that we are interleaving chosen/rejected pairs so when we later shard
                # our inputs, each device gets an equal number of chosen and rejected samples
                chosen_rejected_pairs = []
                for i in range(jax.local_device_count()):
                    chosen_rejected_pairs.extend(chosen_batch[i * minibatch_size: (i+1) *  minibatch_size])
                    chosen_rejected_pairs.extend(rejected_batch[i * minibatch_size: (i+1) *  minibatch_size])
                batch.append(collate(chosen_rejected_pairs))
                minibatch = []
            if len(batch) == B // b:
                # we have accumulated grad_accm_steps minibatches; yield
                ids, targets = zip(*batch)
                batch = []
                yield np.concatenate(ids), np.concatenate(targets)
    

def dist_dataloader(dataset, batch_size, minibatch_size, seq_len, tokenizer, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * world_size
    loader = dataloader(dataset, batch_size, minibatch_size, seq_len, tokenizer)
    # we collect 2x our batch size as a batch comprises B pairs of chosen, rejected samples
    return map(partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size  * 2, seq_len)), loader)


train_loader = dist_dataloader(train_ds, batch_size, minibatch_size, max_seq_len, tokenizer, mesh)
get_val_pref_dataloader = lambda: dist_dataloader(val_pref_ds, minibatch_size, minibatch_size, max_seq_len, tokenizer, mesh)
get_val_bpb_dataloader = lambda: dist_bpb_dataloader(val_bpb_ds, minibatch_size, max_seq_len, tokenizer, mesh)

minibatch_size *= 2

steps_per_epoch = len(train_ds) // (batch_size * world_size)
if num_steps < 0:
    num_steps = steps_per_epoch * num_epochs


def get_lr_multiplier(step):
    # linear lr decay
    return 1 - step / num_steps


# we construct a PartitionSpec with default behaviour indicating to replicate for our model and optimizer states
model_spec = jax.tree.map(lambda _: jax.P(), model)
state_spec = jax.tree.map(lambda _: jax.P(), state)

in_specs = (jax.P("b", None), jax.P("b", None), model_spec, model_spec, state_spec, jax.P())
out_specs = (model_spec, state_spec, jax.P(), jax.P(), jax.P(), jax.P(), jax.P())

def concatenated_forward(idx, targets, model, ignore_idx: int=-1, compute_dtype:jnp.dtype=jnp.bfloat16):
    # our inputs and targets comprise an equal number of chosen and rejected samples
    # obtain logprobs for all of these in one go
    logits, _ = model.forward(idx, compute_dtype=compute_dtype)
    logsumexp = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1, keepdims=True) # bs1
    per_token_logp = jnp.take_along_axis(logits, targets[:, :, None], axis=-1).squeeze(-1).astype(jnp.float32) - logsumexp.squeeze(-1) # bsv -> bs
    # mask out padding tokens
    valid = jnp.not_equal(targets, ignore_idx)
    per_token_logp = jnp.where(valid, per_token_logp, 0.0)
    sum_logp = per_token_logp.sum(axis=-1) # (2B,)
    len_chosen = idx.shape[0] // 2
    return sum_logp[:len_chosen], sum_logp[len_chosen:]
    
def calculate_loss(idx, targets, model, ref_chosen_logp, ref_rejected_logp, ignore_idx: int=-1, compute_dtype: jnp.dtype=jnp.bfloat16):
    pi_chosen_logp, pi_rejected_logp = concatenated_forward(idx, targets, model, ignore_idx=-1, compute_dtype=compute_dtype)

    pi_logratio = pi_chosen_logp - pi_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp

    logits = pi_logratio - ref_logratio
    
    # IPO loss: targets a finite margin of 1/(2*beta) instead of pushing margins to infinity
    loss = (logits - 1 / (2 * beta)) ** 2
    chosen_rewards = jnp.sum(beta * (pi_chosen_logp - ref_chosen_logp))
    rejected_rewards = jnp.sum(beta * (pi_rejected_logp - ref_rejected_logp))
    accuracy = jnp.sum(logits > 0)

    return jnp.mean(loss), (chosen_rewards, rejected_rewards, accuracy)

grad_fn = jax.value_and_grad(calculate_loss, argnums=2, has_aux=True)

@jax.jit(donate_argnums=(2, 4))
@jax.shard_map(in_specs=in_specs, out_specs=out_specs, mesh=mesh, check_vma=False)
def train_step(idx, targets, model, ref_model, state, lr_multiplier):
    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)

        # ref model forward runs outside gradient-captured loss function
        ref_chosen_logp, ref_rejected_logp = concatenated_forward(idx_, targets_, ref_model, ignore_idx=-1, compute_dtype=compute_dtype)

        (loss, (chosen_rewards, rejected_rewards, accuracy)), grads = grad_fn(idx_, targets_, model, ref_chosen_logp, ref_rejected_logp, ignore_idx=-1, compute_dtype=compute_dtype)
        loss_accm, chosen_rewards_accm, rejected_rewards_accm, accuracy_accm, grads_accm = carry
        return (
            loss_accm + loss,
            chosen_rewards_accm + chosen_rewards,
            rejected_rewards_accm + rejected_rewards,
            accuracy_accm + accuracy,
            jax.tree.map(jnp.add, grads_accm, grads)
        ), None

    # loss, chosen_rewards, rejected_rewards, accuracy, grads
    initial_carry = (
        jax.lax.pcast(0.0, ("b",), to="varying"),
        jax.lax.pcast(0.0, ("b",), to="varying"),
        jax.lax.pcast(0.0, ("b",), to="varying"),
        jax.lax.pcast(0.0, ("b",), to="varying"),
        jax.tree.map(jnp.zeros_like, model)
    )
    (loss, chosen_rewards, rejected_rewards, accuracy, grads), _ = jax.lax.scan(inner_step, initial_carry, jnp.arange(grad_accm_steps))
    
    
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps

    loss = jax.lax.pmean(loss, "b")
    # we normalise our rewards w.r.t. the total number of pairs in our batch
    n_pairs = idx.shape[0] // 2
    chosen_rewards = jax.lax.pmean(chosen_rewards / n_pairs, "b")
    rejected_rewards = jax.lax.pmean(rejected_rewards / n_pairs, "b")
    accuracy = jax.lax.pmean(accuracy / n_pairs, "b")
    
    grads = jax.lax.pmean(grads, "b")
    valid_tokens = jnp.sum(targets >= 0)
    total_tokens = jax.lax.psum(valid_tokens, "b")

    updates, state = state.update(model, grads, lr_multiplier)
    model = jax.tree.map(jnp.subtract, model, updates)
    return model, state, loss, chosen_rewards, rejected_rewards, accuracy, total_tokens


eval_in_specs = (jax.P("b", None), jax.P("b", None), model_spec, model_spec)
eval_out_specs = (jax.P(), jax.P(), jax.P(), jax.P())

@jax.jit
@jax.shard_map(in_specs=eval_in_specs, out_specs=eval_out_specs, mesh=mesh, check_vma=False)
def eval_step(idx, targets, model, ref_model):
    pi_chosen_logp, pi_rejected_logp = concatenated_forward(idx, targets, model, ignore_idx=-1, compute_dtype=compute_dtype)
    ref_chosen_logp, ref_rejected_logp = concatenated_forward(idx, targets, ref_model, ignore_idx=-1, compute_dtype=compute_dtype)
    pi_logratio = pi_chosen_logp - pi_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = pi_logratio - ref_logratio
    loss = jnp.mean((logits - 1 / (2 * beta)) ** 2)
    n_pairs = idx.shape[0] // 2
    chosen_rewards = jnp.sum(beta * (pi_chosen_logp - ref_chosen_logp)) / n_pairs
    rejected_rewards = jnp.sum(beta * (pi_rejected_logp - ref_rejected_logp)) / n_pairs
    accuracy = jnp.sum(logits > 0) / n_pairs
    return jax.lax.pmean(loss, "b"), jax.lax.pmean(accuracy, "b"), jax.lax.pmean(chosen_rewards, "b"), jax.lax.pmean(rejected_rewards, "b")

def evaluate_dpo(model, ref_model, val_loader, steps):
    total_loss, total_acc, total_chosen, total_rejected = 0.0, 0.0, 0.0, 0.0
    for i in range(steps):
        x, y = next(val_loader)
        loss, acc, chosen, rejected = eval_step(x, y, model, ref_model)
        total_loss += float(loss)
        total_acc += float(acc)
        total_chosen += float(chosen)
        total_rejected += float(rejected)
    n = max(steps, 1)
    margins = (total_chosen - total_rejected) / n
    return total_loss / n, total_acc / n, margins, total_chosen / n, total_rejected / n


prompts = [
    "Can you fix the bug in hello.py?",
    "Can you implement a MLP layer for me?",
    "Where is the utils.py file located?",
]
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

prompts = [{"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]} for p in prompts]
prompt_idx = [tokenizer.render_conversation(p)[0] for p in prompts]
prompt_idx = [p + [assistant_start] for p in prompt_idx]

total_training_time = 0
x, y = next(train_loader)
for step in range(num_steps):
    last_step = (step + 1) == num_steps
    lr_multiplier = jnp.array(get_lr_multiplier(step))

    d0 = time.perf_counter()
    model, state, loss, chosen_rewards, rejected_rewards, accuracy, total_tokens = train_step(x, y, model, ref_model, state, lr_multiplier)
    x, y = next(train_loader)
    loss = float(loss)  # synchronize
    total_tokens = int(total_tokens)
    dt = time.perf_counter() - d0

    flops_per_sec = num_flops_per_token * total_tokens / dt
    mfu = 100 * flops_per_sec / accelerator_flops
    tkps = int(total_tokens // dt)
    eta = ((num_steps - step) * dt) / 60
    total_training_time += dt

    margins = chosen_rewards - rejected_rewards
    print0(f"Step: {step}/{num_steps} | Loss: {loss:.3f} | Acc: {float(accuracy):.3f} | Margins: {margins:.3f} | Rewards (chosen/rejected): {chosen_rewards:.2f}/{rejected_rewards:.2f} | dt: {dt:.2f}s | tkps: {tkps} | mfu: {mfu:.2f} | min ETA {eta:.1f} | lr_multiplier: {lr_multiplier:.3f}")


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
        val_bpb = evaluate_bpb(model, get_val_bpb_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
        val_loss, val_acc, val_margins, val_chosen, val_rejected = evaluate_dpo(model, ref_model, get_val_pref_dataloader(), eval_steps)
        print0(f"\tval bpb: {float(val_bpb):.4f} | val loss: {val_loss:.3f} | val acc: {val_acc:.3f} | val margins: {val_margins:.3f} | val rewards: {val_chosen:.2f}/{val_rejected:.2f} | dt: {(time.perf_counter() - d0):.2f}s")

print0(f"Total training time: {(total_training_time / 60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
