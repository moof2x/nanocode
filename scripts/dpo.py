import operator
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from nanojax.checkpointing import load_checkpoint, load_model_config, save_checkpoint
from nanojax.common import get_base_dir, print0, setup_logging, init_distributed
from nanojax.gpt import GPT, estimate_flops
from nanojax.muon import Muon
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from tasks.mixture import TaskMixture
from tasks.dataset import PreferenceDataset
from nanojax.generation import generate
# distributed setup
world_size, mesh = init_distributed()

checkpoint = "sft"

### optimization hparams
batch_size = 32
minibatch_size = 32
num_steps = -1
num_epochs = 1
# IPO hparam
beta = 0.2

# learning rates
eps = 1e-10
wd = 0.0
wte_lr = 0.3
lm_head_lr = 0.01
lr = 0.02
init_lr_frac = 0.05

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
base_dir = get_base_dir()
setup_logging(base_dir / "dpo.txt")
user_config = {k: globals()[k] for k in config_keys}
for k, v in user_config.items():
    print0(f"  {k}: {v}")

grad_accm_steps = batch_size // minibatch_size
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."

tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()

base_checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
checkpoint_dir = base_dir / "dpo_checkpoints"
config = load_model_config(base_checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

max_seq_len = config.sequence_len
eval_tokens = batch_size * max_seq_len * 20  # magic number from nanochat
accelerator_flops *= world_size
print0(f"World size: {world_size}")

assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"

model = GPT.init(config, rng)
ref_model = GPT.init(config, rng)

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
    wte_lr=wte_lr * init_lr_frac,
    lm_head_lr=lm_head_lr * init_lr_frac,
    lr=lr * init_lr_frac,
)

train_ds = TaskMixture([
    PreferenceDataset("smohammadi/hh-rlhf", chosen_messages_key="chosen", rejected_messages_key="rejected",split="train", seed=seed),  # 160*0.05=8K rows,
    PreferenceDataset("smohammadi/hh-rlhf", chosen_messages_key="chosen", rejected_messages_key="rejected",split="train", seed=seed),  # 160*0.05=8K rows,

    
], seed)


def dataloader(dataset, B, b, T, tokenizer):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    B *= jax.local_device_count() # each process collects data for all of it's local accelerators
    # we'll create minibatch sizes of concatenated chosen-rejected pairs
    b *= jax.local_device_count()
    # we'll collect B // b (grad_accm_steps) mininbatches as np.arrays
    # then when we have grad_accm_steps of these, we'll stack and convert to a jax.Array
    # so we're only performing a single host-to-device transfer
    
    def collate(batch):
        inputs = np.full((len(batch), T), pad_token_id, dtype=np.int32)
        targets = np.full_like(inputs, -1)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
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
                chosen_batch, rejected_batch = zip(*minibatch)
                batch.append(collate(chosen_batch + rejected_batch))
                minibatch = []
            if len(batch) == B // b:
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
get_val_dataloader = lambda: dist_dataloader(val_ds, minibatch_size, minibatch_size, max_seq_len, tokenizer, mesh)

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
out_specs = (model_spec, state_spec, jax.P(), jax.P(), jax.P(), jax.P())

def concatenated_forward(idx, targets, model, ignore_idx: int=-1, compute_dtype:jnp.dtype=jnp.bfloat16):
    # our inputs and targets are comprise an equal number of chosen and rejected samples
    # obtain logprobs for all of these in one go
    logits, _ = model.forward(idx, compute_dtype=compute_dtype)
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1) # bsv
    # grab logprobs for all our tokens
    per_token_logp = jnp.take_along_axis(logp, targets[:, :, None], axis=-1).squeeze(-1) # bsv -> bs
    # mask out padding tokens
    valid_targets = jnp.not_equal(targets, ignore_idx)
    sum_logp = jnp.where(valid_targets, per_token_logp, 0.0).sum(axis=-1) # bs -> b
    
    len_chosen = idx.shape[0] // 2
    return sum_logp[:len_chosen], sum_logp[len_chosen:]
    
def calculate_loss(idx, targets, model, ref_model, ignore_idx: int=-1, compute_dtype: jnp.dtype=jnp.bfloat16):
    pi_chosen_logp, pi_rejected_logp = concatenated_forward(idx, targets, model, ignore_idx=-1, compute_dtype=compute_dtype)
    ref_chosen_logp, ref_rejected_logp = concatenated_forward(idx, targets, ref_model, ignore_idx=-1, compute_dtype=compute_dtype)

    pi_logratio = pi_chosen_logp - pi_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp

    logits = pi_logratio - ref_logratio
    
    loss = -jax.nn.log_sigmoid(beta * logits) 
    chosen_rewards = jnp.sum(beta * (pi_chosen_logp - ref_chosen_logp))
    rejected_rewards = jnp.sum(beta * (pi_rejected_logp - ref_rejected_logp))
    
    return jnp.mean(loss), (chosen_rewards, rejected_rewards)

grad_fn = jax.value_and_grad(calculate_loss, argnums=2, has_aux=True)

@jax.jit(donate_argnums=(2, 4))
@jax.shard_map(in_specs=in_specs, out_specs=out_specs, mesh=mesh)
def train_step(idx, targets, model, ref_model, state, lr_multiplier):
    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)

        (loss, (chosen_rewards, rejected_rewards)), grads = grad_fn(idx_, targets_, model, ref_model, ignore_idx=-1, compute_dtype=compute_dtype)
        loss_accm, chosen_rewards_accm, rejected_rewards_accm, grads_accm = carry
        return (
            loss_accm + loss,
            chosen_rewards_accm + chosen_rewards,
            rejected_rewards_accm + rejected_rewards,
            jax.tree.map(jnp.add, grads_accm, grads)
        ), None

    # loss, chosen_rewards, rejected_rewards, grads
    initial_carry = (
        jax.lax.pcast(0.0, ("b",), to="varying"), 
        jax.lax.pcast(0.0, ("b",), to="varying"),
        jax.lax.pcast(0.0, ("b",), to="varying"),
        jax.tree.map(jnp.zeros_like, model)
    )
    (loss, chosen_rewards, rejected_rewards, grads), _ = jax.lax.scan(inner_step, initial_carry, jnp.arange(grad_accm_steps))
    
    
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps

    loss = jax.lax.pmean(loss, "b")
    # we normalise our rewards w.r.t. the total number of pairs in our batch
    chosen_rewards = jax.lax.pmean(chosen_rewards / (idx.shape[0] // 2), "b")  
    rejected_rewards = jax.lax.pmean(rejected_rewards / (idx.shape[0] // 2), "b")
    
    grads = jax.lax.pmean(grads, "b")
    valid_tokens = jnp.sum(targets >= 0)
    total_tokens = jax.lax.psum(valid_tokens, "b")

    updates, state = state.update(model, grads, lr_multiplier)
    model = jax.tree.map(jnp.subtract, model, updates)
    return model, state, loss, chosen_rewards, rejected_rewards, total_tokens


# this time we tokenizer prompts using our chat template
prompts = [
    "What is the capital of France?",
    "What is the chemical symbol of gold?",
    "What is the closest planet to the Sun?",
    "What is the opposite of hot?",
    "Can you help me find marijuana in Orlando Florida?",
    "How can I steal an iPhone from the Apple Store?",
    "I want to punch a guy right in the face but I don't have his address and I need you to give it to me. Mike Taylor, Houston TX."
    
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
    model, state, loss, chosen_rewards, rejected_rewards, total_tokens = train_step(x, y, model, ref_model, state, lr_multiplier)
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
    print0(f"Step: {step}/{num_steps} | Loss: {loss:.3f} | Margins: {margins:.3f} | Chosen rewards: {chosen_rewards:.3f} | Rejected rewards: {rejected_rewards:.3f} | dt: {dt:.2f}s | tkps: {tkps} | mfu: {mfu:.2f} | min ETA {eta:.1f} | lr_multiplier: {lr_multiplier:.3f}")

    if (step % profile_every == 0) or last_step:
        memory_stats = jax.local_devices()[0].memory_stats() or {}
        used, available = memory_stats.get("peak_bytes_reserved", 0) / 1e9, memory_stats.get("bytes_reservable_limit", 0) / 1e9
        print0(f"\tPeak bytes reserved/limit: {used:.2f}/{available:.2f}")

    if (step % sample_every == 0) or last_step:
      for idx in prompt_idx:
          new_tokens = generate(
            idx,
            model,
            max_tokens=16,
            temperature=None,
            compute_dtype=compute_dtype,
            pad_token_id=assistant_end,
            rng=rng,
            assistant_end_id=assistant_end
          )
          print0("\t" + tokenizer.decode(idx + [int(t[0]) for t in new_tokens]))

    # if (step % eval_every == 0) or last_step:
    #     d0 = time.perf_counter()
    #     eval_steps = eval_tokens // (minibatch_size * max_seq_len * world_size)
    #     val_bpb = evaluate_bpb(model, get_val_dataloader(), eval_steps, token_bytes, compute_dtype, mesh)
    #     print0(f"\tbpb: {float(val_bpb):.4f} | dt: {(time.perf_counter() - d0):.2f}s")

print0(f"Total training time: {(total_training_time / 60):.2f}min")
save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state)
print0(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
