import math
import operator
import os
import sys
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import trackio
from collections import deque
import numpy as np

from nanojax import configs
from nanojax.adamw import AdamW
from nanojax.checkpointing import save_checkpoint, load_checkpoint
from nanojax.common import get_base_dir
from nanojax.dataloader import tokenizing_data_loader
from nanojax.gpt import GPT, GPTConfig, calculate_loss, estimate_flops
from nanojax.muon import Muon
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from tasks.mixture import TaskMixture
from tasks.dolly import Dolly
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.hhrlhf import HHRLHF

config = configs.d3
lr = 3e-4
batch_size = 64
minibatch_size = 64
seed = 42
num_steps = -1
accelerator_flops = 11.15e12 # 2080 super FLOPs/sec
profile_steps = 50
sample_steps = 100
compute_dtype = jnp.float32

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
tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
base_dir = get_base_dir()
base_checkpoint_dir = base_dir / "base_checkpoints"
checkpoint_dir = base_dir / "mid_checkpoints"
rng = jax.random.key(seed)


assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
model = GPT.init(
    config,
    rng
)
num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
print("="*20)

num_flops_per_token = estimate_flops(model)
print(f"Estimated FLOPs per token: {num_flops_per_token}")

state = Muon.init(model)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

model = load_checkpoint(base_checkpoint_dir / "model.zarr", model)

last_step = False
dataset = TaskMixture([
    SmolTalk("train", seed),
    Dolly(seed),
    HHRLHF("train", seed),
    MMLU("train", seed)
], seed)

if num_steps < 0:
    num_steps = len(dataset)

def dataloader():
    global last_step
    ds_size = len(dataset)
    token_buffer = deque()
    needed_tokens = batch_size * max_seq_len + 1
    cursor = 0
    while True:
        while len(token_buffer) < needed_tokens:
            sample = dataset[cursor]
            ids, _ = tokenizer.render_conversation(sample)
            token_buffer.extend(ids)
            cursor += 1
            if cursor >= ds_size:
                last_step = True
        scratch = np.array([token_buffer.popleft() for _ in range(needed_tokens)], dtype=np.int32)
        inputs = jnp.asarray(scratch[:-1]).reshape(batch_size, max_seq_len)
        targets = jnp.asarray(scratch[1:]).reshape(batch_size, max_seq_len)
        yield inputs, targets
    
train_loader = dataloader()
x, y = next(train_loader)
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

# this time we tokenizer prompts using our chat template
prompts = [
    "What is the capital of France?",
    "Complete the following sentence: 'Einstein's special theory of relatively states that energy'",
    "What is the closest planet to the Sun?"
]
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

prompts = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
prompt_idx = [tokenizer.render_conversation(p)[0] for p in prompts]
prompt_idx = [p + [assistant_start] for p in prompt_idx]
prompt_idx = [jnp.asarray(p, dtype=jnp.int32)[None, :] for p in prompt_idx]
step = 1
while True:
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    log_dict = {"loss": float(loss)}
    print(f"Step {step}/{num_steps} | Loss: {loss:.3f}")
    if step % profile_steps== 0:
        # log profiling every now and then
        jax.block_until_ready(loss)
        dt = time.perf_counter() - d0
        flops_per_sec = num_flops_per_token * x.size / dt
        mfu = 100 * flops_per_sec / accelerator_flops
        print(f"\tdt: {dt:.3f}s | tkps: {int(x.size // dt)} | mfu: {mfu:.2f}")
        print(f"\tEstimated time remaining: {(((num_steps - step) * dt)/60):.1f} min")
        log_dict["tkps"] = int(x.size // dt)

    if step % sample_steps == 0:
        for idx in prompt_idx:
            for i in range(10):
                logits = model.forward(idx, compute_dtype)[:, -1, :] # bsv -> bv
                pred = jnp.argmax(logits, axis=-1, keepdims=True)
                idx = jnp.concat([idx, pred], axis=1)
            print(tokenizer.decode(idx[0]))

    step += 1 
    trackio.log(log_dict)
    if (num_steps and step == num_steps) or last_step:
        break

save_checkpoint(checkpoint_dir / "model.zarr", model)
save_checkpoint(checkpoint_dir / "state.zarr", state) 
print(f"Model (model.zarr) and optimizer state (state.zarr) checkpoints saved to {checkpoint_dir}.")
trackio.finish()

