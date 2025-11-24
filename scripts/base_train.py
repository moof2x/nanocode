from nanojax.dataloader import tokenizing_data_loader
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from nanojax.gpt import GPT, calculate_loss, AdamW, GPTConfig
from nanojax.configs import d6_35m
import operator
import time
import jax
import jax.numpy as jnp
import math
# jax.config.update('jax_compiler_enable_remat_pass', False)
# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

batch_size = 8
grad_accm_steps = 1# 256 // 8
assert batch_size % grad_accm_steps == 0, "batch_size must be evenly divisble by grad_accm_steps."
minibatch_size = batch_size // grad_accm_steps

train_loader = tokenizing_data_loader(batch_size, 1024, "train", tokenizer)
x, y = next(train_loader)

rng = jax.random.key(42)
# config = GPTConfig(n_layer=1, n_head=4, n_kv_head=4, n_embed=320, vocab_size=vocab_size)
config = d6_35m
assert vocab_size == config.vocab_size, f"mismatch between tokenizer vocab_size ({vocab_size}) and config vocab_size ({config.vocab_size})"
# config = GPTConfig(vocab_size=vocab_size)
model = GPT.init(
    config,
    rng
)
print(config)

num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
total_tokens = num_params * 20
num_steps = math.ceil(total_tokens / config.sequence_len / batch_size)
print(f"{num_params} model parameters")
print(f"training on {total_tokens} tokens over {num_steps} steps")

state = AdamW.init(model)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

@jax.jit
def train_step(idx, targets, model, state):

    def inner_step(carry, j):
        idx_ = jax.lax.dynamic_slice_in_dim(idx, j * minibatch_size, minibatch_size, axis=0)
        targets_ = jax.lax.dynamic_slice_in_dim(targets, j * minibatch_size, minibatch_size, axis=0)

        loss, grads = grad_fun(idx_, targets_, model)
        loss_accm, grads_accm = carry
        return (loss_accm + loss, jax.tree.map(jnp.add, grads_accm, grads)), None

    (loss, grads), _ = jax.lax.scan(inner_step, (0.0, jax.tree.map(jnp.zeros_like, model)), jnp.arange(grad_accm_steps))
    grads = jax.tree.map(lambda g: g / grad_accm_steps, grads)
    loss /= grad_accm_steps
    
    updates, state = state.update(model, grads, 1e-3)
    model = jax.tree.map(lambda p, u: p - u, model, updates)
    return model, state, loss
    
step = 0    
while True:
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    print(f"Step {step}/{num_steps} | Loss: {loss:.3f} ")
    step += 1 
    if step % 2 == 0:
        # log profiling every now and then
        jax.block_until_ready(loss)
        dt = time.perf_counter() - d0
        print(f"\tdt: {dt:.3f}s | tkps: {(x.size / dt):.3f}")
        print(f"\tTokens seen: {x.size * step}")
        print(f"Expected time remaining: {(((num_steps - step) * dt)/60):.3f} min")
    if step == num_steps:
        break
