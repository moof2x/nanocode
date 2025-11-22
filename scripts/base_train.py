from nanojax.dataloader import tokenizing_data_loader
from nanojax.tokenizer import get_token_bytes, get_tokenizer
from nanojax.gpt import GPT, calculate_loss, AdamW, GPTConfig
import time
import jax

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

train_loader = tokenizing_data_loader(2, 2048, "train", tokenizer)
x, y = next(train_loader)

rng = jax.random.key(42)
config = GPTConfig(n_layer=5, n_head=4, n_kv_head=4, n_embed=320, vocab_size=vocab_size)
model = GPT.init(
    config,
    rng
)
print(config)
state = AdamW.init(model)
grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

@jax.jit
def train_step(idx, targets, model, state):
    loss, grads = grad_fun(x, y, model)
    updates, state = state.update(model, grads, 1e-3)
    model = jax.tree.map(lambda p, u: p - u, model, updates)
    return model, state, loss
    
step = 0    
while True:
    d0 = time.perf_counter()
    model, state, loss = train_step(x, y, model, state)
    x, y = next(train_loader)
    print(f"{step} | Loss: {loss:.3f} ")
    step += 1 
    if step % 10 == 0:
        # log profiling every now and then
        jax.block_until_ready(loss)
        dt = time.perf_counter() - d0
        print(f"\tdt: {dt:.3f}s | tkps: {(x.size / dt):.3f}")
        print(f"\tTokens seen: {x.size * step}")
