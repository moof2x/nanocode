import jax
import jax.numpy as jnp

from nanojax.gpt import GPT
from nanojax.tokenizer import get_tokenizer

compute_dtype = jnp.float32

def generate(model: GPT, idx: int, n_steps: int, rng, temperature: float=1.0, top_k:int=None):
    
    for i in range(n_steps):
        print(idx.shape)
        logits = model.forward(idx, compute_dtype)[:, -1, :] # bsv -> bv
        pred = jnp.argmax(logits, axis=-1, keepdims=True)
        idx = jnp.concat([idx, pred], axis=1)

    return idx
        
    
from nanojax.configs import d3_4m

tokenizer = get_tokenizer()
rng = jax.random.key(42)
model = GPT.init(d3_4m, rng)
inp = ["The quick brown fox jumped over the"]
idx = tokenizer.encode(inp, prepend=tokenizer.get_bos_token_id())
result = generate(model, jnp.asarray(idx, dtype=jnp.int32), 10, rng)
print(tokenizer.decode(result[0]))
