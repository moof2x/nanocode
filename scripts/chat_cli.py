import jax
import jax.numpy as jnp
from nanojax.tokenizer import get_tokenizer
from nanojax.common import get_base_dir
from nanojax.checkpointing import load_checkpoint, load_model_config
import os
from nanojax.gpt  import GPT


checkpoint = "mid"
compute_dtype = jnp.float32
max_tokens = 16
seed = 42

exec(open(os.path.join('nanojax', 'configurator.py')).read()) # overrides from command line 
command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print(command)

tokenizer = get_tokenizer()
base_dir = get_base_dir()
checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
config = load_model_config(checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)

model = GPT.init(
    config,
    rng
)
model = load_checkpoint(checkpoint_dir / "model.zarr", model)

def generate(idx):
    for i in range(max_tokens):
        logits = model.forward(idx, compute_dtype)[:, -1, :] # bsv -> bv
        pred = jnp.argmax(logits, axis=-1, keepdims=True)
        idx = jnp.concat([idx, pred], axis=1)

        yield pred[0]

user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

tokens = []
while True:
    try:
        user_input = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    
    if not user_input:
        continue

    tokens.append(user_start)
    tokens.extend(tokenizer.encode(user_input))
    tokens.append(user_end)
    tokens.append(assistant_start)
    print("\nAssistant: ", end="", flush=True)

    inp_tokens = jnp.zeros((1, len(tokens) + max_tokens), dtype=jnp.int32)
    inp_tokens = inp_tokens.at[0, :len(tokens)].set(tokens)
    for i in range(max_tokens):
        logits = model.forward(inp_tokens[:, :len(tokens) + i], compute_dtype)[:, -1, :] # bsv -> bv
        token = jnp.argmax(logits, axis=-1, keepdims=True)

        inp_tokens = inp_tokens.at[0, len(tokens) + i].set(token)
        if token[0] == assistant_end:
            break
        token_text = tokenizer.decode(token)
        print(token_text, end="", flush=True)
    
    if int(token) != assistant_end:
        tokens.append(assistant_end)
    print(tokens)
    print()
    
    

    
    
