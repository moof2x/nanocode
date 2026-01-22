import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.common import get_base_dir, print0
from nanojax.generation import generate
from nanojax.gpt import GPT, KVCache
from nanojax.tokenizer import get_tokenizer

checkpoint = "mid"
compute_dtype = jnp.bfloat16
max_tokens = 128
seed = 42
temperature = 0.6

exec(open(os.path.join("nanojax", "configurator.py")).read()) # overrides from command line

tokenizer = get_tokenizer()
base_dir = get_base_dir()
checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
model_cfg = load_model_config(checkpoint_dir / "model.zarr")

command = f"python -m {__spec__.name} " + " ".join(sys.argv[1:])
print0(f"NANOJAX_BASE_DIR={base_dir} {command}")

rng = jax.random.key(seed)

model = GPT.init(model_cfg, rng)
model = load_checkpoint(checkpoint_dir / "model.zarr", model)

pad_token_id = tokenizer.encode_special("<|assistant_end|>")


user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

max_seq_len = 2 * model_cfg.sequence_len
tokens = []
while True:
    try:
        user_input = input("\nUser: ").strip()
    except (EOFError, KeyboardInterrupt):
        print0("\nGoodbye!")
        break
    if not user_input:
        continue

    tokens.append(user_start)
    tokens.extend(tokenizer.encode(user_input))
    tokens.append(user_end)
    tokens.append(assistant_start)
    print0("\nAssistant: ", end="", flush=True)
    
    for token in generate(tokens, model, max_tokens, temperature, compute_dtype, pad_token_id, rng, assistant_end_id=assistant_end):
        tokens.append(int(token[0]))
        
        print0(tokenizer.decode(token), end="", flush=True)
        
    if len(tokens) > max_seq_len:
        print0(f"Max sequence len {max_seq_len} exceeded. Goodbye!")
        break

    rng, _ = jax.random.split(rng)
    print0()
