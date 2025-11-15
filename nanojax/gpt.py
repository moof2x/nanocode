# there's a nice paradigm in JAX where you can register dataclasses as PyTrees.
# In JAX you can apply many tensor-adjacent operations to PyTrees, all of which
# are well-supported by the compiler.
# This is great for us. In this implementation, we'll separate state (model, optimizer params)
# from the modeling code itself. We can define the state as a PyTree and all modeling
# code will be applying transformations on this PyTree (or mapping data through it).
# if you're wondering about the variable names here and in nanochat
# they come from https://github.com/openai/gpt-2/blob/master/src/model.py

from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from nanojax.common import fold_in_str
from jax.tree_util import register_dataclass

jax_dataclass = lambda cls: register_dataclass(dataclass(cls))

# einsum notation:
#    b: batch
#    s: sequence len
#    v: vocab_size
#    e: n_embed, embedding dim
#    q: n_head, number of query heads
#    k: n_kv_head, number of KV-heads
#    h: head_dim
#    Q: n_head * head_dim 
#    K: n_kv_head * head_dim

@dataclass
class GPTConfig:
    # default GPT2-117M params (nanochat version)
    sequence_len: int = 1024
    vocab_size: int = 50304 # originally 50257, nanochat bumps it to the nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embed: int = 768

@jax_dataclass
class MLP:
    c_fc: jax.Array # eE - [n_embed, 4 * n_embed]
    c_proj: jax.Array # Ee - [4 * n_embed, n_embed]

@jax_dataclass
class Attention:
    c_q: jax.Array # eQ [n_embed, n_head * head_dim]
    c_k: jax.Array # eK [n_embed, n_kv_head * head_dim]
    c_v: jax.Array # eK [n_embed, n_kv_head * head_dim]
    c_proj: jax.Array # ee [n_embed, n_embed]

@jax_dataclass
class Block:
    attn: Attention
    mlp: MLP

@jax_dataclass
class GPT:
    # in JAX we seperate "state" and "state transformations".
    # Forward passes, graadient updates, etc. are all examples
    # of purely functional transformations of state.

    # GPT() creates a tree-like data container which houses model parameters
    # and allows us to perform operations with JAX on this tree.
    # GPT.init() is a factory method which initializes this data container
    # with model parameters.
    wte: jax.Array # ve [vocab_size, n_embed] embedding layer
    h: list[Block] # n_layer transformer blocks
    lm_head: jax.Array # ev [n_embed, vocab_size] output proj

    @staticmethod              
    def init(cfg: GPTConfig, rng: jax.Array) -> "GPT":        
        # random state must be explicitly managed in JAX by "splitting"
        # random keys. fold_in_str does this by splitting the base key
        # based on the hash of a given string - in this case the weight name.
        std = 0.02
        wte = jax.random.normal(fold_in_str(rng, "wte"), (cfg.vocab_size, cfg.n_embed)) * std
        h = []
        for _ in range(cfg.n_layer):
            # TODO - why does nanochatdo some funky initialiation?
            head_dim = cfg.n_embed // cfg.n_head
            residual_std = std / math.sqrt(2 * cfg.n_layer)
    
            attn = Attention(
                c_q=jax.random.normal(fold_in_str(rng, "c_q"), (cfg.n_embed, cfg.n_head * head_dim)) * std,
                c_k=jax.random.normal(fold_in_str(rng, "k"), (cfg.n_embed, cfg.n_kv_head * head_dim)) *  std,
                c_v=jax.random.normal(fold_in_str(rng, "v"), (cfg.n_embed, cfg.n_kv_head * head_dim)) * std,
                c_proj=jax.random.normal(fold_in_str(rng, "o"), (cfg.n_embed, cfg.n_embed)) * residual_std,
            )
    
            mlp = MLP(
                c_fc=jax.random.normal(fold_in_str(rng, "c_fc"), (cfg.n_embed, 4 * cfg.n_embed)) * std,
                c_proj=jax.random.normal(fold_in_str(rng, "c_proj"), (4 * cfg.n_embed, cfg.n_embed)) * residual_std,
            )
            h.append(Block(attn=attn, mlp=mlp))

        # TODO - why does nanochat zero out classifier weights and c_proj in mlps?
        lm_head = jax.random.normal(fold_in_str(rng, "lm_head"), (cfg.n_embed, cfg.vocab_size)) * std
        return GPT(
            wte=wte,
            h=h,
            lm_head=lm_head
        )
        
    def forward(self, x: jax.Array):
        # x: bs [batch_size, sequence_len]

        # MLP
        h = jnp.einsum("bse,eE->bsE", x, self.c_f)
        h = jax.nn.gelu(x) # TODO why does nanochat use relu^2?
        h2 = jnp.einsum("bsE,Ee->bse", x, self.c_f)       


rng = jax.random.key(42)
model = GPT.init(GPTConfig(), rng)
