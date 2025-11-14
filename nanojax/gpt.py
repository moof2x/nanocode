# there's a nice paradigm in JAX where you can register dataclasses as PyTrees.
# In JAX you can apply many tensor-adjacent operations to PyTrees, all of which
# are well-supported by the compiler.
# This is great for us. In this implementation, we'll separate state (model, optimizer params)
# from the modeling code itself. We can define the state as a PyTree and all modeling
# code will be applying transformations on this PyTree (or mapping data through it).
# if you're wondering about the variable names here and in nanochat
# they come from https://github.com/openai/gpt-2/blob/master/src/model.py

from dataclasses import dataclass
import jax
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
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embed: int

@jax_dataclass
class MLP:
    c_fc: jax.Array # eE - [n_embed, 4 * n_embed]
    c_proj: jax.Array # Ee - [4 * n_embed, n_embed]

@jax_dataclass
class Attention:
    q_proj: jax.Array # eQ [n_embed, n_head * head_dim]
    k_proj: jax.Array # eK [n_embed, n_kv_head * head_dim]
    v_proj: jax.Array # eK [n_embed, n_kv_head * head_dim]
    o_proj: jax.Array # ee [n_embed, n_embed]

@jax_dataclass
class Block:
    attn: Attention
    mlp: MLP

@jax_dataclass
class GPT:
    wte: jax.Array # ve [vocab_size, n_embed] embedding layer
    h: list[TransformerLayer] # n_layer transformer blocks
    lm_head: jax.Array # ev [n_embed, vocab_size] output proj

    @staticmethod              
    def init(cfg: GPTConfig, rng: jax.Array) -> "GPT":
        # in JAX we seperate "state" and "state transformations".
        # Forward passes, graadient updates, etc. are all examples
        # of purely functional transformations of state.
    
        # GPT() creates a tree-like data container which houses model parameters
        # and allows us to perform operations with JAX on this tree.
        # GPT.init() is a factory method which initializes this data container
        # with model parameters.
        
        # random state must be explicitly managed in JAX by "splitting"
        # random keys. fold_in_str does this based on the hash of a given string
        wte = jax.random.normal(fold_in_str(rng, "wte"), (cfg.vocab_size, cfg.n_embed), dtype=jnp.float32)
        
        
    def forward(self, x: jax.Array):

        # MLP
        h = jnp.einsum("bse,eE->bsE", x, self.c_f)
        h = jax.nn.gelu(x) # todo : does relu^2 work better?
        h2 = jnp.einsum("bsE,Ee->bse", x, self.c_f)       
