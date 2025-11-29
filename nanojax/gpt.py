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
from functools import partial
import operator

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

def rms_norm(x: jax.Array) -> jax.Array:
    # performing rms norm in fp32 is typically more numerically stable
    x_out = x.astype(jnp.float32)
    mean = jnp.mean(jax.lax.square(x_out), axis=-1, keepdims=True)
    return (x_out * jax.lax.rsqrt(mean + 1e-6)).astype(x.dtype)

def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    # x: typically bsQ embedding
    # cos, sin: typically sc (where c = Q // 2)
    # we split our embedding to operate over pairs of dimensions
    x1, x2 = jnp.split(x, 2, axis=-1)
    # rotate our embeddings by transforming with a rotation matrix
    # constructed from our pre-computed scaled frequences
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    # stitch our embedding vector back up
    return jnp.append(y1, y2, axis=-1)


@dataclass
class GPTConfig:
    # default GPT2-117M params 
    sequence_len: int = 1024
    vocab_size: int = 50304 # originally 50257, nanochat bumps it to the nearest multiple of 64. this should be inferred from the tokenizer though
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embed: int = 768

@register_dataclass
@dataclass
class MLP:
    c_fc: jax.Array # eE - [n_embed, 4 * n_embed]
    c_proj: jax.Array # Ee - [4 * n_embed, n_embed]


@register_dataclass
@dataclass
class Attention:
    c_q: jax.Array # eQ [n_embed, n_head * head_dim]
    c_k: jax.Array # eK [n_embed, n_kv_head * head_dim]
    c_v: jax.Array # eK [n_embed, n_kv_head * head_dim]
    c_proj: jax.Array # ee [n_embed, n_embed]

@register_dataclass
@dataclass
class Block:
    attn: Attention
    mlp: MLP

@partial(
    register_dataclass,
    data_fields=["wte", "h", "lm_head"],
    meta_fields=["cfg"]
)
@dataclass
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
    cfg: GPTConfig

    @staticmethod              
    def init(cfg: GPTConfig, rng: jax.Array) -> "GPT":
        # random state must be explicitly managed in JAX by "splitting"
        # random keys. fold_in_str does this by splitting the base key
        # based on the hash of a given string - in this case the weight name.
        std = 0.02
        wte = jax.random.normal(fold_in_str(rng, "wte"), (cfg.vocab_size, cfg.n_embed)) * std
        h = []
        for i in range(cfg.n_layer):
            # regular mean-zero, std 0.02 weight initialisation
            # TODO - why does nanochat do some funky initialiation?
            head_dim = cfg.n_embed // cfg.n_head
            resid_std = std / math.sqrt(2 * cfg.n_layer)
    
            attn = Attention(
                c_q=jax.random.normal(fold_in_str(rng, f"{i}_c_q"), (cfg.n_embed, cfg.n_head * head_dim)) * std,
                c_k=jax.random.normal(fold_in_str(rng, f"{i}_k"), (cfg.n_embed, cfg.n_kv_head * head_dim)) * std,
                c_v=jax.random.normal(fold_in_str(rng, f"{i}_v"), (cfg.n_embed, cfg.n_kv_head * head_dim)) * std,
                c_proj=jax.random.normal(fold_in_str(rng, f"{i}_o"), (cfg.n_embed, cfg.n_embed)) * resid_std,
            )
    
            mlp = MLP(
                c_fc=jax.random.normal(fold_in_str(rng, f"{i}_c_fc"), (cfg.n_embed, 4 * cfg.n_embed)) * std,
                c_proj=jax.random.normal(fold_in_str(rng, f"{i}_c_proj"), (4 * cfg.n_embed, cfg.n_embed)) * resid_std,
            )
            h.append(Block(attn=attn, mlp=mlp))

        # TODO - why does nanochat zero out classifier weights and c_proj in mlps?
        # ANSWER: see modded-nanogpt
        lm_head = jax.random.normal(fold_in_str(rng, "lm_head"), (cfg.n_embed, cfg.vocab_size)) * std
        return GPT(
            wte=wte,
            h=h,
            lm_head=lm_head,
            cfg=cfg
        )
        
    def forward(self, idx: jax.Array, compute_dtype: jnp.dtype):
        cfg = self.cfg        
        b, s =  idx.shape
        # one slight downside to working in pure JAX is that we don't have a nice
        # amp autocast context manager like in torch to automatically handle
        # mixed precision.
        # (recall: mixed precision means we keep model weights and gradients in fp32
        # and perform gradient updates in fp32, but we use bf16 for our forward pass
        # for improved speed)
        # This means we have to manually handle precision everywhere but it comes
        # with the benefit of finer-grained control over mixed precision.
        
        # project our tokens into embedding space
        x = self.wte[idx].astype(compute_dtype) # jnp.einsum("bs,ve->bse", idx, self.wte.astype(compute_dtype))
        # create our causal mask ( not needed for jax sdpa)
        #mask = jnp.tril(jnp.ones((s, s), dtype=jnp.bool))[None, :, :, None]
        
        for block in self.h:
            attn, mlp = block.attn, block.mlp
            h = self.cfg.n_embed // self.cfg.n_head
            attn_in = rms_norm(x)

            ### causal self attention
            q = jnp.einsum("bse,eQ->bsQ", attn_in, attn.c_q.astype(compute_dtype)).reshape(b, s, cfg.n_head, h)
            k = jnp.einsum("bse,eK->bsK", attn_in, attn.c_k.astype(compute_dtype)).reshape(b, s, cfg.n_kv_head, h)
            v = jnp.einsum("bse,eK->bsK", attn_in, attn.c_v.astype(compute_dtype)).reshape(b, s, cfg.n_kv_head, h)
            
            # apply rotary embeddings (on-the-fly) to our queries, keys, and values
            # we operate on every pair of dimensions, so stride our embed dim by 2
            # we're fixing base_theta to be 10K for now
            channel_range = jnp.arange(0, h, 2, dtype=jnp.float32)
            inv_freq = 1.0 / (1e4 ** (channel_range / h))
            # token sequence positions: 0,1,2,...,s
            t = jnp.arange(s, dtype=jnp.float32)
            # calculate the angle by which each pair of dimensions should rotate at
            # each position, then unsqueeze so we can broadcast along the n_head dim
            theta = jnp.einsum("s,c->sc", t, inv_freq)
            cos, sin = jnp.cos(theta)[:, None, :], jnp.sin(theta)[:, None, :]

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            # QK norm
            q = rms_norm(q)
            k = rms_norm(k)

            # scaled dot product attention
            attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
            # scores = jnp.einsum("bsqh,bSkh->bsSh", q, k)
            # scores =  jnp.where(mask, scores, -1e10) / jnp.sqrt(h)
            # # we typically softmax in fp32 
            # probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(compute_dtype)
            # attn_out = jnp.einsum("bsSh,bskh->bskh", probs, v).astype(compute_dtype)
            attn_out = attn_out.reshape(b, s, cfg.n_embed)
            attn_out = jnp.einsum("bse,eE->bsE", attn_out, attn.c_proj.astype(compute_dtype))

            # residual connection with the pre-norm block input
            x = x + attn_out
            
            ### mlp
            mlp_in = rms_norm(x)
            mlp_out = jnp.einsum("bse,eE->bsE", mlp_in, mlp.c_fc.astype(compute_dtype))
            mlp_out = jax.nn.gelu(mlp_out) # TODO why does nanochat use relu^2? ANSWER: see modded-nanogpt
            mlp_out = jnp.einsum("bsE,Ee->bse", mlp_out, mlp.c_proj.astype(compute_dtype))
            x = x + mlp_out
            
        x = rms_norm(x)
        # note: we calculate logits and CE in fp32, so no weight downcasting here
        logits = jnp.einsum("bse,ev->bsv", x.astype(jnp.float32), self.lm_head)
        return logits

def estimate_flops(model: GPT):
    num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
    wte_params = model.wte.size
    l, h, q, t = model.cfg.n_layer, model.cfg.n_head, model.cfg.n_embed // model.cfg.n_head, model.cfg.sequence_len
    num_flops_per_token = 6 * (num_params - wte_params) + 12 * l * h * q * t
    return num_flops_per_token


def calculate_loss(idx: jax.Array, targets: jax.Array, model: GPT, dtype: jnp.dtype) -> jax.Array:
    # TODO try logit softcapping
    logits = model.forward(idx, dtype)
    # cross entropy loss using logsumexp
    logsumexp = jax.nn.logsumexp(logits, axis=-1)
    # TODO add support for ignore index
    loss = -jnp.take_along_axis(logits, targets[:, :, None], axis=-1).squeeze(-1) + logsumexp
    loss = loss.mean()
    return loss

rng = jax.random.key(42)
model = GPT.init(
    GPTConfig(
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embed=192,
        vocab_size=1024
    ),
    rng
)

# state = AdamW.init(model)
# # out = model.forward(jnp.ones((10, 1024), dtype=jnp.uint32))
# idx = jnp.ones((4, 256)).astype(jnp.uint32)
# targets = jnp.ones((4, 1)).astype(jnp.uint32)
# grad_fun = jax.value_and_grad(calculate_loss, argnums=2)

# def train_step(idx, targets, model, state):
#     loss, grads = grad_fun(idx, targets, model)
#     updates, state = state.update(model, grads, 1e-3)
#     model = jax.tree.map(lambda p, u: p - u, model, updates)

#     return model, state, loss
# model_, state_, loss = train_step(idx, targets, model, state)
# x = 10
