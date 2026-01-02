# there's a nice paradigm in JAX where you can register dataclasses as PyTrees.
# In JAX you can apply many tensor-adjacent operations to PyTrees, all of which
# are well-supported by the compiler.
# This is great for us. In this implementation, we'll separate state (model, optimizer params)
# from the modeling code itself. We can define the state as a PyTree and all modeling
# code will be applying transformations on this PyTree (or mapping data through it).
# if you're wondering about the variable names here and in nanochat
# they come from https://github.com/openai/gpt-2/blob/master/src/model.py

import itertools
import math
import operator
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

# einsum notation:
#    b: batch
#    n: n_layer
#    s: sequence len
#    v: vocab_size
#    e: n_embed, embedding dim
#    q: n_head, number of query heads
#    k: n_kv_head, number of KV-heads
#    h: head_dim
#    Q: n_head * head_dim 
#    K: n_kv_head * head_dim

@register_dataclass
@dataclass
class KVCache:
    k: jax.Array # nbskh
    v: jax.Array # nbskh
    pos: jax.Array # current  position in the sequence

    def init(batch_size : int, max_seq_len: int, n_layer: int, embed_dim: int, n_head: int, n_kv_head: int, compute_dtype: jnp.dtype):
        head_dim = embed_dim // n_head
        return KVCache(
            k=jnp.zeros((n_layer, batch_size, max_seq_len, n_kv_head, head_dim), dtype=compute_dtype),
            v=jnp.zeros((n_layer, batch_size, max_seq_len, n_kv_head, head_dim), dtype=compute_dtype),
            pos=jnp.array(0, dtype=jnp.int32)
        )

    def update(self, k: jax.Array, v: jax.Array, layer_idx: int):
        # k,v: bskh
        s = k.shape[1]
        # store our updated kv cache values
        k = jax.lax.dynamic_update_slice(self.k, k[None], (layer_idx, 0, self.pos, 0, 0))
        v = jax.lax.dynamic_update_slice(self.v, v[None], (layer_idx, 0, self.pos, 0, 0))
        return k[layer_idx], v[layer_idx], replace(self, k=k, v=v)

    def forward_pos(self, s: int):
        # forward cache pos sequence len positions along
        return replace(self, pos=self.pos + s)
        

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
    return jnp.concatenate([y1, y2], axis=-1)


@dataclass
class GPTConfig:
    # default GPT2-117M params 
    sequence_len: int = 1024
    vocab_size: int = 50304 # originally 50257, nanochat bumps it to the nearest multiple of 64.
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
        # random keys. fold_in does this by splitting the base key based on
        # a given integer
        def init_linear(rng, shape):
            # fan-in-fan-out initialization https://arxiv.org/pdf/2310.17813
            fan_in, fan_out = shape
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            return jax.random.normal(rng, shape) * std

        key = map(partial(jax.random.fold_in, rng), itertools.count())
        # mean 0, std 1 initialization for embedding layer
        wte = jax.random.normal(next(key), (cfg.vocab_size, cfg.n_embed)) 
        h = []
        head_dim = cfg.n_embed // cfg.n_head
        for i in range(cfg.n_layer):
            # modded-nanogpt suggestion: zero out c_proj in attn and mlp layers
            attn = Attention(
                c_q=init_linear(next(key), (cfg.n_embed, cfg.n_head * head_dim)),
                c_k=init_linear(next(key), (cfg.n_embed, cfg.n_kv_head * head_dim)),
                c_v=init_linear(next(key), (cfg.n_embed, cfg.n_kv_head * head_dim)),
                c_proj = jnp.zeros((cfg.n_embed, cfg.n_embed))
            )
    
            mlp = MLP(
                c_fc=init_linear(next(key), (cfg.n_embed, 4 * cfg.n_embed)),
                c_proj=jnp.zeros((4 * cfg.n_embed, cfg.n_embed))
            )
            h.append(Block(attn=attn, mlp=mlp))

        # modded-nanogpt suggestion: zero out classifier weights
        lm_head = jnp.zeros((cfg.n_embed, cfg.vocab_size))
        return GPT(
            wte=wte,
            h=h,
            lm_head=lm_head,
            cfg=cfg
        )
        
    def forward(self, idx: jax.Array, mask: jax.Array = None, compute_dtype: jnp.dtype = jnp.bfloat16, kv_cache: KVCache = None):
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
        x = self.wte[idx].astype(compute_dtype)
        x = rms_norm(x)

        h = self.cfg.n_embed // self.cfg.n_head
        # prepare our rotary position embeddings once
        channel_range = jnp.arange(0, h, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (1e4 ** (channel_range / h))
        is_causal = mask is None
        
        # token sequence positions: 0,1,2,...,s
        if kv_cache is not None:
            # if we're decoding we only need to generate embeddings for positions we
            # haven't seen yet
            t = jnp.arange(s, dtype=jnp.float32) + kv_cache.pos
        else:
            t = jnp.arange(s, dtype=jnp.float32)
        
        # calculate the angle by which each pair of dimensions should rotate at
        # each position, then unsqueeze so we can broadcast along the n_head dim
        theta = jnp.einsum("s,c->sc", t, inv_freq)
        # we operate on every pair of dimensions, so stride our embed dim by 2
        # we're fixing base_theta to be 10K for now
        cos, sin = jnp.cos(theta)[:, None, :], jnp.sin(theta)[:, None, :]
        cos, sin = cos.astype(compute_dtype), sin.astype(compute_dtype)
        
        for i, block in enumerate(self.h):
            attn, mlp = block.attn, block.mlp
            attn_in = rms_norm(x)

            ### causal self attention
            q = jnp.einsum("bse,eQ->bsQ", attn_in, attn.c_q.astype(compute_dtype)).reshape(b, s, cfg.n_head, h)

            # k,v: bse -> bsK -> bskh
            k = jnp.einsum("bse,eK->bsK", attn_in, attn.c_k.astype(compute_dtype)).reshape(b, s, cfg.n_kv_head, h)
            v = jnp.einsum("bse,eK->bsK", attn_in, attn.c_v.astype(compute_dtype)).reshape(b, s, cfg.n_kv_head, h)
        
            # apply rotary embeddings (on-the-fly) to our queries, keys, and values
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            if kv_cache is not None:
                k, v, kv_cache = kv_cache.update(k, v, i)

            # QK norm
            q = rms_norm(q)
            k = rms_norm(k)

            # scaled dot product attention
            attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=is_causal, mask=mask)
            attn_out = attn_out.reshape(b, s, cfg.n_embed)
            attn_out = jnp.einsum("bse,eE->bsE", attn_out, attn.c_proj.astype(compute_dtype))

            # residual connection with the pre-norm block input
            x = x + attn_out
            
            ### mlp
            mlp_in = rms_norm(x)
            mlp_out = jnp.einsum("bse,eE->bsE", mlp_in, mlp.c_fc.astype(compute_dtype))
            mlp_out = jax.lax.square(jax.nn.relu(mlp_out)) # modded-nanogpt introduced the use of relu^2
            mlp_out = jnp.einsum("bsE,Ee->bse", mlp_out, mlp.c_proj.astype(compute_dtype))
            x = x + mlp_out
        
        if kv_cache is not None:
            kv_cache = kv_cache.forward_pos(s)
            
        x = rms_norm(x)
        # perform logit softcapping
        softcap = 15
        logits = jnp.einsum("bse,ev->bsv", x, self.lm_head.astype(compute_dtype))
        # note: we CE in fp32, so no mixed precision here
        logits = softcap * jax.nn.tanh(logits / softcap)
        # i hate runtime-defined function signatures but this is the price we pay for JIT - all state is immutable
        return logits.astype(jnp.float32), kv_cache 

def estimate_flops(model: GPT):
    num_params = jax.tree.reduce(operator.add, jax.tree.map(jnp.size, model))
    wte_params = model.wte.size
    l, h, q, t = model.cfg.n_layer, model.cfg.n_head, model.cfg.n_embed // model.cfg.n_head, model.cfg.sequence_len
    num_flops_per_token = 6 * (num_params - wte_params) + 12 * l * h * q * t
    return num_flops_per_token


def calculate_loss(idx: jax.Array, targets: jax.Array, model: GPT, ignore_idx: int=-1,  compute_dtype: jnp.dtype=jnp.bfloat16, reduce: bool=True) -> jax.Array:
    logits, _ = model.forward(idx, compute_dtype=compute_dtype)
    # cross entropy loss using logsumexp
    logsumexp = jax.nn.logsumexp(logits, axis=-1)
    valid_targets = jnp.not_equal(targets, ignore_idx)
    safe_targets = jnp.where(valid_targets, targets, 0)
    loss = -jnp.take_along_axis(logits, safe_targets[:, :, None], axis=-1).squeeze(-1) + logsumexp
    loss = jnp.where(valid_targets, loss, 0.0)
    if reduce:
        loss = loss.sum() / (valid_targets.sum() + 1e-9)
    return loss
