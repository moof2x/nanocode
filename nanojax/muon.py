from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from nanojax.common import print0
from nanojax.gpt import GPT, Block


@partial(
    register_dataclass,
    data_fields=["adamw_mu", "adamw_nu", "mu", "step"],
    meta_fields=["b_1", "b_2", "eps", "wd", "ns_steps", "wte_lr", "lm_head_lr", "lr"]
)
@dataclass
class Muon:
    ### Muon w/AdamW
    # Muon momentum states for out decoder blocks
    mu: list[Block]
    # AdamW states for embedding/lm_head respectively
    adamw_mu: tuple[jax.Array, jax.Array]
    adamw_nu: tuple[jax.Array, jax.Array]

    # training step counter for scaling learning rates/momentum
    step: jax.Array

    # AdamW hyperparameters
    b_1: float = 0.8
    b_2: float = 0.95
    eps: float = 1e-10
    wd: float = 0.0

    # number of newton-shulz iteration steps
    ns_steps: int = 5 
    # we need seperate learning rates for our embeddings/lm_head/matrices
    wte_lr: float = 0.2
    lm_head_lr: float = 0.004
    lr: float = 0.02


    @staticmethod              
    def init(model: GPT, **kwargs):
        adamw_mu = (model.wte * 0.0, model.lm_head * 0.0)
        adamw_nu = (model.wte * 0.0, model.lm_head * 0.0)
        mu = jax.tree.map(lambda p: p * 0.0, model.h)

        optimizer = Muon(mu=mu, adamw_mu=adamw_mu, adamw_nu=adamw_nu, step=jnp.array(1, dtype=jnp.int32), **kwargs)
        # lr AdamW scaling
        dmodel_lr_scale = (model.cfg.n_embed / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model.cfg.n_embed}/768) = {dmodel_lr_scale:.6f}")
        return replace(optimizer, wte_lr=optimizer.wte_lr * dmodel_lr_scale, lm_head_lr=optimizer.lm_head_lr * dmodel_lr_scale)

    def update(self, model: GPT, grads: GPT, lr_multiplier: float):
        """
        This function applies Muon updates to all 2D matrices in our model, and AdamW to
        our embedding and classifier layers.
        Since we're returning a single update, u, to be subtracted from our model parameters,
        some of the signage may look slightly different to other implementations.        
        """
        step = self.step
        ### learning rate/momentum updates
        # momentum scheduler
        frac = jnp.minimum(step / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95

        # lr updates
        wte_lr = self.wte_lr * lr_multiplier
        lm_head_lr = self.lm_head_lr * lr_multiplier
        lr = self.lr * lr_multiplier
        
        ### adamw updates
        # first moment estimate
        adamw_mu = jax.tree.map(lambda m, g: m * self.b_1 + (1 - self.b_1) * g, self.adamw_mu, (grads.wte, grads.lm_head))
        # second moment estimate
        adamw_nu = jax.tree.map(lambda v, g: v * self.b_2 + (1 - self.b_2) * jax.lax.square(g), self.adamw_nu, (grads.wte, grads.lm_head))

        # bias corrected estimates
        adamw_mu_ = jax.tree.map(lambda m: m / (1 - (self.b_1 ** step)), adamw_mu)
        adamw_nu_ = jax.tree.map(lambda v: v / (1 - (self.b_2 ** step)), adamw_nu)

        # parameter update with weight decay
        adamw_update = jax.tree.map(
            lambda p, m, v, lr_: (lr_ * m / (jnp.sqrt(v) + self.eps)) + (lr_ * self.wd * p),
            (model.wte, model.lm_head), # p
            adamw_mu_, # m
            adamw_nu_, # v
            (wte_lr, lm_head_lr) # lr_
        )

        ### muon updates
        # mumoentum update
        mu = jax.tree.map(lambda m, g: m * momentum + (1 - momentum) * g, self.mu, grads.h)
        # nesterov update
        v = jax.tree.map(lambda m, g: g * (1 - momentum) + momentum * m, mu, grads.h)

        def newton_shulz(G: jax.Array, ns_steps: int):
            a, b, c = (3.4445, -4.7750,  2.0315)
            transposed = G.shape[-2] > G.shape[-1]
            G = G.astype(jnp.bfloat16)
            if transposed:
                G = G.T
            G = G / (jnp.linalg.norm(G, keepdims=True) + 1e-7)

            def ns_iter(i, X):
                A = X @ X.mT
                B = b * A + c * (A @ A)
                return a * X + B @ X

            G = jax.lax.fori_loop(0, ns_steps, ns_iter, G, unroll=True)
            if transposed:
                G = G.T
            G = G.astype(jnp.float32)
            return G

        # muon update - orthogonalize then scale 
        muon_update = jax.tree.map(lambda u, g: lr * newton_shulz(u, self.ns_steps) * (jnp.maximum(1, g.shape[-2] / g.shape[-1]))**0.5, v, grads.h)

        updates = GPT(
            wte=adamw_update[0],
            h=muon_update,
            lm_head=adamw_update[1],
            cfg=model.cfg,
            attn_impl=model.attn_impl
        )

        # we only need to update our optimizer states and step counter as the rest of the fields are static
        return updates, replace(self, mu=mu, adamw_mu=adamw_mu, adamw_nu=adamw_nu, step=step+1)     


