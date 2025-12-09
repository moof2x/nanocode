from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from nanojax.gpt import GPT, Block


@partial(
    register_dataclass,
    data_fields=["adamw_mu", "adamw_nu", "mu"],# "wte_lr", "lm_head_lr", "lr"],
    meta_fields=["b_1", "b_2", "eps", "adamw_wd", "wd", "ns_steps", "wte_lr", "lm_head_lr", "lr"]#, "warmup_ratio", "warmdown_ratio", "step", ]
)
@dataclass
class Muon:
    # Muon w/AdamW
    # num_steps: int
    # Muon momentum states for out decoder blocks
    mu: list[Block]
    # AdamW states for embedding/lm_head respectively
    adamw_mu: tuple[jax.Array, jax.Array]
    adamw_nu: tuple[jax.Array, jax.Array]

    # hyperparameters
    b_1: float = 0.9
    b_2: float = 0.95
    eps: float = 1e-10
    adamw_wd: float = 0.0

    ns_steps: int = 5 # number of newton-shulz iteration steps
    wd: float = 0.0

    # we need seperate learning rates for our embeddings/lm_head/matrices
    wte_lr: float = 3e-4
    lm_head_lr: float = 3e-4
    lr: float = 0.02

    # lr warmup parameters
    # warmup_ratio: float = 0.0
    # warmdown_ratio: float = 0.2

    @staticmethod              
    def init(model: GPT, **kwargs):
        adamw_mu = (model.wte * 0.0, model.lm_head * 0.0)
        adamw_nu = (model.wte * 0.0, model.lm_head * 0.0)
        mu = jax.tree.map(lambda p: p * 0.0, model.h)

        return Muon(mu=mu, adamw_mu=adamw_mu, adamw_nu=adamw_nu, **kwargs)
        # return Muon(adamw_mu, adamw_nu, b_1, b_2, eps, adamw_wd, ns_steps, momentum, wte_lr, lm_head_lr, lr)

    def update(self, model: GPT, grads: GPT, lr_multiplier: float, step: int):
        # update internal statistics based on gradients
        ### adamw updates
        adamw_mu = jax.tree.map(lambda m, g: m * self.b_1 + (1 - self.b_1) * g, self.adamw_mu, (grads.wte, grads.lm_head))
        adamw_nu = jax.tree.map(lambda v, g: v * self.b_2 + (1 - self.b_2) * jax.lax.square(g), self.adamw_nu, (grads.wte, grads.lm_head))

        # adamw bias updates
        adamw_mu_ = jax.tree.map(lambda m: m / (1 - (self.b_1 ** step)), adamw_mu)
        adamw_nu_ = jax.tree.map(lambda v: v / (1 - (self.b_2 ** step)), adamw_nu)
        
        adamw_update = jax.tree.map(lambda p, m, v, lr_: (lr_ * m / (jnp.sqrt(v) + self.eps)) - (lr_ * self.wd * p), (model.wte, model.lm_head), adamw_mu_, adamw_nu_, (self.wte_lr, self.lm_head_lr))

        ### muon updates
        # momentum scheduler
        frac = jnp.minimum(step / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
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

        muon_update = jax.tree.map(lambda u, g: newton_shulz(u, self.ns_steps) * (jnp.maximum(1, g.shape[-2] / g.shape[-1]))**0.5, v, grads.h)

        updates = GPT(
            wte=adamw_update[0],
            h=muon_update,
            lm_head=adamw_update[1],
            cfg=model.cfg
        )

        return updates, replace(self, mu=mu, adamw_mu=adamw_mu, adamw_nu=adamw_nu) #Muon(mu, adamw_mu, adamw_nu, self.b_1, self.b_2, self.eps, self.adamw_wd, self.ns_steps, self.momentum, self.wte_lr, self.lm_head_lr, self.lr)

    


