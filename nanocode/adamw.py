from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from nanocode.gpt import GPT


@partial(
    register_dataclass,
    data_fields=["mu", "nu"],
    meta_fields=["b_1", "b_2", "eps", "wd", "step"]
)
@dataclass
class AdamW:
    mu: GPT
    nu: GPT
    b_1: float = 0.9
    b_2: float = 0.95
    eps: float = 1e-10
    wd: float = 0.0
    step: int = 1

    @staticmethod              
    def init(model: GPT, b_1: float=0.9, b_2: float=0.95, eps: float=1e-10, wd: float=0.0):
        mu = jax.tree.map(lambda p: p * 0.0, model)
        nu = jax.tree.map(lambda p: p * 0.0, model)

        return AdamW(mu, nu, b_1, b_2, eps, wd)

    def update(self, model: GPT, grads: GPT, lr: float, step: int):
        # update internal statistics based on gradients
        # returns new updates and updated state
        mu = jax.tree.map(lambda m, grad: m * self.b_1 + (1 - self.b_1) * grad, self.mu, grads)
        nu = jax.tree.map(lambda v, grad: v * self.b_2 + (1 - self.b_2) * jax.lax.square(grad), self.nu, grads)

        # bias updates
        mu_ = jax.tree.map(lambda m: m / (1 - (self.b_1 ** step)), mu)
        nu_ = jax.tree.map(lambda v: v / (1 - (self.b_2 ** step)), nu)
        
        updates = jax.tree.map(lambda p, m, v: (lr * m / (jnp.sqrt(v) + self.eps)) - (lr * self.wd * p),model, mu_, nu_)

        return updates, AdamW(mu=mu, nu=nu, step=self.step + 1)

    

