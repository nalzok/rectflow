from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


class Velocity(eqx.Module):
    layers: Sequence

    def __init__(self, key, cond_dim, z_dim):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(cond_dim + z_dim + 1, 500, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(500, 500, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(500, 500, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(500, z_dim, key=key4),
        ]

    def __call__(self, cond, z, t):
        x = jnp.concatenate((cond, z, t))
        for layer in self.layers:
            x = layer(x)
        return x
