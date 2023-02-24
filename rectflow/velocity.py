from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


class Velocity(eqx.Module):
    layers: Sequence

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(3, 256, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(256, 512, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(512, 512, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(512, 256, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(256, 2, key=key3),
        ]

    def __call__(self, z, t):
        x = jnp.concatenate((z, t))
        for layer in self.layers:
            x = layer(x)
        return x
