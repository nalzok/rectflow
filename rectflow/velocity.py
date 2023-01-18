from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


class Velocity(eqx.Module):
    layers: Sequence

    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Linear(3, 1024, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(1024, 1024, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(1024, 1024, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(1024, 1024, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(1024, 2, key=key5),
        ]

    def __call__(self, z, t):
        x = jnp.concatenate((z, t))
        for layer in self.layers:
            x = layer(x)
        return x
