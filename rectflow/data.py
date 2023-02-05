from typing import Tuple

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, PyTree


def sample(key: PyTree, n_samples: int) -> Tuple[Array, Array]:
    assert n_samples % 2 == 0

    key00, key01, key10, key11 = jr.split(key, 4)
    dimension = 2
    x0 = jnp.concatenate(
        (
            jr.normal(key00, (n_samples // 2, dimension)) + jnp.array((0, 0)),
            jr.normal(key01, (n_samples // 2, dimension)) + jnp.array((0, 10)),
        )
    )
    x1 = jnp.concatenate(
        (
            jr.normal(key10, (n_samples // 2, dimension)) + jnp.array((10, 0)),
            jr.normal(key11, (n_samples // 2, dimension)) + jnp.array((15, 15)),
        )
    )
    return x0, x1
