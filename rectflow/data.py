import jax.random as jr
from jaxtyping import Array, PyTree


def gaussian_mixture_2d(key: PyTree, n_samples: int) -> Array:
    key0, key1 = jr.split(key)
    x1 = 10 * jr.bernoulli(key0, 0.5, (n_samples, 2)) - 5
    x1 = x1 + jr.normal(key1, (n_samples, 2))
    return x1
