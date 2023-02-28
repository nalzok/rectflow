from typing import Tuple

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree


def gaussian_llk(x: Array, mean: Array, cov: Array) -> Array:
    llk = jnp.sum(jsp.stats.multivariate_normal.logpdf(x, mean, cov))
    return llk


def gaussian_mixture(key: PyTree, dim: int, n_components: int, n_samples: int) -> Tuple[Array, Array]:
    key0, key1, key2 = jr.split(key, 3)
    components = jr.normal(key0, (n_components, dim))
    indices = jr.categorical(key1, jnp.zeros(n_components,), shape=(n_samples,))
    x1 = components[indices] + jr.normal(key2, (n_samples, dim))
    return components, x1


def gaussian_mixture_llk(x: Array, components: Array, dim: int) -> Array:
    llk_mixture = jax.vmap(lambda component: jsp.stats.multivariate_normal.logpdf(x, component, jnp.eye(dim)))(components)
    llk = jnp.sum(jax.nn.logsumexp(llk_mixture))
    return llk
