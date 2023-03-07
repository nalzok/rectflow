from typing import Tuple

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from rectflow.velocity import Velocity
from rectflow.train import train
from rectflow.ode import flow


def main(n_samples: int, learning_rate: float, n_epochs: int):
    key = jr.PRNGKey(42)
    key, key_x1, key_perm = jr.split(key, 3)
    cond_dim = 1
    z_dim = 64
    n_components = 8

    # Huber
    def f(x):
        delta = 1
        norm = jnp.linalg.norm(x)
        loss = jnp.where(norm < delta, 1/2 * norm**2, delta * (norm - delta/2))
        return loss

    # Power
    # alpha = 1
    # f = lambda x: 1/(1+alpha) * jnp.sum(x**(1+alpha))

    cond = jnp.zeros((n_samples, cond_dim))
    components, x1 = gaussian_mixture(key_x1, z_dim, n_components, n_samples)
    x1 = jr.permutation(key_perm, x1)


    # z0 = z0_mean + sample @ z0_factor.T
    # jnp.mean(z0, axis=0) = z0_mean + jnp.mean(sample, axis=0)
    # jnp.cov(z0, rowvar=False) = z0_factor @ jnp.cov(sample, rowvar=False) @ z0_factor.T

    z0_mean = x1_mean = jnp.mean(x1, axis=0)
    x1_cov = jnp.cov(x1, rowvar=False)
    z0_factor = jnp.linalg.cholesky(x1_cov)


    for reflow in range(1):
        key, key_model, key_train, key_noise = jr.split(key, 4)

        model = Velocity(key_model, cond_dim, z_dim)
        z0_gen = jr.normal
        model = train(key_train, model, cond, z0_gen, z0_mean, z0_factor, x1, f, learning_rate, n_epochs)

        # pi_0 -> pi_1
        x0 = jr.multivariate_normal(key_noise, x1_mean, x1_cov, x1.shape[:-1])
        next_x1 = flow(model, cond, x0, 1e-3, True)
        llk = gaussian_mixture_llk(next_x1, components, z_dim)
        print(f"forward llk: {llk}")

        # pi_1 -> pi_0
        x0_rev = flow(model, cond, x1, 1e-3, False)
        llk = gaussian_llk(x0_rev, x1_mean, x1_cov)
        print(f"backward llk: {llk}")

        x1 = next_x1


def gaussian_llk(x: Array, mean: Array, cov: Array) -> Array:
    llk = jnp.sum(jsp.stats.multivariate_normal.logpdf(x, mean, cov))
    return llk


def gaussian_mixture(key: PyTree, dim: int, n_components: int, n_samples: int) -> Tuple[Array, Array]:
    key0, key1, key2 = jr.split(key, 3)
    components = n_components * jr.normal(key0, (n_components, dim))
    indices = jr.categorical(key1, jnp.zeros(n_components,), shape=(n_samples,))
    x1 = components[indices] + jr.normal(key2, (n_samples, dim))
    return components, x1


def gaussian_mixture_llk(x: Array, components: Array, dim: int) -> Array:
    llk_mixture = jax.vmap(lambda component: jsp.stats.multivariate_normal.logpdf(x, component, jnp.eye(dim)))(components)
    llk = jnp.sum(jax.nn.logsumexp(llk_mixture))
    return llk


if __name__ == "__main__":
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    n_samples = 500
    learning_rate = 1e-4
    n_epochs = 1000
    main(n_samples, learning_rate, n_epochs)
