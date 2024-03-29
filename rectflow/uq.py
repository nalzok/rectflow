from typing import Tuple
from functools import partial
from math import sqrt

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, PyTree
import matplotlib.pyplot as plt

from rectflow.velocity import Velocity
from rectflow.train import train
from rectflow.ode import flow


def main(n_samples: int, learning_rate: float, n_epochs: int):
    key = jr.PRNGKey(42)
    key, key_z = jr.split(key)
    cond_dim = 1
    z_dim = 1

    # Power
    alpha = 1
    f = lambda x: 1/(1+alpha) * jnp.sum(x**(1+alpha))

    cond = jnp.linspace(-10, 10, n_samples * cond_dim).reshape((-1, cond_dim))
    z1, *info = ftrue(key_z, cond)
    z1_mean = jnp.mean(z1, axis=0)
    z1_cov = jnp.cov(z1, rowvar=False).reshape((z_dim, z_dim))

    key, key_cond, key_z1 = jr.split(key, 3)
    noise_cond = jr.uniform(key_cond, cond.shape, minval=-1, maxval=1)
    noise_z1 = jnp.abs(noise_cond)/10 * jr.normal(key_z1, z1.shape)
    cond = cond + noise_cond
    z1 = z1 + noise_z1
    
    fig = plt.figure()
    plt.plot(cond, z1, '.', markersize=1)
    # plt.plot(cond, info[0], '--k')
    plt.savefig("figures/uq2.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    for reflow in range(1):
        key, key_model, key_train, key_z0 = jr.split(key, 4)

        model = Velocity(key_model, cond_dim, z_dim)
        z0_gen = partial(jr.uniform, minval=-sqrt(3), maxval=sqrt(3))
        model = train(key_train, model, cond, z0_gen, z1_mean, z1_cov, z1, f, learning_rate, n_epochs)

        # pi_0 -> pi_1
        cond0 = jnp.linspace(-20, 20, n_samples * cond_dim).reshape((-1, cond_dim))
        z0 = jr.multivariate_normal(key_z0, z1_mean, z1_cov, z1.shape[:-1])
        next_z1 = flow(model, cond0, z0, 1e-3, True)
    
        fig = plt.figure()
        plt.plot(cond0, z0, ".", markersize=1)
        plt.savefig(f"figures/uq2-forward-{reflow}-before.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    
        fig = plt.figure()
        plt.plot(cond0, next_z1, ".", markersize=1)
        plt.savefig(f"figures/uq2-forward-{reflow}-after.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        # pi_1 -> pi_0
        z0_rev = flow(model, cond, z1, 1e-3, False)
    
        fig = plt.figure()
        plt.plot(cond, z1, ".", markersize=1)
        plt.savefig(f"figures/uq2-backward-{reflow}-before.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    
        fig = plt.figure()
        plt.plot(cond, z0_rev, ".", markersize=1)
        plt.savefig(f"figures/uq2-backward-{reflow}-after.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        z1 = next_z1


def ftrue(key: PyTree, cond: Array) -> Tuple[Array, Array, Array]:
  fmean = jnp.cos(cond)
  fstd = jnp.abs(jnp.maximum(cond, 0))**.5/10
  fnoise = jr.normal(key, fmean.shape)**2
  fsign = 1
  z1 = (fmean+fstd*fnoise) * fsign
  return z1, fmean, fstd


if __name__ == "__main__":
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    n_samples = 10000
    learning_rate = 1e-4
    n_epochs = 3000
    main(n_samples, learning_rate, n_epochs)
