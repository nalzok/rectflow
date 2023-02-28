import jax.random as jr
import jax.numpy as jnp

from rectflow.data import gaussian_llk, gaussian_mixture, gaussian_mixture_llk
from rectflow.velocity import Velocity
from rectflow.train import train
from rectflow.ode import flow


def main(n_samples: int, learning_rate: float, n_epochs: int):
    key = jr.PRNGKey(42)
    key, key_x1, key_perm = jr.split(key, 3)
    dim = 64
    n_components = 8
    components, x1 = gaussian_mixture(key_x1, dim, n_components, n_samples)
    x1 = jr.permutation(key_perm, x1)
    x1_mean = jnp.mean(x1, axis=0)
    x1_cov = jnp.cov(x1, rowvar=False)
    # x1_mean = jnp.zeros((dim,))
    # x1_cov = jnp.eye(dim)

    for reflow in range(1):
        key, key_model, key_train, key_noise = jr.split(key, 4)

        model = Velocity(key_model, dim)
        model = train(key_train, model, x1, x1_mean, x1_cov, learning_rate, n_epochs)

        # pi_0 -> pi_1
        x0 = jr.multivariate_normal(key_noise, x1_mean, x1_cov, x1.shape[:-1])
        next_x1 = flow(model, x0, 1e-3, True)
        llk = gaussian_mixture_llk(next_x1, components, dim)
        print(f"forward llk: {llk}")

        # pi_1 -> pi_0
        x0_rev = flow(model, x1, 1e-3, False)
        llk = gaussian_llk(x0_rev, x1_mean, x1_cov)
        print(f"backward llk: {llk}")

        x1 = next_x1


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_enable_x64", True)

    n_samples = 500
    learning_rate = 1e-4
    n_epochs = 1000
    main(n_samples, learning_rate, n_epochs)
