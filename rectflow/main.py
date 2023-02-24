import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rectflow.data import gaussian_mixture_2d
from rectflow.velocity import Velocity
from rectflow.train import train
from rectflow.ode import flow


def main(n_samples: int, learning_rate: float, n_epochs: int):
    key = jr.PRNGKey(42)
    key, key_x1, key_perm = jr.split(key, 3)
    x1 = gaussian_mixture_2d(key_x1, n_samples)
    x1 = jr.permutation(key_perm, x1)

    colors = plt.get_cmap("tab10").colors
    for reflow in range(1):
        key, key_model, key_train, key_noise = jr.split(key, 4)

        model = Velocity(key_model)
        model = train(key_train, model, x1, learning_rate, n_epochs)

        print("forward")
        z0 = jr.normal(key_noise, x1.shape)
        ts, traces = flow(model, z0, 1e-3, True)

        fig = plt.figure(figsize=(12, 6))
        plt.scatter(z0[:, 0], z0[:, 1], s=1, color=colors[0])
        plt.scatter(x1[:, 0], x1[:, 1], s=1, color=colors[1])
        for i in range(n_samples):
            plt.scatter(traces[i, -1, 0], traces[i, -1, 1], s=1, color=colors[2])
            rgb = (1 - ts[i, :, jnp.newaxis]) * jnp.array(colors[0]) + ts[i, :, jnp.newaxis] * jnp.array(colors[2])
            plt.scatter(traces[i, :, 0], traces[i, :, 1], s=0.1, c=rgb, alpha=0.01)
        plt.xlim((-10, 10))
        plt.ylim((-10, 10))
        plt.axis("equal")
        plt.savefig(f"figures/reflow{reflow+1}-forward.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print("backward")
        ts, traces = flow(model, x1, 1e-3, False)

        fig = plt.figure(figsize=(12, 6))
        plt.scatter(z0[:, 0], z0[:, 1], s=1, color=colors[0])
        plt.scatter(x1[:, 0], x1[:, 1], s=1, color=colors[1])
        for i in range(n_samples):
            plt.scatter(traces[i, -1, 0], traces[i, -1, 1], s=1, color=colors[2])
            rgb = (1 - ts[i, :, jnp.newaxis]) * jnp.array(colors[1]) + ts[i, :, jnp.newaxis] * jnp.array(colors[2])
            plt.scatter(traces[i, :, 0], traces[i, :, 1], s=0.1, c=rgb, alpha=0.01)
        plt.xlim((-10, 10))
        plt.ylim((-10, 10))
        plt.axis("equal")
        plt.savefig(f"figures/reflow{reflow+1}-backward.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_enable_x64", True)

    n_samples = 500
    learning_rate = 1e-4
    n_epochs = 1000
    main(n_samples, learning_rate, n_epochs)
