from typing import Tuple

import numpy as np
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array
import matplotlib.pyplot as plt

from rectflow.data import sample
from rectflow.velocity import Velocity
from rectflow.train import train
from rectflow.path import path


def main(n_samples: int, learning_rate: float, n_epochs: int):
    key = jr.PRNGKey(42)
    key, key_sample = jr.split(key)
    x0, x1 = sample(key_sample, n_samples)

    key, key_perm = jr.split(key)
    x1 = jr.permutation(key_perm, x1)

    colors = plt.get_cmap("tab10").colors
    for reflow in range(1):
        num_stages = 2
        key, *key_stages = jr.split(key, 1 + 2 * num_stages)

        fig = plt.figure(figsize=(12, 6))
        plt.scatter(x0[:, 0], x0[:, 1], s=1, color=colors[0])
        plt.scatter(x1[:, 0], x1[:, 1], s=1, color=colors[num_stages+1])

        outset = x0
        for stage in range(num_stages):
            key_model, key_train = key_stages[2*stage : 2*stage+2]
            model = Velocity(key_model)
            model = train(key_train, model, outset, x1, learning_rate, n_epochs)

            next_outset, traces = transport(model, outset, 1e-3)
            plt.scatter(next_outset[:, 0], next_outset[:, 1], s=1, color=colors[stage+1])
            for i in range(x0.shape[0]):
                plt.plot(traces[i, :, 0], traces[i, :, 1], color=colors[stage+1], alpha=0.05)

            outset = next_outset

        plt.xlim((-10, 20))
        plt.ylim((-10, 20))
        plt.savefig(f"figures/reflow{reflow+1}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def transport(model: Velocity, z0: Array, dt0: float) -> Tuple[Array, Array]:
    tz0 = np.empty_like(z0)
    traces = np.empty((z0.shape[0], int(1 / dt0), z0.shape[1]))
    for i in range(n_samples):
        solution = path(model, z0[i, :], dt0)
        traces[i, :, :] = solution.ys
        tz0[i, :] = solution.ys[-1]

    tz0 = jnp.array(tz0)
    traces = jnp.array(traces)

    return tz0, traces


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_enable_x64", True)

    n_samples = 500
    learning_rate = 1e-4
    n_epochs = 1000
    main(n_samples, learning_rate, n_epochs)
