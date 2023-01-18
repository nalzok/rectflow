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
    tz0 = jr.permutation(key_perm, x1)

    for i in range(2):
        key, key_model, key_train = jr.split(key, 3)
        model = Velocity(key_model)
        model = train(key_train, model, x0, tz0, learning_rate, n_epochs)
        tz0 = trace(model, x0, tz0, f"figures/reflow{i+1}.png")


def trace(model: Velocity, z0: Array, z1: Array, output: str) -> Array:
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.scatter(z0[:, 0], z0[:, 1], s=1, color="red")
    plt.scatter(z1[:, 0], z1[:, 1], s=1, color="blue")

    tz0 = jnp.empty_like(z0)
    for i in range(n_samples):
        solution = path(model, z0[i, :])
        track = solution.ys
        plt.plot(track[:, 0], track[:, 1], color="gray", alpha=0.2)
        tz0 = tz0.at[i, :].set(track[-1])

    plt.xlim((-10, 20))
    plt.ylim((-10, 20))

    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return tz0


if __name__ == "__main__":
    n_samples = 200
    learning_rate = 1e-4
    n_epochs = 10000
    main(n_samples, learning_rate, n_epochs)
