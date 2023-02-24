from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
import equinox as eqx
import optax

from rectflow.velocity import Velocity


def train(
    key: PyTree,
    model: Velocity,
    tz0: Array,
    learning_rate: float,
    n_epochs: int,
) -> Velocity:
    n_samples, _ = tz0.shape

    @eqx.filter_jit
    def train_step(
        key: PyTree, model: Velocity, opt_state: optax.OptState
    ) -> Tuple[Array, Velocity, optax.OptState]:
        key_noise, key_t = jr.split(key)
        z0 = jr.normal(key_noise, tz0.shape)
        v = tz0 - z0
        t = jr.uniform(key_t, (n_samples, 1))
        z_t = z0 + t * v

        @eqx.filter_value_and_grad
        def loss_fn(model):
            fitted = jax.vmap(model)(z_t, t)
            residual = v - fitted
            return jnp.mean(residual**2)

        loss, grads = loss_fn(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return loss, model, opt_state

    optim = optax.adamw(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for i in range(n_epochs):
        key_step = jr.fold_in(key, i)

        key_step, key_perm = jr.split(key_step)
        tz0 = jr.permutation(key_perm, tz0)

        loss, model, opt_state = train_step(key_step, model, opt_state)
        if i % 100 == 0:
            print(f"Iteration #{i+1}, {loss = }")

    return model
