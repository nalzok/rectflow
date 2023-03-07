from typing import Tuple, Callable

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
    cond: Array,
    z0_gen: Callable,
    z0_mean: Array,
    z0_factor: Array,
    z1: Array,
    f: Callable[[Array], Array],
    learning_rate: float,
    n_epochs: int,
) -> Velocity:
    n_samples, _ = z1.shape

    @eqx.filter_jit
    def train_step(
        key: PyTree, model: Velocity, opt_state: optax.OptState
    ) -> Tuple[Array, Velocity, optax.OptState]:
        key_noise, key_t = jr.split(key)
        samples = z0_gen(key=key_noise, shape=z1.shape)
        z0 = z0_mean + samples @ z0_factor.T
        v_truth = z1 - z0
        t = jr.uniform(key_t, (n_samples, 1))
        z_t = z0 + t * v_truth

        @eqx.filter_value_and_grad
        def loss_fn(model):
            def bregman(x, y):
                return f(x) - f(y) - jnp.dot(jax.grad(f)(y), x - y)

            v_fitted = jax.vmap(model)(cond, z_t, t)
            losses = jax.vmap(bregman)(v_truth, v_fitted)
            return jnp.mean(losses)

        loss, grads = loss_fn(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return loss, model, opt_state

    optim = optax.adamw(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for i in range(n_epochs):
        key_step = jr.fold_in(key, i)

        key_step, key_perm = jr.split(key_step)
        z1 = jr.permutation(key_perm, z1)

        loss, model, opt_state = train_step(key_step, model, opt_state)
        if i % (n_epochs // 10) == 0:
            print(f"Iteration #{i+1}, {loss = }")

    return model
