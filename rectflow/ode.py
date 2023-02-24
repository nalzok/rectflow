from typing import Tuple

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, Solution

from rectflow.velocity import Velocity


@eqx.filter_jit
def solve(model: Velocity, z0: Array, dt0: float, forward: bool) -> Solution:
    def f(t, z, args):
        del args
        v = model(z, jnp.expand_dims(t, 0))
        return v if forward else -v

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.linspace(0, 1, int(1 / dt0)))
    solution = diffeqsolve(term, solver, 0, 1, dt0, z0, saveat=saveat)

    return solution


def flow(model: Velocity, z0: Array, dt0: float, forward: bool) -> Tuple[Array, Array]:
    ts = np.empty((z0.shape[0], int(1 / dt0)))
    traces = np.empty((z0.shape[0], int(1 / dt0), z0.shape[1]))
    for i in range(z0.shape[0]):
        solution = solve(model, z0[i, :], dt0, forward)
        ts[i, :] = solution.ts
        traces[i, :, :] = solution.ys

    ts = jnp.array(ts)
    traces = jnp.array(traces)

    return ts, traces
