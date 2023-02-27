from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, Solution

from rectflow.velocity import Velocity


@eqx.filter_jit
def solve(model: Velocity, z0: Array, dt0: float, ts: Array, forward: bool) -> Solution:
    def f(t, z, args):
        del args
        v = model(z, jnp.expand_dims(t, 0))
        return v if forward else -v

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=ts)
    solution = diffeqsolve(term, solver, 0, 1, dt0, z0, saveat=saveat)

    return solution


def flow(model: Velocity, z0: Array, dt0: float, forward: bool) -> Tuple[Array, Array]:
    ts = jnp.linspace(0, 1, int(1 / dt0))

    def path(z0_i):
        soln = solve(model, z0_i, dt0, ts, forward)
        return soln.ys

    traces = jax.vmap(path)(z0)

    return ts, traces
