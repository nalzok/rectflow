import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, Solution

from rectflow.velocity import Velocity


@eqx.filter_jit
def solve(model: Velocity, cond: Array, z0: Array, dt0: float, forward: bool) -> Solution:
    def f(t, z, args):
        del args
        v = model(cond, z, jnp.expand_dims(t, 0))
        return v if forward else -v

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(t1=True)
    solution = diffeqsolve(term, solver, 0, 1, dt0, z0, saveat=saveat)

    return solution


def flow(model: Velocity, cond: Array, z0: Array, dt0: float, forward: bool) -> Array:
    def path(cond_i, z0_i):
        soln = solve(model, cond_i, z0_i, dt0, forward)
        return soln.ys[0, :]

    traces = jax.vmap(path)(cond, z0)

    return traces
