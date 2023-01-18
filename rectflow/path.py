import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, Solution

from rectflow.velocity import Velocity


@eqx.filter_jit
def path(model: Velocity, z0: Array) -> Solution:
    def f(t, z, args):
        del args
        return model(z, jnp.expand_dims(t, 0))

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.linspace(0, 1, 1001))
    solution = diffeqsolve(term, solver, 0, 1, 0.001, z0, saveat=saveat)

    return solution
