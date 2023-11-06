__all__ = ["invert_matrix", "invert_diagonal"]
from jax import scipy as jsc
from jax import numpy as jnp
from typing import Tuple
from jaxtyping import Array, Float


def invert_matrix(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    L = jsc.linalg.cho_factor(A)
    A_inv = jsc.linalg.cho_solve(L, jnp.tile(jnp.eye(A.shape[1])[None], (len(A), 1, 1)))
    ln_det_A = 2.0 * jnp.sum(jnp.log(L[0].diagonal(axis1=-1, axis2=-2)), axis=1)
    return A_inv, ln_det_A


def invert_diagonal(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A_inv = 1.0 / A.diagonal(axis1=1, axis2=2)[:,:,None] * jnp.eye(A.shape[1])
    ln_det_A = jnp.sum(jnp.log(A.diagonal(axis1=1, axis2=2)), axis=1)
    return A_inv, ln_det_A
