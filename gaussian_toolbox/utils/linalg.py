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
    A_inv = jnp.concatenate(
        [jnp.diag(mat)[None] for mat in 1.0 / A.diagonal(axis1=1, axis2=2)], axis=0
    )
    ln_det_A = jnp.sum(jnp.log(A.diagonal(axis1=1, axis2=2)), axis=1)
    return A_inv, ln_det_A

def invert_block_matrix(A: Float[Array, "R A A"], B: Float[Array, "R A C"], C: Float[Array, "R C C"], 
                        A_inv: Float[Array, "R A A"] = None, ln_det_A: Float[Array, "R"]=None, 
                        C_inv: Float[Array, "R C C"]= None, ln_det_C: Float[Array, "R"]=None):
    if A_inv != None and ln_det_A != None:
        M = C - jnp.einsum('abc, abd -> acd', jnp.einsum('abc,abd->acd', A_inv, B), B) # [R C C]
        M_inv, ln_det_M = invert_matrix(M) # [R]
        L = jnp.einsum('abc, abd->acd', A_inv, jnp.einsum('abc, adc -> abd', B, M_inv)) # [R A C]
        K = A_inv + jnp.einsum('abc, abd -> acd', jnp.einsum('abc, adc -> abd', L, B), A_inv)
        mat_inv = jnp.block(
            [[K, - L], [M, -jnp.swapaxes(L, axis1=1, axis2=2)]]
        )
        ln_det_mat = ln_det_M + ln_det_A
    elif C_inv != None and ln_det_C != None:
        M = A - jnp.einsum('abc, adb -> acd', jnp.einsum('abc,adb->acd', C_inv, B), B) # [R A A]
        M_inv, ln_det_M = invert_matrix(M)
        L = jnp.einsum('abc, adb->acd', C_inv, jnp.einsum('acb, adc -> adb', B, M_inv)) # [R A C]
        K = C_inv + jnp.einsum('abc, abd -> acd', jnp.einsum('abc, abd -> acd', L, B), A_inv)
        mat_inv = jnp.block(
            [[M, - L], [K, -jnp.swapaxes(L, axis1=1, axis2=2)]]
        )
        ln_det_mat = ln_det_M + ln_det_C
    else:
        raise AttributeError('Inverse and log determinant of A or C must be provided.')
    return mat_inv, ln_det_mat
    
