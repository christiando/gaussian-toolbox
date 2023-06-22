__all__ = ["invert_matrix", "invert_diagonal"]
from jax import scipy as jsc
from jax import numpy as jnp
from typing import Tuple
from jaxtyping import Array, Float


def invert_matrix(
    A: Float[Array, "R D D"]
) -> Tuple[Float[Array, "R D D"], Float[Array, "R"]]:
    L = jsc.linalg.cho_factor(A)
    A_inv = jsc.linalg.cho_solve(L, jnp.tile(jnp.eye(A.shape[1])[None], (len(A), 1, 1)))
    ln_det_A = 2.0 * jnp.sum(jnp.log(L[0].diagonal(axis1=-1, axis2=-2)), axis=1)
    return A_inv, ln_det_A


def invert_diagonal(
    A: Float[Array, "R D D"]
) -> Tuple[Float[Array, "R D D"], Float[Array, "R"]]:
    A_inv = jnp.concatenate(
        [jnp.diag(mat)[None] for mat in 1.0 / A.diagonal(axis1=1, axis2=2)], axis=0
    )
    ln_det_A = jnp.sum(jnp.log(A.diagonal(axis1=1, axis2=2)), axis=1)
    return A_inv, ln_det_A


def invert_woodbury_diag(
    A_diagonal: Float[Array, "R D"],
    B_inv: Float[Array, "R L L"],
    M: Float[Array, "R D L"],
    ln_det_B: Float[Array, "R"],
) -> Tuple[Float[Array, "R D D"], Float[Array, "R"]]:
    """
    Invert a matrix of the form A + M B_inv M^T$ where A is diagonal.
    """
    assert A_diagonal.shape[0] == M.shape[0] == B_inv.shape[0]
    A_inv_diagonal = 1.0 / A_diagonal
    U = A_inv_diagonal[:, :, None] * M
    C = B_inv + jnp.einsum("abc,abd->acd", U, M)
    C_inv, C_ln_det = invert_matrix(C)
    Lambda = -jnp.einsum("abc, adb, aec -> ade", C_inv, U, U)
    i, j = jnp.diag_indices(min(Lambda.shape[-2:]))
    Lambda = Lambda.at[..., i, j].add(A_inv_diagonal)
    ln_det_Sigma = jnp.sum(jnp.log(A_diagonal), axis=-1) + C_ln_det + ln_det_B
    return Lambda, ln_det_Sigma


def invert_woodbury(
    A_inv: Float[Array, "R D D"],
    B_inv: Float[Array, "R L L"],
    M: Float[Array, "R D L"],
    ln_det_A: Float[Array, "R"],
    ln_det_B: Float[Array, "R"],
) -> Tuple[Float[Array, "R D D"], Float[Array, "R"]]:
    """
    Invert a matrix of the form A + M B M^T
    """
    U = jnp.einsum("abc,abd->acd", A_inv, M)
    C = B_inv + jnp.einsum("abc,abd->acd", U, M)
    C_inv, C_ln_det = invert_matrix(C)
    mat_inv = A_inv - jnp.einsum("abc, adb, aec -> ade", C_inv, U, U)
    ln_det_mat = ln_det_A + C_ln_det + ln_det_B
    return mat_inv, ln_det_mat


def invert_block_matrix(
    A_inv: Float[Array, "R D D"],
    B: Float[Array, "R L L"],
    C: Float[Array, "R D L"],
    ln_det_A: Float[Array, "R"],
) -> Tuple[Float[Array, "R D+L D+L"], Float[Array, "R"]]:
    """
    Invert a block matrix of the form

    [[A  C]              [[P  S],
     [C' B]], given by    [S' Q]].
    """
    assert A_inv.shape[0] == B.shape[0] == C.shape[0]
    assert B.shape[-1] == C.shape[-1]
    AC = jnp.einsum("abc, abd -> acd", A_inv, C)
    D_CAC = B - jnp.einsum("abc, abd -> acd", AC, C)
    Q, ln_det_D_CAC = invert_matrix(D_CAC)
    S = -jnp.einsum("abc, acd -> abd", AC, Q)
    P = A_inv + jnp.einsum("abc, adc -> abd", -S, AC)
    mat_inv = jnp.block([[P, S], [S.transpose((0, 2, 1)), Q]])
    ln_det_mat = ln_det_A + ln_det_D_CAC
    return mat_inv, ln_det_mat
