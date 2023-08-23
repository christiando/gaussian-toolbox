import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import random
from gaussian_toolbox.utils.linalg import (
    invert_matrix,
    invert_woodbury_diag,
    invert_woodbury_diag_diag,
    invert_woodbury,
    invert_block_matrix,
)
import pytest


@pytest.mark.parametrize(
    "R, D, L, ",
    [(1, 5, 2), (10, 10, 3), (100, 2, 5), (100, 1, 1), (10, 1, 5), (2, 5, 1)],
)
def test_matrix_inversions(R, D, L):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    A = random.uniform(subkey, (R, D, D))
    A = D * jnp.eye(D)[None] + jnp.einsum("abc,abd->acd", A, A)
    A_inv, ln_det_A = invert_matrix(A)

    M = random.uniform(subkey, (R, D, L))
    B = random.uniform(subkey, (R, L, L))
    B = jnp.eye(L)[None] + jnp.einsum("abc,abd->acd", B, B)
    B_inv, ln_det_B = invert_matrix(B)

    mat = A + jnp.einsum("abc, adb, aec->ade", B, M, M)
    mat_inv, ln_det_mat = invert_matrix(mat)

    mat_inv2, ln_det_mat2 = invert_woodbury(A_inv, B_inv, M, ln_det_A, ln_det_B)
    assert jnp.allclose(mat_inv, mat_inv2)
    assert jnp.allclose(ln_det_mat, ln_det_mat2)

    A_diag = random.uniform(subkey, (R, D))
    A = jnp.eye(D)[None] * A_diag[:, :, None]
    A_inv, ln_det_A = invert_matrix(A)

    M = random.uniform(subkey, (R, D, L))
    B = random.uniform(subkey, (R, L, L))
    B = L * jnp.eye(L)[None] + jnp.einsum("abc,abd->acd", B, B)
    B_inv, ln_det_B = invert_matrix(B)
    mat = A + jnp.einsum("abc, adb, aec->ade", B, M, M)
    mat_inv, ln_det_mat = invert_matrix(mat)

    mat_inv2, ln_det_mat2 = invert_woodbury_diag(A_diag, B_inv, M, ln_det_B)
    assert jnp.allclose(mat_inv, mat_inv2)
    assert jnp.allclose(ln_det_mat, ln_det_mat2)

    A_diag = random.uniform(subkey, (R, D))
    A = jnp.eye(D)[None] * A_diag[:, :, None]
    A_inv, ln_det_A = invert_matrix(A)

    M = random.uniform(subkey, (R, D, L))
    B_diag = random.uniform(subkey, (R, L))
    B = jnp.eye(L)[None] * B_diag[:, :, None]
    B_inv, ln_det_B = invert_matrix(B)
    mat = A + jnp.einsum("abc, adb, aec->ade", B, M, M)
    mat_inv, ln_det_mat = invert_matrix(mat)

    mat_inv2, ln_det_mat2 = invert_woodbury_diag_diag(A_diag, B_diag, M)
    assert jnp.allclose(mat_inv, mat_inv2)
    assert jnp.allclose(ln_det_mat, ln_det_mat2)


@pytest.mark.parametrize(
    "R, D, L, ",
    [(1, 5, 2), (10, 10, 3), (100, 2, 5), (100, 1, 1), (10, 1, 5), (2, 5, 1)],
)
def test_block_matrix_inversions(R, D, L):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    A = random.uniform(subkey, (R, D, D))
    A = D * jnp.eye(D)[None] + jnp.einsum("abc,abd->acd", A, A)
    A_inv, ln_det_A = invert_matrix(A)
    M = random.uniform(subkey, (R, D, L))
    B = random.uniform(subkey, (R, L, L))
    B = L * jnp.eye(L)[None] + jnp.einsum("abc,abd->acd", B, B)
    B_inv, ln_det_B = invert_matrix(B)
    mat = jnp.block([[A, M], [M.transpose((0, 2, 1)), B]])
    mat_inv, ln_det_mat = invert_matrix(mat)
    mat_inv2, ln_det_mat2 = invert_block_matrix(A_inv, B, M, ln_det_A)
    assert jnp.allclose(mat_inv, mat_inv2)
    assert jnp.allclose(ln_det_mat, ln_det_mat2)
    mat_inv3, ln_det_mat3 = invert_block_matrix(B_inv, A, M, ln_det_B, False)
    assert jnp.allclose(mat_inv, mat_inv3)
    assert jnp.allclose(ln_det_mat, ln_det_mat3)
