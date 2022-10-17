from gaussian_toolbox import pdf, conditional, factor
from gaussian_process import gp
from typing import Union
from jax import numpy as jnp
from gaussian_toolbox.utils.linalg import invert_matrix, invert_diagonal
import objax


class GaussianLikelihood(objax.Module):
    def __init__(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray,
        Sigma_y: jnp.ndarray,
    ):
        self.N = y.shape[0]
        self.X = X
        Lambda_y, ln_det_Sigma = invert_matrix(Sigma_y)
        nu = jnp.einsum("abc, ac -> ab", Lambda_y, y)
        y_Lambda_y = jnp.einsum("ab, ab -> a", nu, y)
        ln_beta = -0.5 * (self.N * jnp.log(2 * jnp.pi) + ln_det_Sigma + y_Lambda_y)
        self.likelihood_factor = factor.ConjugateFactor(
            Lambda=Lambda_y, nu=nu, ln_beta=ln_beta
        )
        self.sparse = False

    def evaluate(self, f: jnp.ndarray):
        return self.likelihood_factor(f)

    def evaluate_ln(self, f: jnp.ndarray):
        return self.likelihood_factor.evaluate_ln(f)

    def sparsify(self, prior: gp.SparseGaussianProcess):
        conditional_prior = prior.get_conditional_prior(self.X)
        K = prior.kernel.evaluate(self.X)
        Lambda_M = jnp.einsum(
            "abc, abd -> acd", self.likelihood_factor.Lambda, conditional_prior.M
        )
        Lambda_new = jnp.einsum("abc, abd -> abd", conditional_prior.M, Lambda_M)
        nu_new = jnp.einsum(
            "ab, abc -> ac", self.likelihood_factor.nu, conditional_prior.M
        ) - jnp.einsum("ab,abc -> ac", conditional_prior.b, Lambda_M)
        b_Lambda_b = jnp.einsum(
            "ab,ab->a",
            jnp.einsum(
                "abc,ab->ac", self.likelihood_factor.Lambda, conditional_prior.b
            ),
            conditional_prior.b,
        )
        ln_beta_new = (
            self.likelihood_factor.ln_beta
            - 0.5
            * jnp.trace(
                jnp.einsum("abc,abd->acd", self.likelihood_factor.Lambda, K),
                axis1=1,
                axis2=2,
            )
            + jnp.einsum("ab,ab->a", self.likelihood_factor.nu, conditional_prior.b)
            - 0.5 * b_Lambda_b
        )
        self.sparse_likelihood_factor = factor.ConjugateFactor(
            Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new
        )
        self.sparse = True


class GaussianDiagLikelihood(GaussianLikelihood):
    def __init__(
        self,
        y: jnp.ndarray,
        Sigma_y: jnp.ndarray,
    ):
        self.N = y.shape[0]
        Lambda_y, ln_det_Sigma = invert_diagonal(Sigma_y)
        nu = jnp.einsum("abc, ac -> ab", Lambda_y, y)
        y_Lambda_y = jnp.einsum("ab, ab -> a", nu, y)
        ln_beta = -0.5 * (self.N * jnp.log(2 * jnp.pi) + ln_det_Sigma + y_Lambda_y)
        self.likelihood_factor = factor.ConjugateFactor(
            Lambda=Lambda_y, nu=nu, ln_beta=ln_beta
        )
        self.sparse = False

    def sparsify(self, prior: gp.SparseGaussianProcess):
        conditional_prior = prior.get_conditional_prior(self.X)
        K_diag = prior.kernel.eval_diag(self.X)
        Lambda_diag = jnp.diagonal(self.likelihood_factor.Lambda, axis1=1, axis2=2)
        Lambda_M = jnp.einsum("ab, abd -> abd", Lambda_diag, conditional_prior.M)
        Lambda_new = jnp.einsum("abc, abd -> abd", conditional_prior.M, Lambda_M)
        nu_new = jnp.einsum(
            "ab, abc -> ac", self.likelihood_factor.nu, conditional_prior.M
        ) - jnp.einsum("ab,abc -> ac", conditional_prior.b, Lambda_M)
        b_Lambda_b = jnp.einsum(
            "ab,ab->a",
            Lambda_diag * conditional_prior.b,
            conditional_prior.b,
        )
        ln_beta_new = (
            self.likelihood_factor.ln_beta
            - 0.5 * jnp.sum(*K_diag, axis=1)
            + jnp.einsum("ab,ab->a", self.likelihood_factor.nu, conditional_prior.b)
            - 0.5 * b_Lambda_b
        )
        self.sparse_likelihood_factor = factor.ConjugateFactor(
            Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new
        )
        self.sparse = True
