from gaussian_toolbox import conditional, factor
from jax import numpy as jnp
import objax


class GaussianLikelihood(objax.Module):
    def __init__(
        self,
        sigma_y: float = 1.0,
    ):
        self.log_sigma_y = objax.TrainVar(jnp.array(jnp.log(sigma_y)))

    @property
    def sigma_y(self):
        return jnp.exp(self.log_sigma_y)

    @property
    def lambda_y(self):
        return jnp.exp(-self.log_sigma_y)

    def get_conditional(self, N: int):
        Sigma_y = self.sigma_y * jnp.eye(N)[None]
        Lambda_y = 1.0 / self.sigma_y * jnp.eye(N)[None]
        ln_det_Sigma_y = jnp.log(self.sigma_y)[None] * N
        lk_conditional = conditional.ConditionalIdentityDiagGaussianPDF(
            Sigma=Sigma_y, Lambda=Lambda_y, ln_det_Sigma=ln_det_Sigma_y
        )
        return lk_conditional

    def get_likelihood_factor(self, y: jnp.ndarray):
        N = y.shape[0]
        lk = self.get_conditional(N)
        lk_y = lk.set_y(y)
        return lk_y

    def get_sparse_likelihood_factor(
        self,
        y: jnp.ndarray,
        cond_prior: conditional.ConditionalGaussianPDF,
    ):
        N = y.shape[0]
        Lambda = (
            jnp.sum(
                jnp.einsum("abc, abd -> acd", cond_prior.M, cond_prior.M),
                axis=0,
                keepdims=True,
            )
            * self.lambda_y
        )
        ymb = y[:, None] - cond_prior.b
        nu = jnp.sum(ymb[:, :, None] * cond_prior.M * self.lambda_y, axis=0)
        ln_beta = (
            -0.5
            * jnp.sum((ymb) ** 2 + cond_prior.Sigma[:, :, 0], axis=0)
            * self.lambda_y
        )
        ln_beta -= 0.5 * N * (jnp.log(2 * jnp.pi) + self.log_sigma_y)
        sparse_lk_factor = factor.ConjugateFactor(Lambda=Lambda, nu=nu, ln_beta=ln_beta)
        return sparse_lk_factor
