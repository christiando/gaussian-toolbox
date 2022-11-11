from gaussian_toolbox import conditional, factor
from jax import numpy as jnp
import objax


class GaussianLikelihood(objax.Module):
    r"""Gaussian likelihood of the form
    
    .. math
        
        p(y\vert g) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_y^2}}\exp(-\frac{(y_i-f_i)^2}{2\sigma_y^2}) 

    :param sigma_y: Standard deviation of observation noise, defaults to 1.0
    :type sigma_y: float, optional
    """

    def __init__(
        self,
        sigma_y: float = 1.0,
    ):
        self.log_sigma_y = objax.TrainVar(jnp.array(jnp.log(sigma_y)))

    @property
    def sigma_y(self) -> float:
        """Standard deviation

        :return: Standard deviation
        :rtype: float
        """
        return jnp.exp(self.log_sigma_y.value)

    @property
    def lambda_y(self) -> float:
        """Precision.

        :return: Precision
        :rtype: float
        """
        return jnp.exp(-2 * self.log_sigma_y.value)
    
    @property
    def sigma2_y(self) -> float:
        """Variance.

        :return: Variance
        :rtype: float
        """
        return jnp.exp(2 * self.log_sigma_y.value)

    def get_conditional(self, N: int) -> conditional.ConditionalIdentityDiagGaussianPDF:
        """Get likelihood factors.

        :param N: Number of data points
        :type N: int
        :return: Likelihood as Gaussian conditional object.
        :rtype: conditional.ConditionalIdentityDiagGaussianPDF
        """
        Sigma_y = self.sigma2_y * jnp.eye(N)[None]
        Lambda_y = 1.0 / self.sigma2_y * jnp.eye(N)[None]
        ln_det_Sigma_y = 2 * self.log_sigma_y.value[None] * N
        lk_conditional = conditional.ConditionalIdentityDiagGaussianPDF(
            Sigma=Sigma_y, Lambda=Lambda_y, ln_det_Sigma=ln_det_Sigma_y
        )
        return lk_conditional

    def get_likelihood_factor(self, y: jnp.ndarray) -> factor.ConjugateFactor:
        """Likelihood factor with y set to observations.

        :param y: Observations. [N, 1]
        :type y: jnp.ndarray
        :return: Likelihood factors.
        :rtype: factor.ConjugateFactor
        """
        N = y.shape[0]
        lk = self.get_conditional(N)
        lk_y = lk.set_y(y)
        return lk_y

    def get_sparse_likelihood_factor(
        self,
        y: jnp.ndarray,
        cond_prior: conditional.ConditionalGaussianPDF,
    ) -> factor.ConjugateFactor:
        r"""Calculates the sparse likelihood.
        
        .. math
        
            p(y\vert g_u) = \exp\left(-\int p(g\vert g_u)\ln p(y\vert g){\rm d}g\right),
        
        where :math:`g_u` are the function values at the inducing points.

        :param y: Observations. [N, 1]
        :type y: jnp.ndarray
        :param cond_prior: Conditional prior :math:`p(g\vert g_u)`.
        :type cond_prior: conditional.ConditionalGaussianPDF
        :return: Sparse likelihood factor.
        :rtype: factor.ConjugateFactor
        """
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
        ln_beta -= 0.5 * N * (jnp.log(2 * jnp.pi) + 2 * self.log_sigma_y.value)
        sparse_lk_factor = factor.ConjugateFactor(Lambda=Lambda, nu=nu, ln_beta=ln_beta)
        return sparse_lk_factor
