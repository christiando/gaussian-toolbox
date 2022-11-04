from gaussian_toolbox import conditional, factor
from gaussian_toolbox.gaussian_process import prior
from jax import numpy as jnp
from gaussian_toolbox.utils.linalg import invert_matrix, invert_diagonal
import objax


class GaussianLikelihood(objax.Module):
    def __init__(
        self,
        sigma_y: float = 1.0,
    ):
        """Defines the Gaussian Likelihood

        .. math::

            p(y|f) = {\cal N}(f,\Sigma_y),

        where :math:`\Sigma_y` can be a a full covariance matrix.

        :param y: Output observations.
        :type y: jnp.ndarray
        :param X: Input observations.
        :type X: jnp.ndarray
        :param Sigma_y: Covariance matrix.
        :type Sigma_y: jnp.ndarray
        """

        self.log_sigma_y = objax.TrainVar(jnp.array(jnp.log(sigma_y)))

    @property
    def sigma_y(self):
        return jnp.exp(self.log_sigma_y)

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
        X: jnp.ndarray,
        sgp_prior: prior.SparseGP_Prior,
    ):
        sparse_conditional_prior = self.prior.get_conditional_prior(
            self.X, self.prior_density, ignore_covariances=True
        )
        lk_factor = self.get_likelihood_factor(y)
        Lambda_M = (
            lk_factor.Lambda.diagonal(axis1=1, axis2=2)[:, :, None]
            * sparse_conditional_prior.M
        )

        Lambda_new = jnp.einsum("abc, abd -> abd", sparse_conditional_prior.M, Lambda_M)
        nu_new = jnp.einsum(
            "ab, abc -> ac", lk_factor.nu, sparse_conditional_prior.M
        ) - jnp.einsum("ab,abc -> ac", sparse_conditional_prior.b, Lambda_M)
        b_Lambda_b = jnp.einsum(
            "ab,ab->a",
            lk_factor.Lambda.diagonal(axis1=1, axis2=2) * sparse_conditional_prior.b,
            sparse_conditional_prior.b,
        )
        ln_beta_new = (
            lk_factor.ln_beta
            - 0.5
            * jnp.trace(
                jnp.einsum("abc,abd->acd", lk_factor.Lambda, K),
                axis1=1,
                axis2=2,
            )
            + jnp.einsum("ab,ab->a", lk_factor.nu, sparse_conditional_prior.b)
            - 0.5 * b_Lambda_b
        )
        sparse_likelihood_factor = factor.ConjugateFactor(
            Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new
        )
        return sparse_likelihood_factor


'''
class GaussianDiagLikelihood(GaussianLikelihood):
    def __init__(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray,
        Sigma_y: jnp.ndarray,
    ):
        """Defines the Gaussian Likelihood

        .. math::

            p(y|f) = {\cal N}(f,\Sigma_y),

        where :math:`\Sigma_y` can is a diagonal covariance matrix.

        :param y: Output observations.
        :type y: jnp.ndarray
        :param X: Input observations.
        :type X: jnp.ndarray
        :param Sigma_y: Covariance matrix.
        :type Sigma_y: jnp.ndarray
        """
        self.N = y.shape[0]
        self.X = X
        self.y = y[None]
        Lambda_y, ln_det_Sigma = invert_diagonal(Sigma_y)
        nu = jnp.einsum("abc, ac -> ab", Lambda_y, self.y)
        y_Lambda_y = jnp.einsum("ab, ab -> a", nu, self.y)
        ln_beta = -0.5 * (self.N * jnp.log(2 * jnp.pi) + ln_det_Sigma + y_Lambda_y)
        self.likelihood_factor = factor.ConjugateFactor(
            Lambda=Lambda_y, nu=nu, ln_beta=ln_beta
        )
        self.sparse = False

    def sparsify(self, prior: prior.SparseGaussianProcess):
        r"""Sparsify the likelihood, such that it becomes only dependent on the function values of the inducing points,
        meaning

        .. math::

            p(y|f_U) \approx \exp \ln \int p(y|f)p(f|f_u)df.

        :param prior: The sparse Gaussian process prior.
        :type prior: priorSparseGaussianProcess
        """
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
'''
