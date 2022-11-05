from gaussian_toolbox.gaussian_process import kernel
from gaussian_toolbox import pdf, conditional
from jax import numpy as jnp
import objax


class GP_Prior(objax.Module):
    def __init__(
        self,
        kernel: kernel.Kernel,
        mean: callable = lambda x: jnp.zeros(x.shape[0]),
    ):
        """
        Base class for Gaussian Processes.

        :param kernel: kernel function of the gaussian process
        :type kernel: kernels.Kernel

        :param mean: mean function of the gaussian process, defaults to 0
        :type mean: lambda function, optional
        """
        self.kernel = kernel
        self.mean = mean

    def get_density(self, X: jnp.ndarray) -> pdf.GaussianPDF:
        """Compute the prediction density at the requested points. If not training data, prior is returned.

        .. math::

            p(f^\star|X^\star) = \int p(f^\star|X^\star,f)p(f|X) df.

        # TODO: Shouldn't be the prior part of the GP?

        :param X_star: Data points for which prediction is required.
        :type X_star: jnp.ndarray
        :param posterior_X: Posterior density over the points X, defaults to =None
        :type posterior_X: pdf.GaussianPDF, optional
        :raises ValueError: A posterior needs to be provided if there is training data.
        :return: Returns the predictive density over X_star.
        :rtype: pdf.GaussianPDF
        """
        Sigma = self.kernel.evaluate(X, X)
        mu = self.mean(X)
        prior_density = pdf.GaussianPDF(Sigma=Sigma[None], mu=mu[None])
        return prior_density

    def get_conditional_prior(
        self,
        X_star: jnp.ndarray,
        X: jnp.ndarray,
        prior_density: pdf.GaussianPDF = None,
        only_marginals: bool = False,
    ) -> conditional.ConditionalGaussianPDF:
        """Computes the conditional GP prior :math:`p(y^\star|X^\star,y)`.

        :param X_star: Points for which the conditional priot should be computed.
        :type X_star: jnp.ndarray
        :return: Conditional density for the GP.
        :rtype: conditional.ConditionalGaussianPDF
        """
        N, N_star = X.shape[0], X_star.shape[0]
        if prior_density is None:
            prior_density = self.get_density(X)

        mu_star = self.mean(X_star)
        K_cross = self.kernel.evaluate(X, X_star)
        K_cross_T = jnp.transpose(K_cross)
        M = jnp.dot(K_cross_T, prior_density.Lambda[0])

        if only_marginals:
            K_star_diag = self.kernel.eval_diag(X_star)
            Sigma_f_diag = K_star_diag - jnp.sum(M * K_cross_T, axis=1)
            Sigma_f = Sigma_f_diag.reshape(N_star, 1, 1)
            M = M.reshape(N_star, 1, N)
            b = mu_star[:, None] - jnp.dot(M, self.mean(X))
        else:
            K_star = self.kernel.evaluate(X_star)[None]
            K_star = K_star.reshape(1, N_star, N_star)
            Sigma_f = K_star - jnp.dot(M, K_cross)
            Sigma_f = Sigma_f.reshape(1, N_star, N_star)
            M = M.reshape(1, N_star, N)
            b = mu_star - jnp.dot(M, self.mean(X))

        return conditional.ConditionalGaussianPDF(M, b, Sigma_f)


class SparseGP_Prior(GP_Prior):
    def __init__(
        self,
        kernel: kernel.Kernel,
        Xu: jnp.ndarray,
        mean: callable = lambda x: jnp.zeros(x.shape[0]),
        optimize_Xu: bool = False,
    ):
        """Class for sparse GP according to [Titsias, 2009], i.e.

        .. math::

            p(f|X,X_u) = \int p(f|f_u, X)p(f_u|X_u) df_u.


        :param kernel: Kernel function that defines the GP.
        :type kernel: kernel.Kernel
        :param Xu: Inducing points.
        :type Xu: jnp.ndarray
        :param mean: Mean function, defaults to lambdax:jnp.zeros(x.shape[0])
        :type mean: callable, optional
        """
        super().__init__(kernel=kernel, mean=mean)
        self.optimize_Xu = optimize_Xu
        if self.optimize_Xu:
            self._Xu = objax.TrainVar(Xu)
        else:
            self._Xu = Xu

    @property
    def Xu(self):
        if self.optimize_Xu:
            return self._Xu.value
        else:
            return self._Xu
