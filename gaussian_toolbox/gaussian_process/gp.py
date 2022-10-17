from gaussian_toolbox.gaussian_process import kernel
from gaussian_toolbox import pdf, conditional
from jax import numpy as jnp
import objax


class GaussianProcess(objax.Module):
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
        self.X = None
        self.N = 0
        self.prior_density = None

    def delete_data(self):
        self.X = None
        self.N = 0
        self.prior_density = None

    def predict(
        self, X_star: jnp.ndarray, posterior_X: pdf.GaussianPDF == None
    ) -> pdf.GaussianPDF:
        if self.N == 0:
            Sigma = self.kernel.evaluate(X_star, X_star)
            mu = self.mean(X_star)
            predictive_density = pdf.GaussianPDF(Sigma=Sigma, mu=mu)
        else:
            if posterior_X is None:
                raise ValueError("Please provide a posterior p(f|X).")
            conditional_prior = self.get_conditional_prior(X_star)
            predictive_density = conditional_prior.affine_marginal_transformation(
                posterior_X
            )
        return predictive_density

    def update_prior(self, X):
        if self.N == 0:
            Sigma = self.kernel.evaluate(X, X)
            mu = self.mean(X)
            self.prior_density = pdf.GaussianPDF(Sigma=Sigma, mu=mu)
        else:
            self.X = jnp.concatenate([self.X, X])
            cond_prior = self.get_conditional_prior(X)
            self.prior_density = cond_prior.affine_joint_transformation(
                self.prior_density
            )
        self.N = X.shape[0]

    def get_conditional_prior(self, X_star) -> conditional.ConditionalGaussianPDF:
        N_star = X_star.shape[0]
        mu_star = self.mean(X_star)
        mu_star = mu_star.reshape(1, N_star)
        K_star = self.kernel.evaluate(X_star)
        K_cross = self.kernel.evaluate(self.X, X_star)
        K_cross_T = jnp.transpose(K_cross)
        K_star = K_star.reshape(1, N_star, N_star)
        K_cross = K_cross.reshape(1, self.N, N_star)
        M = jnp.dot(K_cross_T, self.prior.Lambda[0])
        Sigma_f = K_star - jnp.dot(M, K_cross)
        Sigma_f = Sigma_f.reshape(1, N_star, N_star)
        M = M.reshape(1, self.N_star, self.N)
        b = mu_star - jnp.dot(M, self.mu.reshape(self.N))
        return conditional.ConditionalGaussianPDF(M, b, Sigma_f)


class SparseGaussianProcess(GaussianProcess):
    def __init__(
        self,
        kernel: kernel.Kernel,
        Xu: jnp.ndarray,
        mean: callable = lambda x: jnp.zeros(x.shape[0]),
    ):
        """
        Base class for Gaussian Processes.

        :param kernel: kernel function of the gaussian process
        :type kernel: kernels.Kernel

        :param mean: mean function of the gaussian process, defaults to 0
        :type mean: lambda function, optional
        """
        super().__init__(kernel=kernel, mean=mean)
        self.update_prior(Xu)
