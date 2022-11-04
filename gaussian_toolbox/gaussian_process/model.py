from jax import numpy as jnp
import jax
from sklearn.covariance import log_likelihood
from gaussian_toolbox.gaussian_process import prior, likelihood
from gaussian_toolbox import pdf, measure
import objax
from gaussian_toolbox.utils.jax_minimize_wrapper import ScipyMinimize


class GPRegressionModel(objax.Module):
    def __init__(self, prior: prior.GP_Prior, lk: likelihood.GaussianLikelihood):
        self.prior = prior
        self.likelihood = lk
        self.prior_density = None
        self.posterior_density = None
        self.objective = self.get_log_likelihood

    def infer(self, X: jnp.ndarray, y: jnp.ndarray, update_hyperparams: bool = True):
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        if update_hyperparams:
            self.optimize_hyperparameters()
        self.posterior_density = self.get_posterior()

    def predict_gp(
        self, X_star: jnp.ndarray, only_marginals: bool = False
    ) -> pdf.GaussianPDF:
        conditional_prior_density = self.prior.get_conditional_prior(
            X_star, self.X, self.prior_density, only_marginals=only_marginals
        )
        predictive_density = conditional_prior_density.affine_marginal_transformation(
            self.posterior_density
        )
        return predictive_density

    def predict_data(
        self, X_star: jnp.ndarray, only_marginals: bool = False
    ) -> pdf.GaussianPDF:
        predictive_gp_density = self.predict_gp(X_star, only_marginals=only_marginals)
        if only_marginals:
            lk_conditional = self.likelihood.get_conditional(1)
        else:
            lk_conditional = self.likelihood.get_conditional(X_star.shape[0])
        predictive_data_density = lk_conditional.affine_marginal_transformation(
            predictive_gp_density
        )
        return predictive_data_density

    def get_posterior(self) -> pdf.GaussianPDF:
        unormalized_posterior = self._get_unnormalized_posterior()
        posterior_density = unormalized_posterior.get_density()
        return posterior_density

    def _get_unnormalized_posterior(self) -> measure.GaussianMeasure:
        lk_factor = self.likelihood.get_likelihood_factor(self.y)
        unormalized_posterior = self.prior_density * lk_factor
        return unormalized_posterior

    def get_log_likelihood(self) -> float:
        unormalized_posterior = self._get_unnormalized_posterior()
        log_likelihood = unormalized_posterior.log_integral_light()
        return log_likelihood

    def optimize_hyperparameters(self):
        @objax.Function.with_vars(self.vars())
        def loss():
            self.prior_density = self.prior.get_density(self.X)
            return -self.objective().squeeze()

        minimizer = ScipyMinimize(loss, self.vars(), method="L-BFGS-B")
        minimizer.minimize()
        self.prior_density = self.prior.get_density(self.X)


class SGPRegressionModel(GPRegressionModel):
    def __init__(self, prior: prior.SparseGP_Prior, lk: likelihood.GaussianLikelihood):
        self.prior = prior
        self.likelihood = lk
        self.prior_density = None
        self.posterior_density = None
        self.objective = self.get_elbo

    def get_posterior(self) -> pdf.GaussianPDF:
        unormalized_posterior = self._get_unnormalized_posterior()
        posterior_density = unormalized_posterior.get_density()
        return posterior_density

    def _get_unnormalized_posterior(self) -> measure.GaussianMeasure:
        sparse_lk_factor = self.likelihood.get_sparse_likelihood_factor(
            self.y, self.X, self.prior
        )
        unormalized_posterior = self.prior_density * sparse_lk_factor
        return unormalized_posterior

    def get_elbo(self) -> float:
        unormalized_posterior = self._get_unnormalized_posterior()
        posterior_density = unormalized_posterior.get_density()
        elbo = posterior_density.integrate("log u(x)", factor=unormalized_posterior)
        elbo += posterior_density.entropy()
        return elbo


"""
class VariationalInference(ExactInference):
    def __init__(
        self, prior: gp.SparseGaussianProcess, lk: likelihood.GaussianLikelihood
    ):
        super().__init__(prior, lk)
        if not self.likelihood.sparse:
            self.likelihood.sparsify(self.prior)

    def get_log_likelihood(self) -> float:
        raise NotImplementedError("Exact likelihood non-tractable. Use ELBO instead.")

    def get_elbo(self, posterior: pdf.GaussianPDF) -> float:
        kl_div = posterior.kl_divergence(self.prior.prior_density)
        exp_log_likelihood = posterior.integrate(
            "log u(x)", factor=self.likelihood.sparse_likelihood_factor
        )
        elbo = exp_log_likelihood - kl_div
        return elbo

    def optimize_hyperparameters(self, posterior: pdf.GaussianPDF):
        @objax.Function.with_vars(self.vars())
        def loss():
            self.prior.update_prior(self.likelihood.X)
            self.likelihood.sparsify(self.prior)
            return -self.get_elbo(posterior)

        minimizer = ScipyMinimize(loss, self.vars(), method="L-BFGS-B")
        minimizer.minimize()
"""
