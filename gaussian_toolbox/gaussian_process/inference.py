from jax import numpy as jnp
from sklearn.covariance import log_likelihood
import gp
import likelihood
from gaussian_toolbox import pdf
import objax
from gaussian_toolbox.utils.jax_minimize_wrapper import ScipyMinimize


class ExactInference(objax.Module):
    def __init__(self, prior: gp.GaussianProcess, lk: likelihood.GaussianLikelihood):
        self.prior = prior
        self.likelihood = lk
        if self.prior.N == 0:
            self.prior.update_prior(self.likelihood.X)

    def get_posterior(self, prior_density: pdf.GaussianPDF = None) -> pdf.GaussianPDF:
        if prior_density is None:
            prior_density = self.prior.prior_density
        unormalized_posterior = self.likelihood.likelihood_factor * prior_density
        posterior = unormalized_posterior.get_density()
        return posterior

    def get_log_likelihood(self) -> float:
        unormalized_posterior = (
            self.likelihood.likelihood_factor * self.prior.prior_density
        )
        log_likelihood = unormalized_posterior.log_integral_light()
        return log_likelihood

    def optimize_hyperparameters(self):
        @objax.Function.with_vars(self.vars())
        def loss():
            self.prior.update_prior(self.likelihood.X)
            return -self.get_log_likelihood()

        minimizer = ScipyMinimize(loss, self.vars(), method="L-BFGS-B")
        minimizer.minimize()


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
