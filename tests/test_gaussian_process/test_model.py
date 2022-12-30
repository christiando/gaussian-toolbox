import pytest
import numpy as np
from matplotlib import pyplot as plt
from gaussian_toolbox.gaussian_process import kernel, prior, likelihood, model
from jax import numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

np.random.seed(1)

class TestGPRegressionModel:
    
    @pytest.mark.parametrize(
        "num_points",
        [
         (100), (20), (50)    
        ],
    )
    def test_evaluate1D(self, num_points):
        sigma_y = .1
        x = jnp.array(np.sort(2 * np.pi * np.random.rand(num_points, 1), axis=0))
        y = jnp.array(np.cos(2 * x) + sigma_y * np.random.randn(num_points, 1)) + 1.
        y = y.flatten()
        
        k_func = kernel.Matern32()
        gp_prior = prior.GP_Prior(k_func)
        lk = likelihood.GaussianLikelihood()
        gp_model = model.GPRegressionModel(gp_prior, lk)
        gp_model.infer(x, y)
        
        prior_log_likelihood = gp_prior.get_density(x).evaluate_ln(y[None])
        posterior_log_likelihood = gp_model.predict_data(x, only_marginals=False).evaluate_ln(y[None])
        
        assert jnp.alltrue(jnp.greater(posterior_log_likelihood, prior_log_likelihood))
        
        
class TestSGPRegressionModel:
    
    @pytest.mark.parametrize(
        "num_points",
        [
         (100), (20), (50)    
        ],
    )
    def test_evaluate1D(self, num_points):
        sigma_y = .1
        x = jnp.array(np.sort(2 * np.pi * np.random.rand(num_points, 1), axis=0))
        y = jnp.array(np.cos(2 * x) + sigma_y * np.random.randn(num_points, 1)) + 1.
        y = y.flatten()
        
        num_inducing_points = 20
        k_func = kernel.Matern32()
        Xu = 2 * jnp.pi * jnp.array(np.random.rand(num_inducing_points,1))
        sgp_prior = prior.SparseGP_Prior(k_func, Xu, optimize_Xu=True)
        lk = likelihood.GaussianLikelihood()
        sgp_model = model.SGPRegressionModel(sgp_prior, lk)
        sgp_model.infer(x, y)
        
        prior_log_likelihood = sgp_prior.get_density(x).evaluate_ln(y[None])
        posterior_log_likelihood = sgp_model.predict_data(x, only_marginals=False).evaluate_ln(y[None])
        
        assert jnp.alltrue(jnp.greater(posterior_log_likelihood, prior_log_likelihood))
        