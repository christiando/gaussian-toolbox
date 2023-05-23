from gaussian_toolbox import pdf
from gaussian_toolbox.utils import linalg
import pytest
from jax import numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal
import jax
from jax import config
config.update("jax_enable_x64", True)
np.random.seed(0)


class TestGaussianPDF:
    def setup_class(self):
        self.test_class = pdf.GaussianPDF

    @staticmethod
    def get_pd_matrix(R, D, eigen_mu=1):
        A = jnp.array(np.random.rand(R, D, D))
        psd_mat = jnp.einsum("abc,abd->acd", A, A)
        psd_mat += jnp.eye(D)[None]
        return psd_mat

    def create_instance(self, R, D):
        Sigma = self.get_pd_matrix(R, D)
        mu = jnp.array(np.random.randn(R, D))
        return pdf.GaussianPDF(Sigma=Sigma, mu=mu)

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_init(self, R, D):
        d = self.create_instance(R, D)
        assert d.Sigma.shape == (d.R, d.D, d.D)
        assert d.nu.shape == (d.R, d.D)
        assert d.ln_beta.shape == (d.R,)
        assert jnp.alltrue(d.is_normalized())
        assert jnp.allclose(d.integrate(), 1)

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_sample(self, R, D):
        np.random.seed(0)
        d = self.create_instance(R, D)
        num_samples = 1000000
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, num_samples)
        mu_sample = jnp.mean(samples, axis=0)
        std_sample = jnp.std(samples, axis=0)
        std = jnp.sqrt(jnp.diagonal(d.Sigma, axis1=1, axis2=2))
        assert jnp.allclose(mu_sample, d.mu, atol=1e-2)
        assert jnp.allclose(std_sample, std, atol=1e-2)

    @pytest.mark.parametrize(
        "R, D, idx",
        [
            (100, 5, jnp.array([0, 1, 50])),
            (1, 5, jnp.array([0])),
            (100, 1, jnp.array([0, 20, 30])),
            (100, 5, jnp.array([0, 20, 70])),
        ],
    )
    def test_slice(self, R, D, idx):
        d = self.create_instance(R, D)
        d_new = d.slice(idx)
        assert jnp.allclose(d_new.Lambda, d.Lambda[idx])
        assert jnp.allclose(d_new.nu, d.nu[idx])
        assert jnp.allclose(d_new.Sigma, d.Sigma[idx])
        assert jnp.allclose(d_new.mu, d.mu[idx])
        assert jnp.allclose(d_new.ln_beta, d.ln_beta[idx])
        assert jnp.allclose(d_new.ln_det_Sigma, d.ln_det_Sigma[idx])

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_update(self, R, D):
        d = self.create_instance(R, D)
        d_update = self.create_instance(1, D)
        idx = jnp.array(
            [
                0,
            ]
        )
        d.update(idx, d_update)
        assert jnp.allclose(d_update.Lambda, d.Lambda[idx])
        assert jnp.allclose(d_update.nu, d.nu[idx])
        assert jnp.allclose(d_update.Sigma, d.Sigma[idx])
        assert jnp.allclose(d_update.mu, d.mu[idx])
        assert jnp.allclose(d_update.ln_beta, d.ln_beta[idx])
        assert jnp.allclose(d_update.ln_det_Sigma, d.ln_det_Sigma[idx])

    @pytest.mark.parametrize(
        "R, D, dim_x",
        [
            (2, 5, jnp.array([0, 1, 2])),
            (1, 5, jnp.array([0, 4])),
            (2, 1, jnp.array([0])),
            (2, 5, jnp.array([0, 3, 1])),
        ],
    )
    def test_get_marginal(self, R, D, dim_x):
        d = self.create_instance(R, D)
        md = d.get_marginal(dim_x)
        assert jnp.allclose(md.mu, d.mu[:, dim_x])
        idx = jnp.ix_(jnp.arange(R), dim_x, dim_x)
        assert jnp.allclose(md.Sigma, d.Sigma[idx])
        assert jnp.alltrue(md.is_normalized())
        assert jnp.alltrue(md.integrate() == 1)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (2, 5, jnp.array([0, 1, 2])),
            (1, 5, jnp.array([0, 4])),
            (2, 5, jnp.array([0, 3, 1])),
        ],
    )
    def test_condition_on(self, R, D, dim_y):
        dim_xy = jnp.arange(D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        idx_x = jnp.ix_(jnp.arange(R), dim_x, dim_x)
        d = self.create_instance(R, D)
        Lambda_x = d.Lambda[idx_x]
        Sigma_x, ln_det_Lambda_x = linalg.invert_matrix(Lambda_x)
        M_x = -jnp.einsum("abc,acd->abd", Sigma_x, d.Lambda[:, dim_x][:, :, dim_y])
        b_x = d.mu[:, dim_x] - jnp.einsum("abc,ac->ab", M_x, d.mu[:, dim_y])
        cd = d.condition_on(dim_y)
        assert jnp.allclose(cd.Lambda, d.Lambda[idx_x])
        assert jnp.allclose(cd.M, M_x)
        assert jnp.allclose(cd.b, b_x)
        assert jnp.allclose(cd.ln_det_Sigma, -ln_det_Lambda_x)

        cd2 = d.condition_on_explicit(dim_y, dim_x)
        assert jnp.allclose(cd2.Lambda, d.Lambda[idx_x])
        assert jnp.allclose(cd2.M, M_x)
        assert jnp.allclose(cd2.b, b_x)
        assert jnp.allclose(cd2.ln_det_Sigma, -ln_det_Lambda_x)

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_to_dict(self, R, D):
        d = self.create_instance(R, D)
        d_dict = d.to_dict()
        assert jnp.allclose(d_dict["Sigma"], d.Sigma)
        assert jnp.allclose(d_dict["mu"], d.mu)
        assert jnp.allclose(d_dict["Lambda"], d.Lambda)
        Lambda, ln_det_Sigma = linalg.invert_matrix(d.Sigma)
        assert jnp.allclose(
            d_dict["ln_det_Sigma"],
            ln_det_Sigma,
            atol=1e-6,
        )

    @pytest.mark.parametrize("R, D", [(1, 5), (1, 5), (1, 1)])
    def test_entropy(self, R, D):
        d = self.create_instance(R, D)
        entropy_sc = multivariate_normal(mean=d.mu[0], cov=d.Sigma[0]).entropy()
        assert np.allclose(entropy_sc, d.entropy()[0])

    @pytest.mark.parametrize("R, D", [(1, 5), (1, 5), (1, 1)])
    def test_kl_divergence(self, R, D):
        d = self.create_instance(R, D)
        assert np.allclose(d.kl_divergence(d), 0, atol=1e-6)
        d2 = self.create_instance(R, D)

        assert np.alltrue(d.kl_divergence(d2) >= 0)
        
    @pytest.mark.parametrize("R, D, Dsum", [(1, 5, 2), (1, 5, 1), (1, 3, 2), (1, 3, 5)])
    def test_get_density_of_linear_sum(self, R, D, Dsum):
        d = self.create_instance(R, D)
        W = jnp.array(np.random.randn(R, Dsum, D))
        b = jnp.array(np.random.randn(R, Dsum))
        if Dsum <= D:
            d_sum = d.get_density_of_linear_sum(W, b)
            key = jax.random.PRNGKey(0)
            subkey, key = jax.random.split(key)
            d_sample = d.sample(subkey, num_samples=100000)[:,0]
            W_d_sample  = jnp.einsum("abc,ac->ab", W, d_sample)
            sum_sampled_mean = jnp.mean(W_d_sample + b, axis=0, keepdims=True)
            var_sampled_mean = jnp.mean(jnp.einsum("ab,ac->abc", W_d_sample + b, W_d_sample + b), axis=0, keepdims=True)
            var_sampled_mean -= jnp.einsum("ab,ac->abc", sum_sampled_mean, sum_sampled_mean)
            assert jnp.allclose(d_sum.mu, sum_sampled_mean, rtol=1e-1, atol=1e-2)
            assert jnp.allclose(d_sum.Sigma, var_sampled_mean, rtol=1e-1, atol=1e-2)
            # When b is not sepcified
            d_sum = d.get_density_of_linear_sum(W)
            W_d_sample  = jnp.einsum("abc,ac->ab", W, d_sample)
            sum_sampled_mean = jnp.mean(W_d_sample, axis=0, keepdims=True)
            var_sampled_mean = jnp.mean(jnp.einsum("ab,ac->abc", W_d_sample, W_d_sample), axis=0, keepdims=True)
            var_sampled_mean -= jnp.einsum("ab,ac->abc", sum_sampled_mean, sum_sampled_mean)
            assert jnp.allclose(d_sum.mu, sum_sampled_mean, rtol=1e-1, atol=1e-2)
            assert jnp.allclose(d_sum.Sigma, var_sampled_mean, rtol=1e-1, atol=1e-2)
        else:
            with pytest.raises(AssertionError):
                d_sum = d.get_density_of_linear_sum(W, b)

class TestGaussianDiagPDF(TestGaussianPDF):
    def setup_class(self):
        self.test_class = pdf.GaussianDiagPDF

    def create_instance(self, R, D):
        Sigma = jnp.tile(
            (jnp.array(np.random.rand(D, D)) * jnp.eye(D))[None], [R, 1, 1]
        )
        mu = jnp.array(np.random.randn(R, D))
        return pdf.GaussianDiagPDF(Sigma=Sigma, mu=mu)
