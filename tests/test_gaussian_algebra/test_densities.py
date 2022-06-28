from gaussian_toolbox.gaussian_algebra import densities
from gaussian_toolbox.utils import linalg
import pytest
from jax import numpy as jnp
from jax import scipy as jsc
import numpy as np
import objax
from scipy.stats import multivariate_normal


class TestGaussianDensity:
    def setup_class(self):
        self.test_class = densities.GaussianDensity

    @staticmethod
    def get_pd_matrix(R, D, eigen_mu=1):
        # Q = objax.random.normal((R, D, D))
        # eig_vals = jnp.abs(eigen_mu + objax.random.normal((R, D)))
        # psd_mat = jnp.einsum("abc,abd->acd", Q * eig_vals[:, :, None], Q)
        # psd_mat = 0.5 * (psd_mat + jnp.swapaxes(psd_mat, -1, -2))
        A = objax.random.uniform((R, D, D))
        psd_mat = jnp.einsum("abc,abd->acd", A, A)
        psd_mat += jnp.eye(D)[None]
        return psd_mat

    def create_instance(self, R, D):
        Sigma = self.get_pd_matrix(R, D)
        mu = objax.random.normal((R, D))
        return densities.GaussianDensity(Sigma, mu)

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_init(self, R, D):
        d = self.create_instance(R, D)
        assert d.Sigma.shape == (d.R, d.D, d.D)
        assert jnp.alltrue(d.ln_det_Lambda == -d.ln_det_Sigma)
        assert d.nu.shape == (d.R, d.D)
        assert d.ln_beta.shape == (d.R,)
        assert jnp.alltrue(d.is_normalized())
        assert jnp.alltrue(d.integrate() == 1)

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_sample(self, R, D):
        np.random.seed(0)
        d = self.create_instance(R, D)
        num_samples = 1000000
        samples = d.sample(num_samples)
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
        assert jnp.alltrue(d_new.Lambda == d.Lambda[idx])
        assert jnp.alltrue(d_new.nu == d.nu[idx])
        assert jnp.alltrue(d_new.Sigma == d.Sigma[idx])
        assert jnp.alltrue(d_new.mu == d.mu[idx])
        assert jnp.alltrue(d_new.ln_beta == d.ln_beta[idx])
        assert jnp.alltrue(d_new.ln_det_Sigma == d.ln_det_Sigma[idx])
        assert jnp.alltrue(d_new.ln_det_Lambda == d.ln_det_Lambda[idx])

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_update(self, R, D):
        d = self.create_instance(R, D)
        d_update = self.create_instance(1, D)
        idx = jnp.array([0,])
        d.update(idx, d_update)
        assert jnp.alltrue(d_update.Lambda == d.Lambda[idx])
        assert jnp.alltrue(d_update.nu == d.nu[idx])
        assert jnp.alltrue(d_update.Sigma == d.Sigma[idx])
        assert jnp.alltrue(d_update.mu == d.mu[idx])
        assert jnp.alltrue(d_update.ln_beta == d.ln_beta[idx])
        assert jnp.alltrue(d_update.ln_det_Sigma == d.ln_det_Sigma[idx])
        assert jnp.alltrue(d_update.ln_det_Lambda == d.ln_det_Lambda[idx])

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
        assert jnp.alltrue(md.mu == d.mu[:, dim_x])
        idx = jnp.ix_(jnp.arange(R), dim_x, dim_x)
        assert jnp.alltrue(md.Sigma == d.Sigma[idx])
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
        assert jnp.allclose(cd.ln_det_Lambda, ln_det_Lambda_x)

        cd2 = d.condition_on_explicit(dim_y, dim_x)
        assert jnp.allclose(cd2.Lambda, d.Lambda[idx_x])
        assert jnp.allclose(cd2.M, M_x)
        assert jnp.allclose(cd2.b, b_x)
        assert jnp.allclose(cd2.ln_det_Lambda, ln_det_Lambda_x)

    @pytest.mark.parametrize("R, D", [(2, 5), (1, 5), (2, 1)])
    def test_to_dict(self, R, D):
        d = self.create_instance(R, D)
        d_dict = d.to_dict()
        assert jnp.allclose(d_dict["Sigma"], d.Sigma)
        assert jnp.allclose(d_dict["mu"], d.mu)
        assert jnp.allclose(d_dict["Lambda"], d.Lambda)
        Lambda, ln_det_Sigma = linalg.invert_matrix(d.Sigma)
        assert jnp.allclose(d_dict["ln_det_Sigma"], ln_det_Sigma, atol=1e-6,)

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


class TestGaussianDiagDensity(TestGaussianDensity):
    def setup_class(self):
        self.test_class = densities.GaussianDiagDensity

    def create_instance(self, R, D):
        Sigma = jnp.tile((objax.random.uniform((D, D)) * jnp.eye(D))[None], [R, 1, 1])
        mu = objax.random.normal((R, D))
        return densities.GaussianDiagDensity(Sigma, mu)