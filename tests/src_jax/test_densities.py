import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src_jax import densities
import pytest
from jax import numpy as jnp
from jax import scipy as jsc
import numpy as np


class TestGaussianDensity:
    def setup_class(self):
        self.test_class = densities.GaussianDensity

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        Sigma = jnp.zeros((R, D, D))
        Sigma = Sigma.at[:].set(jnp.eye(D))
        mu = jnp.zeros((R, D))
        d = self.test_class(Sigma, mu)
        assert d.Sigma.shape == (d.R, d.D, d.D)
        assert jnp.alltrue(d.ln_det_Lambda == -d.ln_det_Sigma)
        assert d.nu.shape == (d.R, d.D)
        assert d.ln_beta.shape == (d.R,)
        assert jnp.alltrue(d.is_normalized())
        assert jnp.alltrue(d.integrate() == 1)

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_sample(self, R, D):
        np.random.seed(0)
        Sigma = jnp.zeros((R, D, D))
        Sigma = Sigma.at[:].set(jnp.eye(D))
        mu = jnp.asarray(np.random.randn(R, D))
        d = self.test_class(Sigma, mu)
        num_samples = 100000
        samples = d.sample(num_samples)
        mu_sample = jnp.mean(samples, axis=0)
        std_sample = jnp.std(samples, axis=0)
        std = jnp.sqrt(jnp.diagonal(Sigma, axis1=1, axis2=2))
        assert jnp.allclose(mu_sample, mu, atol=1e-2)
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
        Sigma = jnp.zeros((R, D, D))
        Sigma = Sigma.at[:].set(jnp.eye(D))
        Sigma = Sigma.at[idx].set(
            np.random.rand(len(idx))[:, None, None] * jnp.eye(D)[None]
        )
        mu = jnp.zeros((R, D))
        mu = mu.at[idx].set(np.random.randn(len(idx), D))
        d = self.test_class(Sigma, mu)
        d_new = d.slice(idx)
        assert jnp.alltrue(d_new.Lambda == d.Lambda[idx])
        assert jnp.alltrue(d_new.nu == d.nu[idx])
        assert jnp.alltrue(d_new.Sigma == d.Sigma[idx])
        assert jnp.alltrue(d_new.mu == d.mu[idx])
        assert jnp.alltrue(d_new.ln_beta == d.ln_beta[idx])
        assert jnp.alltrue(d_new.ln_det_Sigma == d.ln_det_Sigma[idx])
        assert jnp.alltrue(d_new.ln_det_Lambda == d.ln_det_Lambda[idx])

    @pytest.mark.parametrize(
        "R, D, dim_x",
        [
            (100, 5, jnp.array([0, 1, 2])),
            (1, 5, jnp.array([0, 4])),
            (100, 1, jnp.array([0])),
            (100, 5, jnp.array([0, 3, 1])),
        ],
    )
    def test_get_marginal(self, R, D, dim_x):
        Sigma = jnp.zeros((R, D, D))
        Sigma = Sigma.at[:].set(jnp.eye(D)[None] * np.random.rand(R)[:, None, None])
        mu = jnp.asarray(np.random.randn(R, D))
        d = self.test_class(Sigma, mu)
        md = d.get_marginal(dim_x)
        assert jnp.alltrue(md.mu == mu[:, dim_x])
        idx = jnp.ix_(jnp.arange(R), dim_x, dim_x)
        assert jnp.alltrue(md.Sigma == Sigma[idx])
        assert jnp.alltrue(md.is_normalized())
        assert jnp.alltrue(md.integrate() == 1)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (100, 5, jnp.array([0, 1, 2])),
            (1, 5, jnp.array([0, 4])),
            (100, 5, jnp.array([0, 3, 1])),
        ],
    )
    def test_condition_on(self, R, D, dim_y):
        dim_xy = jnp.arange(D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        Sigma = jnp.zeros((R, D, D))
        rand_nums = np.random.rand(R)
        Sigma = Sigma.at[:].set(jnp.eye(D)[None] * rand_nums[:, None, None])
        Lambda = jnp.eye(D)[None] * 1.0 / rand_nums[:, None, None]
        idx_x = jnp.ix_(jnp.arange(R), dim_x, dim_x)
        idx_y = jnp.ix_(jnp.arange(R), dim_y, dim_y)
        mu = jnp.asarray(np.random.randn(R, D))
        d = self.test_class(Sigma, mu)
        M_x = -jnp.einsum("abc,acd->abd", Sigma[idx_x], Lambda[:, dim_x][:, :, dim_y])
        b_x = mu[:, dim_x] - jnp.einsum("abc,ac->ab", M_x, mu[:, dim_y])
        cd = d.condition_on(dim_y)
        assert jnp.alltrue(cd.Lambda == d.Lambda[idx_x])
        assert jnp.alltrue(cd.M == M_x)
        assert jnp.alltrue(cd.b == b_x)

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_to_dict(self, R, D):
        Sigma = jnp.zeros((R, D, D))
        rand_nums = np.random.rand(R) + 0.1
        Sigma = Sigma.at[:].set(jnp.eye(D)[None] * rand_nums[:, None, None])
        Lambda = jnp.eye(D)[None] * 1.0 / rand_nums[:, None, None]
        mu = jnp.asarray(np.random.randn(R, D))
        d = self.test_class(Sigma, mu)
        d_dict = d.to_dict()
        assert jnp.alltrue(d_dict["Sigma"] == Sigma)
        assert jnp.alltrue(d_dict["mu"] == mu)
        assert jnp.allclose(d_dict["Lambda"], Lambda)
        assert jnp.allclose(
            d_dict["ln_det_Sigma"],
            jnp.sum(jnp.log(jnp.diagonal(Sigma, axis1=1, axis2=2)), axis=1),
            atol=1e-6,
        )

