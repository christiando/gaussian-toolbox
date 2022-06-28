from gaussian_toolbox.gaussian_algebra import densities, conditionals
import pytest
from jax import numpy as jnp
from jax import scipy as jsc
import numpy as np


class TestConditionalGaussianDensity:
    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (200, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_affine_transformation1(self, R, D, dim_y):
        dim_xy = jnp.arange(D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        Sigma = jnp.zeros((R, D, D))
        rand_nums = np.random.rand(D)[None] * np.ones(R)[:,None]
        Sigma = Sigma.at[:].set(jnp.eye(D)[None] * rand_nums[:, None])
        mu = jnp.asarray(np.ones(R)[:, None] * np.random.randn(D)[None])
        pxy = densities.GaussianDensity(Sigma, mu)
        px_given_y = pxy.condition_on(dim_y).slice(jnp.array([0]))
        py = pxy.get_marginal(dim_y)
        pxy_new = px_given_y.affine_joint_transformation(py)
        idx = jnp.concatenate([dim_y, dim_x])
        assert jnp.allclose(mu[:, idx], pxy_new.mu)
        assert jnp.allclose(Sigma[:, idx][:,:,idx], pxy_new.Sigma)
        
    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (200, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_affine_transformation2(self, R, D, dim_y):
        dim_xy = jnp.arange(D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        Sigma = jnp.zeros((R, D, D))
        rand_nums = np.random.rand(D)[None] * np.ones(R)[:,None]
        Sigma = Sigma.at[:].set(jnp.eye(D)[None] * rand_nums[:, None])
        mu = jnp.asarray(np.ones(R)[:, None] * np.random.randn(D)[None])
        pxy = densities.GaussianDensity(Sigma, mu)
        px_given_y = pxy.condition_on(dim_y).slice(jnp.array([0]))
        py = pxy.get_marginal(dim_y)
        px = pxy.get_marginal(dim_x)
        px_new = px_given_y.affine_marginal_transformation(py)
        assert jnp.allclose(px.mu, px_new.mu)
        assert jnp.allclose(px.Sigma, px_new.Sigma)
        
        
    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (200, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_affine_transformation3(self, R, D, dim_y):
        dim_xy = jnp.arange(D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        Sigma = jnp.zeros((R, D, D))
        rand_nums = np.random.rand(D)[None] * np.ones(R)[:,None]
        Sigma = Sigma.at[:].set(jnp.eye(D)[None] * rand_nums[:, None])
        mu = jnp.asarray(np.ones(R)[:, None] * np.random.randn(D)[None])
        pxy = densities.GaussianDensity(Sigma, mu)
        px_given_y = pxy.condition_on(dim_y).slice(jnp.array([0]))
        py_given_x = pxy.condition_on(dim_x).slice(jnp.array([0]))
        py = pxy.get_marginal(dim_y)
        py_given_x_new = px_given_y.affine_conditional_transformation(py).slice(jnp.array([0]))
        sort_idx = jnp.argsort(dim_y)
        assert jnp.allclose(py_given_x.M, py_given_x_new.M[:,sort_idx,:])
        assert jnp.allclose(py_given_x.b, py_given_x_new.b[:,sort_idx])
        assert jnp.allclose(py_given_x.Sigma, py_given_x_new.Sigma[:,sort_idx][:,:,sort_idx])
