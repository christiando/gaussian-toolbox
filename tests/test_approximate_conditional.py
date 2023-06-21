from gaussian_toolbox import (
    approximate_conditional,
    pdf,
)
import pytest
from jax import numpy as jnp
from jax import config
import jax

config.update("jax_enable_x64", True)
import numpy as np

np.random.seed(0)


class TestLRBFGaussianConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Dk):
        Dphi = Dx + Dk
        M = jnp.array(np.random.randn(R, Dy, Dphi))
        b = jnp.array(np.random.randn(R, Dy))
        length_scale = jnp.array(np.random.randn(Dk, Dx))
        mu_k = jnp.array(np.random.randn(Dk, Dx))
        Sigma = self.get_pd_matrix(R, Dy)
        cond = approximate_conditional.LRBFGaussianConditional(
            M=M, b=b, mu=mu_k, length_scale=length_scale, Sigma=Sigma
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X

    @staticmethod
    def get_pd_matrix(R, D, eigen_mu=1):
        A = jnp.array(np.random.randn(R, D, D))
        psd_mat = jnp.einsum("abc,abd->acd", A, A)
        psd_mat += jnp.eye(D)[None]
        return psd_mat

    @pytest.mark.parametrize(
        "R, Dx, Dy, Dk",
        [
            (1, 5, 2, 2),
            (1, 10, 3, 1),
            (1, 2, 5, 5),
        ],
    )
    def test_affine_tranformations(self, R, Dx, Dy, Dk):
        cond, p_X = self.create_instance(R, Dx, Dy, Dk)
        p_YX = cond.affine_joint_transformation(p_X)
        p_Y = cond.affine_marginal_transformation(p_X)
        p_X_given_Y = cond.affine_conditional_transformation(p_X)
        dim_y = jnp.arange(Dx, Dy + Dx)
        p_Y2 = p_YX.get_marginal(dim_y)
        assert jnp.allclose(p_Y.mu, p_Y2.mu)
        assert jnp.allclose(p_Y.Sigma, p_Y2.Sigma)
        assert jnp.allclose(p_Y.Lambda, p_Y2.Lambda)
        p_X_given_Y2 = p_YX.condition_on(dim_y)
        assert jnp.allclose(p_X_given_Y.M, p_X_given_Y2.M)
        assert jnp.allclose(p_X_given_Y.b, p_X_given_Y2.b)
        assert jnp.allclose(p_X_given_Y.Sigma, p_X_given_Y2.Sigma)
        assert jnp.allclose(p_X_given_Y.Lambda, p_X_given_Y2.Lambda)

    @pytest.mark.parametrize(
        "R, Dx, Dy, Dk",
        [
            (1, 5, 2, 2),
            (1, 10, 3, 1),
            (1, 2, 5, 5),
        ],
    )
    def test_condition_on_x(self, R, Dx, Dy, Dk):
        cond, p_X = self.create_instance(R, Dx, Dy, Dk)
        x = jnp.array(np.random.randn(2, Dx))
        cond_x = cond.condition_on_x(x)
        mu_x = jnp.einsum("abc,dc->adb", cond.M, cond.evaluate_phi(x)) + cond.b[:, None]
        assert jnp.allclose(cond_x.mu, mu_x)

    @pytest.mark.parametrize(
        "R, Dx, Dy, Dk",
        [
            (1, 5, 2, 2),
            (1, 10, 3, 1),
            (1, 2, 5, 5),
        ],
    )
    def test_integrate_log_conditional(self, R, Dx, Dy, Dk):
        cond, p_X = self.create_instance(R, Dx, Dy, Dk)
        pxy = cond.affine_joint_transformation(p_X)
        dim_y = jnp.arange(Dx, Dy + Dx)
        dim_x = jnp.arange(0, Dx)
        key = jax.random.PRNGKey(0)
        xy = pxy.sample(key, 100000)[:, 0]
        y = xy[:, dim_y]
        r_ana = cond.integrate_log_conditional(pxy)
        r_ana_sample = jnp.mean(cond.integrate_log_conditional_y(p_X)(y), axis=0)
        assert jnp.allclose(r_ana_sample, r_ana, atol=1e-2, rtol=np.inf)


class TestLSEMGaussianConditional(TestLRBFGaussianConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy, Dk):
        Dphi = Dx + Dk
        M = jnp.array(np.random.randn(R, Dy, Dphi))
        b = jnp.array(np.random.randn(R, Dy))
        W = jnp.array(np.random.randn(Dk, Dx + 1))
        mu_k = jnp.array(np.random.randn(Dk, Dx))
        Sigma = self.get_pd_matrix(R, Dy)
        cond = approximate_conditional.LSEMGaussianConditional(
            M=M, b=b, W=W, Sigma=Sigma
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X
