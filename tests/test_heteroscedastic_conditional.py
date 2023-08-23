from gaussian_toolbox import (
    heteroscedastic_conditional,
    pdf,
)
import pytest
from jax import numpy as jnp
from jax import config
import jax

config.update("jax_enable_x64", True)
import numpy as np

np.random.seed(0)


class TestHeteroscedasticExpConditional:
    @classmethod
    def create_instance(
        self,
        R,
        Dx,
        Dy,
    ):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Dy - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dy, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="exp",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=0.1 * Sigma_x, mu=0.1 * mu_x)
        return cond, p_X

    @staticmethod
    def get_pd_matrix(R, D, eigen_mu=1):
        # Q = jnp.array(np.random.randn((R, D, D))
        # eig_vals = jnp.abs(eigen_mu + jnp.array(np.random.randn((R, D)))
        # psd_mat = jnp.einsum("abc,abd->acd", Q * eig_vals[:, :, None], Q)
        # psd_mat = 0.5 * (psd_mat + jnp.swapaxes(psd_mat, -1, -2))
        A = jnp.array(np.random.rand(R, D, D))
        psd_mat = jnp.einsum("abc,abd->acd", A, A)
        psd_mat += jnp.eye(D)[None]
        return psd_mat

    @pytest.mark.parametrize(
        "R, Dx, Dy",
        [
            (
                1,
                5,
                2,
            ),
            (
                1,
                10,
                3,
            ),
            (
                1,
                2,
                5,
            ),
        ],
    )
    def test_affine_tranformations(
        self,
        R,
        Dx,
        Dy,
    ):
        cond, p_X = self.create_instance(
            R,
            Dx,
            Dy,
        )
        p_YX = cond.affine_joint_transformation(p_X)
        p_Y = cond.affine_marginal_transformation(p_X)
        p_X_given_Y = cond.affine_conditional_transformation(p_X)
        y = jnp.array(np.random.randn(R, Dy))
        p_X_given_Y2 = cond.affine_variational_conditional_transformation(p_X, y)
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
        "R, Dx, Dy,",
        [
            (
                1,
                5,
                2,
            ),
            (
                1,
                10,
                3,
            ),
            (
                1,
                2,
                5,
            ),
        ],
    )
    def test_integrate_log_conditional_y(
        self,
        R,
        Dx,
        Dy,
    ):
        N = 1
        y = jnp.array(np.random.randn(N, Dy))
        cond, p_X = self.create_instance(
            R,
            Dx,
            Dy,
        )
        integral_lb = cond.integrate_log_conditional_y(p_X, y=y)
        key = jax.random.PRNGKey(42)
        X_sample = p_X.sample(key, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)

        assert jnp.all(integral_lb <= integral_sample_mean + 0.1 * integral_sample_std)


class TestHeteroscedasticSigmoidConditional(TestHeteroscedasticExpConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Dy - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dy, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="sigmoid",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X
