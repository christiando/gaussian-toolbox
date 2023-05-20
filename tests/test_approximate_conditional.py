from gaussian_toolbox import (
    pdf,
    approximate_conditional,
)
import pytest
from jax import numpy as jnp
from jax import config
import jax

config.update("jax_enable_x64", True)
import numpy as np


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


"""

class TestFullHCCovGaussianConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Du):
        noise_x = 1.0
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        rand_mat = np.random.rand(Dy, Dy) - 0.5
        Q, _ = np.linalg.qr(rand_mat)
        U = jnp.array(Q[:, :Du])
        # self.U = jnp.eye(Dx)[:, :Du]
        W = 1e-1 * np.random.randn(Du, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        L = np.random.rand(Dy, Dy)
        Sigma = Dy * np.eye(Dy) + 0.5 * (L + L.T)
        cond = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            Sigma=jnp.array([Sigma]),
            U=U,
            W=W,
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = 10 * self.get_pd_matrix(
            R, Dx
        )  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
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
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 2),
            (1, 10, 3, 1),
            (1, 2, 5, 2),
        ],
    )
    def test_affine_tranformations(self, R, Dx, Dy, Du):
        cond, p_X = self.create_instance(R, Dx, Dy, Du)
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
        "R, Dx, Dy, Du",
        [(1, 5, 1, 1), (1, 10, 1, 1), (1, 5, 2, 2), (1, 1, 10, 2)],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy, Du):
        N = 1
        y = jnp.array(np.random.randn(N, Dy))
        cond, p_X = self.create_instance(R, Dx, Dy, Du)
        integral_lb = cond.integrate_log_conditional_y(p_X, y=y)
        print(integral_lb.shape)
        key = jax.random.PRNGKey(42)
        X_sample = p_X.sample(key, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        assert jnp.all(integral_lb <= integral_sample_mean + 1e-2 * integral_sample_std)
"""


class TestHCCovGaussianConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Du):
        noise_x = 1.0
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        rand_mat = np.random.rand(Dy, Dy) - 0.5
        Q, _ = np.linalg.qr(rand_mat)
        U = jnp.array(Q[:, :Du])
        # self.U = jnp.eye(Dx)[:, :Du]
        W = 1e-1 * np.random.randn(Du, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        beta = 0.5**2 * jnp.ones(Du)
        sigma_x = jnp.array([noise_x])
        cond = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            sigma_x=sigma_x,
            U=U,
            W=W,
            beta=beta,
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = 10 * self.get_pd_matrix(
            R, Dx
        )  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
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
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 2),
            (1, 10, 3, 1),
            (1, 2, 5, 2),
        ],
    )
    def test_affine_tranformations(self, R, Dx, Dy, Du):
        cond, p_X = self.create_instance(R, Dx, Dy, Du)
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
        "R, Dx, Dy, Du",
        [(1, 5, 1, 1), (1, 10, 1, 1), (1, 5, 2, 2), (1, 1, 10, 2)],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy, Du):
        N = 1
        y = jnp.array(np.random.randn(N, Dy))
        cond, p_X = self.create_instance(R, Dx, Dy, Du)
        integral_lb = cond.integrate_log_conditional_y(p_X, y=y)
        key = jax.random.PRNGKey(42)
        X_sample = p_X.sample(key, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        assert jnp.all(integral_lb <= integral_sample_mean + 1e-2 * integral_sample_std)


class TestHeteroscedasticExpConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        rand_mat = np.random.rand(Dy, Dy) - 0.5
        Q, _ = np.linalg.qr(rand_mat)
        # self.U = jnp.eye(Dx)[:, :Du]
        A = jnp.array(np.random.randn(Dy, Da))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = approximate_conditional.HeteroscedasticExpConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A=jnp.array([A]),
            W=W,
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(
            R, Dx
        )  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
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
        "R, Dx, Dy, Da, Dk",
        [
            (1, 5, 2, 3, 2),
            (1, 10, 3, 4, 1),
            (1, 2, 5, 5, 3),
        ],
    )
    def test_affine_tranformations(self, R, Dx, Dy, Da, Dk):
        cond, p_X = self.create_instance(R, Dx, Dy, Da, Dk)
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
        "R, Dx, Dy, Da, Dk",
        [
            (1, 5, 2, 3, 2),
            (1, 10, 3, 4, 1),
            (1, 2, 5, 5, 3),
        ],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy, Da, Dk):
        N = 1
        y = jnp.array(np.random.randn(N, Dy))
        cond, p_X = self.create_instance(R, Dx, Dy, Da, Dk)
        integral_lb = cond.integrate_log_conditional_y(p_X, y=y)
        key = jax.random.PRNGKey(42)
        X_sample = p_X.sample(key, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)

        assert jnp.all(integral_lb <= integral_sample_mean + .1 * integral_sample_std)
        
        
class TestHeteroscedasticCoshM1Conditional(TestHeteroscedasticExpConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        rand_mat = np.random.rand(Dy, Dy) - 0.5
        Q, _ = np.linalg.qr(rand_mat)
        # self.U = jnp.eye(Dx)[:, :Du]
        A = jnp.array(np.random.randn(Dy, Da))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = approximate_conditional.HeteroscedasticCoshM1Conditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A=jnp.array([A]),
            W=W,
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(
            R, Dx
        )  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X
