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
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
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

        assert jnp.all(integral_lb <= integral_sample_mean + 0.1 * integral_sample_std)


class TestHeteroscedasticCoshM1Conditional(TestHeteroscedasticExpConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="coshm1",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X


class TestHeteroscedasticReLUConditional(TestHeteroscedasticExpConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="ReLU",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X


class TestHeteroscedasticHeavisideConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="heaviside",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
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
            (1, 1, 2, 3, 2),
            (1, 5, 2, 3, 3),
            (1, 10, 3, 4, 4),
            (1, 2, 5, 5, 1),
        ],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy, Da, Dk):
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        N = 10
        cond, p_X = self.create_instance(R, Dx, Dy, Da, Dk)
        p_X_tiled = pdf.GaussianPDF(
            Sigma=jnp.tile(p_X.Sigma, (N, 1, 1)), mu=jnp.tile(p_X.mu, (N, 1))
        )
        p_Y = cond.affine_marginal_transformation(p_X)
        y = p_Y.sample(subkey, N)[:, 0]
        print(y.shape, p_X_tiled.R)
        integral_lb = cond.integrate_log_conditional_y(p_X_tiled, y=y)
        key, subkey = jax.random.split(key)
        X_sample = p_X.sample(subkey, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        print(integral_lb, integral_sample_mean, integral_sample_std)
        assert jnp.allclose(
            integral_lb, integral_sample_mean, atol=1e-1 * integral_sample_std
        )


class TestHeteroscedasticExpConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
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

        assert jnp.all(integral_lb <= integral_sample_mean + 0.1 * integral_sample_std)


class TestHeteroscedasticCoshM1Conditional(TestHeteroscedasticExpConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="coshm1",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X


class TestHeteroscedasticReLUConditional(TestHeteroscedasticExpConditional):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="ReLU",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X


class TestHeteroscedasticHeavisideConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy * Da - Dy * (Dy - 1) // 2))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.HeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="heaviside",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
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
            (1, 1, 2, 3, 2),
            (1, 5, 2, 3, 3),
            (1, 10, 3, 4, 4),
            (1, 2, 5, 5, 1),
        ],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy, Da, Dk):
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        N = 10
        cond, p_X = self.create_instance(R, Dx, Dy, Da, Dk)
        p_X_tiled = pdf.GaussianPDF(
            Sigma=jnp.tile(p_X.Sigma, (N, 1, 1)), mu=jnp.tile(p_X.mu, (N, 1))
        )
        p_Y = cond.affine_marginal_transformation(p_X)
        y = p_Y.sample(subkey, N)[:, 0]
        print(y.shape, p_X_tiled.R)
        integral_lb = cond.integrate_log_conditional_y(p_X_tiled, y=y)
        key, subkey = jax.random.split(key)
        X_sample = p_X.sample(subkey, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        print(integral_lb, integral_sample_mean, integral_sample_std)
        assert jnp.allclose(
            integral_lb, integral_sample_mean, atol=1e-1 * integral_sample_std
        )


class TestScalableHeteroscedasticExpConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy + Dy * (Da - Dy)))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.ScalableHeteroscedasticConditional(
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

        assert jnp.all(integral_lb <= integral_sample_mean + 0.1 * integral_sample_std)


class TestScalableHeteroscedasticCoshM1Conditional(
    TestScalableHeteroscedasticExpConditional
):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy + Dy * (Da - Dy)))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.ScalableHeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="coshm1",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X


class TestScalableHeteroscedasticReLUConditional(
    TestScalableHeteroscedasticExpConditional
):
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy + Dy * (Da - Dy)))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.ScalableHeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="ReLU",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
        p_X = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu_x)
        return cond, p_X


class TestScalableHeteroscedasticHeavisideConditional:
    @classmethod
    def create_instance(self, R, Dx, Dy, Da, Dk):
        C = jnp.array(np.random.randn(Dy, Dx))
        C = C / jnp.sqrt(jnp.sum(C**2, axis=0))[None]
        d = 1e-1 * jnp.array(np.random.randn(Dy))
        A_vec = jnp.array(np.random.randn(Dy + Dy * (Da - Dy)))
        W = 1e-1 * np.random.randn(Dk, Dx + 1)
        W[:, 0] = 0
        W = jnp.array(W)
        cond = heteroscedastic_conditional.ScalableHeteroscedasticConditional(
            M=jnp.array([C]),
            b=jnp.array([d]),
            A_vec=A_vec,
            W=W,
            link_function="heaviside",
        )
        mu_x = jnp.array(np.random.randn(R, Dx))
        Sigma_x = self.get_pd_matrix(R, Dx)  # jnp.tile(jnp.eye(Dx)[None], (R, 1, 1))#
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
            (1, 1, 2, 3, 2),
            (1, 5, 2, 3, 3),
            (1, 10, 3, 4, 4),
            (1, 2, 5, 5, 1),
        ],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy, Da, Dk):
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        N = 10
        cond, p_X = self.create_instance(R, Dx, Dy, Da, Dk)
        p_X_tiled = pdf.GaussianPDF(
            Sigma=jnp.tile(p_X.Sigma, (N, 1, 1)), mu=jnp.tile(p_X.mu, (N, 1))
        )
        p_Y = cond.affine_marginal_transformation(p_X)
        y = p_Y.sample(subkey, N)[:, 0]
        print(y.shape, p_X_tiled.R)
        integral_lb = cond.integrate_log_conditional_y(p_X_tiled, y=y)
        key, subkey = jax.random.split(key)
        X_sample = p_X.sample(subkey, 100000)
        integral_sample_mean = jnp.mean(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        integral_sample_std = jnp.std(cond(X_sample[:, 0]).evaluate_ln(y), axis=0)
        print(integral_lb, integral_sample_mean, integral_sample_std)
        assert jnp.allclose(
            integral_lb, integral_sample_mean, atol=1e-1 * integral_sample_std
        )
