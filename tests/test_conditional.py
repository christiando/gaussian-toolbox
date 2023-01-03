from gaussian_toolbox import pdf, conditional, measure
from gaussian_toolbox.utils import linalg
import pytest
from jax import numpy as jnp
from jax import scipy as jsc
from jax import config
import jax

config.update("jax_enable_x64", True)
import numpy as np
import objax


class TestConditionalGaussianPDF:
    @classmethod
    def create_instance(self, R, D, dim_y):
        dim_xy = jnp.arange(D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        Sigma = self.get_pd_matrix(R, D)
        mu = objax.random.normal((R, D))
        pxy = pdf.GaussianPDF(Sigma=Sigma, mu=mu)
        px_given_y = pxy.condition_on(dim_y).slice(jnp.array([0]))
        # py = pxy.get_marginal(dim_y)
        return px_given_y, pxy, dim_x

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

    @pytest.mark.parametrize(
        "R, Dx, Dy",
        [
            (1, 5, 2),
            (1, 10, 3),
            (1, 2, 5),
        ],
    )
    def test_init(self, R, Dx, Dy):
        Sigma = self.get_pd_matrix(R, Dy)
        M = objax.random.normal((R, Dy, Dx))
        b = objax.random.normal((R, Dy))
        Lambda, ln_det_Sigma = linalg.invert_matrix(Sigma)
        cond = conditional.ConditionalGaussianPDF(M=M, b=b, Sigma=Sigma)
        assert jnp.allclose(cond.Lambda, Lambda)
        assert jnp.allclose(cond.ln_det_Sigma, ln_det_Sigma)
        assert jnp.allclose(cond.b, b)
        assert jnp.allclose(cond.M, M)
        Lambda, ln_det_Sigma = linalg.invert_matrix(Sigma)
        cond = conditional.ConditionalGaussianPDF(M=M, Lambda=Lambda)
        assert jnp.allclose(cond.Sigma, Sigma)
        assert jnp.allclose(cond.Lambda, Lambda)
        assert jnp.allclose(cond.ln_det_Sigma, ln_det_Sigma)
        assert jnp.allclose(cond.b, jnp.zeros((R, Dy)))
        assert jnp.allclose(cond.M, M)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_affine_transformation1(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        py = pxy.get_marginal(dim_y)
        pxy_new = px_given_y.affine_joint_transformation(py)
        idx = jnp.concatenate([dim_y, dim_x])
        assert jnp.allclose(pxy.mu[:1, idx], pxy_new.mu, atol=1e-6)
        assert jnp.allclose(pxy.Sigma[:1, idx][:, :, idx], pxy_new.Sigma, atol=1e-6)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_affine_transformation2(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        py = pxy.get_marginal(dim_y)
        px = pxy.get_marginal(dim_x)
        px_new = px_given_y.affine_marginal_transformation(py)
        assert jnp.allclose(px.mu, px_new.mu)
        assert jnp.allclose(px.Sigma, px_new.Sigma)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_affine_transformation3(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        py_given_x = pxy.condition_on(dim_x).slice(jnp.array([0]))
        py = pxy.get_marginal(dim_y)
        py_given_x_new = px_given_y.affine_conditional_transformation(py).slice(
            jnp.array([0])
        )
        sort_idx = jnp.argsort(dim_y)
        assert jnp.allclose(py_given_x.M, py_given_x_new.M[:, sort_idx, :])
        assert jnp.allclose(py_given_x.b, py_given_x_new.b[:, sort_idx])
        assert jnp.allclose(
            py_given_x.Sigma, py_given_x_new.Sigma[:, sort_idx][:, :, sort_idx]
        )

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_condition_on_x(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        xy = objax.random.normal((2, D))
        x = xy[:, dim_x]
        y = xy[:, dim_y]
        cond_x = px_given_y.condition_on_x(y)
        mu_x = jnp.einsum("abc,dc->adb", px_given_y.M, y) + px_given_y.b[:, None]

        assert jnp.allclose(cond_x.mu, mu_x)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([1, 3, 5])),
            (1, 10, jnp.array(jnp.array([8, 3, 5]))),
        ],
    )
    def test_set_y(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        xy = objax.random.normal((2, D))
        x = xy[:, dim_x]
        y = xy[:, dim_y]
        set_y = px_given_y.set_y(x)

        x_minus_b = x - px_given_y.b
        Lambda_new = jnp.einsum(
            "abc,acd->abd",
            jnp.einsum("abd, abc -> adc", px_given_y.M, px_given_y.Lambda),
            px_given_y.M,
        )
        nu_new = jnp.einsum(
            "abc, ab -> ac",
            jnp.einsum("abc, acd -> abd", px_given_y.Lambda, px_given_y.M),
            x_minus_b,
        )
        yb_Lambda_yb = jnp.einsum(
            "ab, ab-> a",
            jnp.einsum("ab, abc -> ac", x_minus_b, px_given_y.Lambda),
            x_minus_b,
        )
        ln_beta_new = -0.5 * (
            yb_Lambda_yb + px_given_y.Dx * jnp.log(2 * jnp.pi) + px_given_y.ln_det_Sigma
        )

        assert jnp.allclose(set_y.Lambda, Lambda_new)
        assert jnp.allclose(set_y.nu, nu_new)
        assert jnp.allclose(set_y.ln_beta, ln_beta_new)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([0, 1, 2, 3])),
            (1, 10, jnp.array(jnp.array([0, 1, 2]))),
        ],
    )
    def test_integrate_log_conditional(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        py = pxy.get_marginal(dim_y)
        key = jax.random.PRNGKey(0)
        xy = pxy.sample(key, 100000)[:, 0]
        x = xy[:, dim_x]
        y = xy[:, dim_y]
        r_sample = jnp.mean(
            px_given_y.set_y(x).evaluate_ln(y, element_wise=True), axis=0
        )
        print(x.shape)
        r_ana = px_given_y.integrate_log_conditional(pxy)
        r_ana_sample = jnp.mean(px_given_y.integrate_log_conditional_y(py)(x), axis=0)
        assert jnp.allclose(r_sample, r_ana, atol=1e-2, rtol=np.inf)
        assert jnp.allclose(r_sample, r_ana_sample, atol=1e-2, rtol=np.inf)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([0, 1, 2, 3])),
            (1, 10, jnp.array(jnp.array([0, 1, 2]))),
        ],
    )
    def test_mutual_information(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        py = pxy.get_marginal(dim_y)
        px = pxy.get_marginal(dim_x)
        py_given_x = pxy.condition_on(dim_x)
        mi1 = py_given_x.mutual_information(px)
        mi2 = px_given_y.mutual_information(py)
        assert jnp.allclose(mi1, mi2)

    @pytest.mark.parametrize(
        "R, D, dim_y",
        [
            (1, 5, jnp.array([0, 1, 2])),
            (1, 10, jnp.array([0, 1, 2, 3])),
            (1, 10, jnp.array(jnp.array([0, 1, 2]))),
        ],
    )
    def test_update_Sigma(self, R, D, dim_y):
        px_given_y, pxy, dim_x = self.create_instance(R, D, dim_y)
        Sigma_new = self.get_pd_matrix(R, len(dim_x))
        Lambda_new, ln_det_Sigma = linalg.invert_matrix(Sigma_new)
        px_given_y.update_Sigma(Sigma_new)
        assert jnp.allclose(px_given_y.Sigma, Sigma_new)
        assert jnp.allclose(px_given_y.Lambda, Lambda_new)
        assert jnp.allclose(px_given_y.ln_det_Sigma, ln_det_Sigma)
       

class TestNNControlGaussianConditional:
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

    @classmethod
    def create_instance(self, R, Dx, Dy, Du):
        Sigma = self.get_pd_matrix(R, Dy)
        control_func = lambda u: jnp.tanh(jnp.dot(jnp.ones((Dy * (Dx + 1), Du)), u.T)).T
        cond = conditional.NNControlGaussianConditional(Sigma=Sigma, num_cond_dim=Dx, num_control_dim=Du, control_func=control_func)
        return cond

    @pytest.mark.parametrize(
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 4),
            (1, 10, 1, 3),
            (1, 2, 4, 10),
        ],
    )
    def test_init(self, R, Dx, Dy, Du):
        cond = self.create_instance(R, Dx, Dy, Du)
        Lambda, ln_det_Sigma = linalg.invert_matrix(cond.Sigma)
        assert jnp.allclose(cond.Lambda, Lambda)
        assert jnp.allclose(cond.ln_det_Sigma, ln_det_Sigma)

    @pytest.mark.parametrize(
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 4),
            (1, 10, 1, 3),
            (1, 2, 4, 10),
        ],
    )
    def test_set_control_variable(self, R, Dx, Dy, Du):
        cond_u = self.create_instance(R, Dx, Dy, Du)
        Lambda, ln_det_Sigma = linalg.invert_matrix(cond_u.Sigma)
        u = objax.random.normal((R, Du))
        cond = cond_u.set_control_variable(u)
        assert jnp.allclose(cond.Lambda, Lambda)
        assert jnp.allclose(cond.ln_det_Sigma, ln_det_Sigma)
        assert cond.M.shape == (R, Dy, Dx)
        assert cond.b.shape == (R, Dy)

    @pytest.mark.parametrize(
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 4),
            (1, 10, 1, 3),
            (1, 2, 4, 10),
        ],
    )
    def test_mutual_information(self, R, Dx, Dy, Du):
        cond_u = self.create_instance(R, Dx, Dy, Du)
        u = objax.random.normal((R, Du))
        Sigma_x = self.get_pd_matrix(R, Dx)
        mu = objax.random.normal((R, Dx))
        px = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu)
        cond2 = cond_u.affine_conditional_transformation(px, u=u)
        py = cond_u.affine_marginal_transformation(px, u=u)
        mi1 = cond_u.mutual_information(px, u=u)
        mi2 = cond2.mutual_information(py)
        assert jnp.allclose(mi1, mi2)

    @pytest.mark.parametrize(
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 4),
            (1, 10, 1, 3),
            (1, 2, 4, 10),
        ],
    )
    def test_set_y(self, R, Dx, Dy, Du):
        cond_u = self.create_instance(R, Dx, Dy, Du)
        u = objax.random.normal((R, Du))
        py_given_x = cond_u.set_control_variable(u)
        D = Dx + Dy
        dim_y = jnp.arange(Dy)
        dim_x = jnp.arange(Dy, D)
        xy = objax.random.normal((2, D))
        x = xy[:, dim_x]
        y = xy[:, dim_y]
        set_y = cond_u.set_y(y, u=u)

        y_minus_b = y - py_given_x.b
        Lambda_new = jnp.einsum(
            "abc,acd->abd",
            jnp.einsum("abd, abc -> adc", py_given_x.M, py_given_x.Lambda),
            py_given_x.M,
        )
        nu_new = jnp.einsum(
            "abc, ab -> ac",
            jnp.einsum("abc, acd -> abd", py_given_x.Lambda, py_given_x.M),
            y_minus_b,
        )
        yb_Lambda_yb = jnp.einsum(
            "ab, ab-> a",
            jnp.einsum("ab, abc -> ac", y_minus_b, py_given_x.Lambda),
            y_minus_b,
        )
        ln_beta_new = -0.5 * (
            yb_Lambda_yb + py_given_x.Dx * jnp.log(2 * jnp.pi) + py_given_x.ln_det_Sigma
        )

        assert jnp.allclose(set_y.Lambda, Lambda_new)
        assert jnp.allclose(set_y.nu, nu_new)
        assert jnp.allclose(set_y.ln_beta, ln_beta_new)

    @pytest.mark.parametrize(
        "R, Dx, Dy, Du",
        [
            (1, 5, 2, 4),
            (1, 10, 1, 3),
            (1, 2, 4, 10),
        ],
    )
    def test_integrate_log_conditional(self, R, Dx, Dy, Du):
        cond_u = self.create_instance(R, Dx, Dy, Du)
        u = objax.random.normal((R, Du))
        Sigma_x = self.get_pd_matrix(R, Dx)
        mu = objax.random.normal((R, Dx))
        px = pdf.GaussianPDF(Sigma=Sigma_x, mu=mu)
        pxy = cond_u.affine_joint_transformation(px, u)
        D = Dx + Dy
        dim_y = jnp.arange(Dy)
        dim_x = jnp.arange(Dy, D)
        py = pxy.get_marginal(dim_y)
        key = jax.random.PRNGKey(0)
        xy = pxy.sample(key, 100000)[:, 0]
        x = xy[:, dim_x]
        y = xy[:, dim_y]
        r_sample = jnp.mean(
            cond_u.set_y(y, u).evaluate_ln(x, element_wise=True), axis=0
        )
        r_ana = cond_u.integrate_log_conditional(pxy, u=u)
        r_ana_sample = jnp.mean(cond_u.integrate_log_conditional_y(px, u=u)(y), axis=0)
        assert jnp.allclose(r_sample, r_ana, atol=1e-2, rtol=np.inf)
        assert jnp.allclose(r_sample, r_ana_sample, atol=1e-2, rtol=np.inf)

class TestConditionalIdentityGaussianPDF:
    @classmethod
    def create_instance(self, R, D):
        Sigma_y = self.get_pd_matrix(R, D)
        mu = objax.random.normal((R, D))
        py = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu)
        Sigma_xy = self.get_pd_matrix(R, D)
        M = jnp.tile(jnp.eye(D)[None], (R, 1, 1))
        b = jnp.zeros((R, D))
        px_given_y1 = conditional.ConditionalGaussianPDF(M=M, b=b, Sigma=Sigma_xy)
        px_given_y2 = conditional.ConditionalIdentityGaussianPDF(Sigma=Sigma_xy)

        # py = pxy.get_marginal(dim_y)
        return px_given_y1, px_given_y2, py

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

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_affine_transformation1(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        pxy1 = px_given_y1.affine_joint_transformation(py)
        pxy2 = px_given_y2.affine_joint_transformation(py)
        assert jnp.allclose(pxy1.mu, pxy2.mu, atol=1e-6)
        assert jnp.allclose(pxy1.ln_det_Sigma, pxy2.ln_det_Sigma, atol=1e-6)
        assert jnp.allclose(pxy1.Sigma, pxy2.Sigma, atol=1e-6)

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_affine_transformation3(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        py_given_x1 = px_given_y1.affine_conditional_transformation(py).slice(
            jnp.array([0])
        )
        py_given_x2 = px_given_y2.affine_conditional_transformation(py).slice(
            jnp.array([0])
        )
        assert jnp.allclose(py_given_x1.M, py_given_x2.M)
        assert jnp.allclose(py_given_x1.b, py_given_x2.b)
        assert jnp.allclose(py_given_x1.Sigma, py_given_x2.Sigma)
        assert jnp.allclose(py_given_x1.Lambda, py_given_x2.Lambda)

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_affine_transformation2(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        px1 = px_given_y1.affine_marginal_transformation(py)
        px2 = px_given_y2.affine_marginal_transformation(py)
        assert jnp.allclose(px1.mu, px2.mu)
        assert jnp.allclose(px1.Sigma, px2.Sigma)
        assert jnp.allclose(px1.Lambda, px2.Lambda)

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_condition_on_x(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        x = objax.random.normal((2, D))
        cond_x1 = px_given_y1.condition_on_x(x)
        cond_x2 = px_given_y2.condition_on_x(x)
        assert jnp.allclose(cond_x1.mu, cond_x2.mu)
        assert jnp.allclose(cond_x1.Sigma, cond_x2.Sigma)

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_set_y(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        y = objax.random.normal((2, D))
        set_y1 = px_given_y1.set_y(y)
        set_y2 = px_given_y2.set_y(y)

        assert jnp.allclose(set_y1.Lambda, set_y2.Lambda)
        assert jnp.allclose(set_y1.nu, set_y2.nu)
        assert jnp.allclose(set_y1.ln_beta, set_y2.ln_beta)

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_integrate_log_conditional(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        pxy1 = px_given_y1.affine_joint_transformation(py)
        pxy2 = px_given_y2.affine_joint_transformation(py)
        r1 = px_given_y1.integrate_log_conditional(pxy1)
        r2 = px_given_y2.integrate_log_conditional(pxy2)
        px = px_given_y1.affine_marginal_transformation(py)
        key = jax.random.PRNGKey(0)
        y = py.sample(key, 1)[0]
        r_ana1_y = px_given_y1.integrate_log_conditional_y(px, y=y)
        r_ana2_y = px_given_y2.integrate_log_conditional_y(px, y=y)
        assert jnp.allclose(r1, r2)
        assert jnp.allclose(r_ana1_y, r_ana2_y, atol=1e-2, rtol=np.inf)
        

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_mutual_information(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        mi1 = px_given_y1.mutual_information(py)
        mi2 = px_given_y2.mutual_information(py)
        assert jnp.allclose(mi1, mi2)

    @pytest.mark.parametrize(
        "R, D",
        [
            (1, 1),
            (1, 2),
            (1, 10),
        ],
    )
    def test_update_Sigma(self, R, D):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        Sigma_new = self.get_pd_matrix(R, D)
        px_given_y1.update_Sigma(Sigma_new)
        px_given_y2.update_Sigma(Sigma_new)
        assert jnp.allclose(px_given_y1.Sigma, px_given_y2.Sigma)
        assert jnp.allclose(px_given_y1.Lambda, px_given_y2.Lambda)
        assert jnp.allclose(px_given_y1.ln_det_Sigma, px_given_y2.ln_det_Sigma)
        
        
    
    @pytest.mark.parametrize(
        "R, D, idx",
        [
            (10, 1, jnp.array([1,4,9])),
            (10, 2, jnp.array([1,4,5])),
            (10, 10, jnp.array([1,4,-1])),
        ],
    )
    def test_update_Sigma(self, R, D, idx):
        px_given_y1, px_given_y2, py = self.create_instance(R, D)
        px_given_y2_sliced = px_given_y2.slice(idx)
        assert jnp.allclose(px_given_y2_sliced.Sigma, px_given_y2.Sigma[idx]) 
        assert jnp.allclose(px_given_y2_sliced.Lambda, px_given_y2.Lambda[idx]) 
        assert jnp.allclose(px_given_y2_sliced.ln_det_Sigma, px_given_y2.ln_det_Sigma[idx]) 
    



class TestConditionalIdentityDiagGaussianPDF(TestConditionalIdentityGaussianPDF):
    @classmethod
    def create_instance(self, R, D):
        Sigma_y = self.get_pd_matrix(R, D)
        mu = objax.random.normal((R, D))
        py = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu)
        Sigma_xy = self.get_diagonal_mat(R, D)
        M = jnp.tile(jnp.eye(D)[None], (R, 1, 1))
        b = jnp.zeros((R, D))
        px_given_y1 = conditional.ConditionalGaussianPDF(M=M, b=b, Sigma=Sigma_xy)
        px_given_y2 = conditional.ConditionalIdentityDiagGaussianPDF(Sigma=Sigma_xy)
        # py = pxy.get_marginal(dim_y)
        return px_given_y1, px_given_y2, py

    @staticmethod
    def get_diagonal_mat(R, D):
        diag_mat = jnp.tile(jnp.diag(objax.random.uniform((D,)))[None], (R, 1, 1))
        return diag_mat
    
    
class TestclassConditionalGaussianDiagPDF(TestConditionalIdentityGaussianPDF):
    @classmethod
    def create_instance(self, R, D):
        Sigma_y = self.get_pd_matrix(R, D)
        mu = objax.random.normal((R, D))
        py = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu)
        Sigma_xy = self.get_diagonal_mat(R, D)
        M = objax.random.normal((R, D, D))
        b = objax.random.normal((R, D))
        px_given_y1 = conditional.ConditionalGaussianPDF(M=M, b=b, Sigma=Sigma_xy)
        px_given_y2 = conditional.ConditionalGaussianDiagPDF(M=M, b=b, Sigma=Sigma_xy)
        # py = pxy.get_marginal(dim_y)
        return px_given_y1, px_given_y2, py
    
    @staticmethod
    def get_diagonal_mat(R, D):
        diag_mat = jnp.tile(jnp.diag(objax.random.uniform((D,)))[None], (R, 1, 1))
        return diag_mat
    
