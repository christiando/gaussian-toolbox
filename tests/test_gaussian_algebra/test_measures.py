from gaussian_toolbox.gaussian_algebra import measures, factors
from gaussian_toolbox.utils import linalg

from test_factors import (
    TestConjugateFactor,
    TestOneRankFactor,
    TestLinearFactor,
    TestConstantFactor,
)

import pytest
from jax import numpy as jnp
from jax import scipy as jsc
from jax import config
from jax import jit

config.update("jax_enable_x64", True)
import numpy as np
import objax
from scipy import integrate as sc_integrate


class TestGaussianMeasure(TestConjugateFactor):
    def setup_class(self):
        self.test_class = measures.GaussianMeasure

    @classmethod
    def create_instance(self, R, D):
        Lambda = self.get_pd_matrix(R, D)
        nu = objax.random.normal((R, D))
        ln_beta = objax.random.normal((R,))
        return measures.GaussianMeasure(Lambda, nu, ln_beta)

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):

        m = self.create_instance(R, D)
        assert m.Lambda.shape == (m.R, m.D, m.D)
        assert m.Sigma == None
        assert m.mu == None
        assert m.ln_det_Lambda == None
        assert m.ln_det_Sigma == None
        assert m.nu.shape == (m.R, m.D)
        assert m.ln_beta.shape == (m.R,)
        Lambda = self.get_pd_matrix(R, D)
        nu = None
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        assert m.Lambda.shape == (m.R, m.D, m.D)
        assert m.nu.shape == (m.R, m.D)
        assert m.ln_beta.shape == (m.R,)
        assert m.Sigma == None
        assert m.mu == None
        assert m.ln_det_Lambda == None
        assert m.ln_det_Sigma == None
        Lambda = self.get_pd_matrix(R, D)
        nu = jnp.zeros((R, D))
        ln_beta = None
        m = self.test_class(Lambda, nu, ln_beta)
        assert m.Lambda.shape == (m.R, m.D, m.D)
        assert m.nu.shape == (m.R, m.D)
        assert m.ln_beta.shape == (m.R,)
        assert m.Sigma == None
        assert m.mu == None
        assert m.ln_det_Lambda == None
        assert m.ln_det_Sigma == None
        Lambda = self.get_pd_matrix(R, D)
        nu = None
        ln_beta = None
        m = self.test_class(Lambda, nu, ln_beta)
        assert m.Lambda.shape == (m.R, m.D, m.D)
        assert m.nu.shape == (m.R, m.D)
        assert m.ln_beta.shape == (m.R,)
        assert m.Sigma == None
        assert m.mu == None
        assert m.ln_det_Lambda == None
        assert m.ln_det_Sigma == None
        Lambda = None
        nu = None
        ln_beta = None
        with pytest.raises(AttributeError):
            m = self.test_class(Lambda, nu, ln_beta)

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
        super().test_slice(R, D, idx)
        m = self.create_instance(R, D)
        m._prepare_integration()
        m_new = m.slice(idx)
        assert jnp.alltrue(m_new.Lambda == m.Lambda[idx])
        assert jnp.alltrue(m_new.nu == m.nu[idx])
        assert jnp.alltrue(m_new.ln_beta == m.ln_beta[idx])
        assert jnp.alltrue(m_new.Sigma == m.Sigma[idx])
        assert jnp.alltrue(m_new.ln_det_Lambda == m.ln_det_Lambda[idx])
        assert jnp.alltrue(m_new.ln_det_Sigma == m.ln_det_Sigma[idx])

    @pytest.mark.parametrize("R, D", [(11, 5), (1, 5), (13, 1), (5, 5)])
    def test_product(self, R, D):
        super().test_product(R, D)
        m = self.create_instance(R, D)
        m._prepare_integration()
        m_prod = m.product()
        Lambda_new = jnp.sum(m.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(m.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(m.ln_beta, axis=0, keepdims=True)
        Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
        assert jnp.allclose(Lambda_new, m_prod.Lambda)
        assert jnp.allclose(nu_new, m_prod.nu)
        assert jnp.allclose(ln_beta_new, m_prod.ln_beta)
        assert jnp.allclose(Sigma_new, m_prod.Sigma)
        assert jnp.allclose(ln_det_Lambda_new, m_prod.ln_det_Lambda)
        assert jnp.allclose(-ln_det_Lambda_new, m_prod.ln_det_Sigma)

    @pytest.mark.parametrize(
        "R, D", [(2, 5,), (1, 5,), (6, 1,), (7, 5,),],
    )
    def test_compute_lnZ(
        self, R, D,
    ):
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D) * np.random.rand(R)[:, None, None])
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        m.compute_lnZ()
        A = Lambda
        L = jsc.linalg.cho_factor(A)
        Sigma = jsc.linalg.cho_solve(
            L, jnp.tile(jnp.eye(A.shape[1])[None], (len(A), 1, 1))
        )
        ln_det_Lambda = 2.0 * jnp.sum(
            jnp.log(L[0].diagonal(axis1=-1, axis2=-2)), axis=1
        )
        ln_det_Sigma = -ln_det_Lambda
        nu_Lambda_nu = jnp.einsum("ab,ab->a", nu, jnp.einsum("abc,ac->ab", Sigma, nu))
        lnZ = 0.5 * (nu_Lambda_nu + D * jnp.log(2.0 * jnp.pi) + ln_det_Sigma)
        assert jnp.allclose(lnZ, m.lnZ)

    @pytest.mark.parametrize(
        "R1, R2, D, test_class2",
        [
            (10, 7, 5, TestConjugateFactor),
            (1, 4, 5, TestOneRankFactor),
            (12, 5, 1, TestLinearFactor),
            (11, 1, 10, TestConstantFactor),
            (11, 7, 10, None),
        ],
    )
    def test_multiply(self, R1, R2, D, test_class2):
        m1 = self.create_instance(R1, D)
        if test_class2 is None:
            m2 = self.create_instance(R2, D)
        else:
            m2 = test_class2.create_instance(R2, D)
        m12 = m1 * m2
        assert m12.R == m1.R * m2.R
        assert m12.D == m1.D == m2.D
        Lambda_new = jnp.reshape(
            (m1.Lambda[:, None] + m2.Lambda[None]), (m1.R * m2.R, m2.D, m2.D),
        )
        nu_new = jnp.reshape((m1.nu[:, None] + m2.nu[None]), (m1.R * m2.R, m2.D))
        ln_beta_new = jnp.reshape(
            (m1.ln_beta[:, None] + m2.ln_beta[None]), (m1.R * m2.R)
        )
        assert jnp.allclose(m12.Lambda, Lambda_new)
        assert jnp.allclose(m12.nu, nu_new)
        assert jnp.allclose(m12.ln_beta, ln_beta_new)
        m1 = self.create_instance(R1, D)
        Lambda_new = jnp.reshape(
            (m1.Lambda[:, None] + m2.Lambda[None]), (m1.R * m2.R, m2.D, m2.D),
        )
        nu_new = jnp.reshape((m1.nu[:, None] + m2.nu[None]), (m1.R * m2.R, m2.D))
        ln_beta_new = jnp.reshape(
            (m1.ln_beta[:, None] + m2.ln_beta[None]), (m1.R * m2.R)
        )
        m12 = m1.multiply(m2, update_full=True)
        # print(jnp.einsum('abc,acd->abd', Lambda_new, m12.Sigma))
        assert jnp.allclose(
            jnp.einsum("abc,acd->abd", Lambda_new, m12.Sigma),
            jnp.eye(D)[None],
            atol=1e-5,
        )
        assert jnp.allclose(m12.ln_det_Sigma, -m12.ln_det_Lambda)
        m1 = self.create_instance(R1, D)
        m1._prepare_integration()
        Lambda_new = jnp.reshape(
            (m1.Lambda[:, None] + m2.Lambda[None]), (m1.R * m2.R, m2.D, m2.D),
        )
        nu_new = jnp.reshape((m1.nu[:, None] + m2.nu[None]), (m1.R * m2.R, m2.D))
        ln_beta_new = jnp.reshape(
            (m1.ln_beta[:, None] + m2.ln_beta[None]), (m1.R * m2.R)
        )
        m12 = m1.multiply(m2, update_full=True)
        # print(jnp.einsum('abc,acd->abd', Lambda_new, m12.Sigma))
        assert jnp.allclose(
            jnp.einsum("abc,acd->abd", Lambda_new, m12.Sigma),
            jnp.eye(D)[None],
            atol=1e-5,
        )
        assert jnp.allclose(m12.ln_det_Sigma, -m12.ln_det_Lambda)

    @pytest.mark.parametrize(
        "R1, R2, D, test_class2",
        [
            (5, 5, 5, TestConjugateFactor),
            (1, 4, 5, TestConjugateFactor),
            (4, 4, 1, TestOneRankFactor),
            (7, 7, 1, TestConstantFactor),
            (10, 1, 3, TestLinearFactor),
            (4, 1, 6, None),
        ],
    )
    def test_hadamard(self, R1, R2, D, test_class2):
        m1 = self.create_instance(R1, D)
        if test_class2 is None:
            m2 = self.create_instance(R2, D)
        else:
            m2 = test_class2.create_instance(R2, D)
        m12 = m1.hadamard(m2)
        assert m12.R == jnp.amax(jnp.array([R1, R2]))
        assert m12.D == m1.D == m2.D
        Lambda_new = m1.Lambda + m2.Lambda
        nu_new = m1.nu + m2.nu
        ln_beta_new = m1.ln_beta + m2.ln_beta
        assert jnp.allclose(m12.Lambda, Lambda_new)
        assert jnp.allclose(m12.nu, nu_new)
        assert jnp.allclose(m12.ln_beta, ln_beta_new)
        m1 = self.create_instance(R1, D)
        Lambda_new = m1.Lambda + m2.Lambda
        nu_new = m1.nu + m2.nu
        ln_beta_new = m1.ln_beta + m2.ln_beta
        m12 = m1.hadamard(m2, update_full=True)
        assert jnp.allclose(
            jnp.einsum("abc,acd->abd", Lambda_new, m12.Sigma),
            jnp.eye(D)[None],
            atol=1e-5,
        )
        m1 = self.create_instance(R1, D)
        m1.invert_lambda()
        Lambda_new = m1.Lambda + m2.Lambda
        nu_new = m1.nu + m2.nu
        ln_beta_new = m1.ln_beta + m2.ln_beta
        m12 = m1.hadamard(m2, update_full=True)
        assert jnp.allclose(
            jnp.einsum("abc,acd->abd", Lambda_new, m12.Sigma),
            jnp.eye(D)[None],
            atol=1e-5,
        )
        assert jnp.allclose(m12.ln_det_Sigma, -m12.ln_det_Lambda)

    @pytest.mark.parametrize(
        "R, D", [(2, 2,), (1, 1,), (2, 1,), (1, 2,),],
    )
    def test_integrate1(self, R, D):
        np.random.seed(1)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D) * (np.random.rand(R, D) + 1)[:, :, None])
        nu = jnp.array(np.random.randn(R, D))
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        Sigma_diag = 1.0 / jnp.diagonal(Lambda, axis1=1, axis2=2)
        mu = Sigma_diag * nu
        normalization = jnp.exp(
            jnp.sum(
                0.5 * jnp.log(2.0 * jnp.pi * Sigma_diag) + 0.5 * mu ** 2 / Sigma_diag,
                axis=1,
            )
        )
        integral_analytic_light = m.integral_light()
        integral_analytic = m.integrate("1")
        assert np.allclose(normalization, integral_analytic)
        assert np.allclose(normalization, integral_analytic_light)

    @pytest.mark.parametrize(
        "R, D", [(2, 2,), (1, 1,), (2, 1,), (1, 2,),],
    )
    def test_integratex(self, R, D):
        np.random.seed(1)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D) * (np.random.rand(R, D) + 1)[:, :, None])
        nu = jnp.array(np.random.randn(R, D))
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        Sigma_diag = 1.0 / jnp.diagonal(Lambda, axis1=1, axis2=2)
        mu = Sigma_diag * nu
        normalization = jnp.exp(
            jnp.sum(
                0.5 * jnp.log(2.0 * jnp.pi * Sigma_diag) + 0.5 * mu ** 2 / Sigma_diag,
                axis=1,
            )
        )
        integral_analytic = m.integrate("x")
        assert np.allclose(normalization[:, None] * mu, integral_analytic)

    @pytest.mark.parametrize(
        "R, D", [(10, 2,), (1, 1,), (10, 1,), (1, 2,),],
    )
    def test_integrate_Ax_a(self, R, D):
        np.random.seed(1)
        A, a = jnp.asarray(np.random.rand(D, D)), jnp.asarray(np.random.rand(D,))
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D) * (np.random.rand(R, D) + 1)[:, :, None])
        nu = jnp.array(np.random.randn(R, D))
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        Sigma_diag = 1.0 / jnp.diagonal(Lambda, axis1=1, axis2=2)
        mu = Sigma_diag * nu
        normalization = jnp.exp(
            jnp.sum(
                0.5 * jnp.log(2.0 * jnp.pi * Sigma_diag) + 0.5 * mu ** 2 / Sigma_diag,
                axis=1,
            )
        )
        integral_analytic = m.integrate("(Ax+a)", A_mat=A, a_vec=a)
        integral = normalization[:, None] * (jnp.einsum("ab,cb->ca", A, mu) + a)
        assert np.allclose(integral, integral_analytic, atol=1e-2)

    @pytest.mark.parametrize(
        "R, D", [(10, 2,), (1, 1,), (10, 1,), (10, 2,),],
    )
    def test_integrate_xx(self, R, D):
        np.random.seed(1)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D) * (np.random.rand(R, D) + 1)[:, :, None])
        nu = jnp.array(np.random.randn(R, D))
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        Sigma_diag = 1.0 / jnp.diagonal(Lambda, axis1=1, axis2=2)
        mu = Sigma_diag * nu
        normalization = jnp.exp(
            jnp.sum(
                0.5 * jnp.log(2.0 * jnp.pi * Sigma_diag) + 0.5 * mu ** 2 / Sigma_diag,
                axis=1,
            )
        )
        integral_analytic = m.integrate("xx'")
        integral = normalization[:, None, None] * (
            jnp.linalg.inv(Lambda) + jnp.einsum("ab,ac->abc", mu, mu)
        )
        assert np.allclose(integral, integral_analytic)

    @pytest.mark.parametrize(
        "R, D, k, l", [(1, 2, 2, 5), (2, 2, 3, 1), (1, 1, 2, 5)],
    )
    def test_integrate_general_quadratic_outer(self, R, D, k, l):
        # Generate matrices & vectors
        A_mat, a_vec = (
            jnp.array(np.random.rand(k, D) / k / D),
            jnp.array(np.random.rand(k) / k),
        )
        B_mat, b_vec = (
            jnp.array(np.random.rand(l, D) / l / D),
            jnp.array(np.random.rand(l) / l),
        )
        # Integrate with respect to Gaussian density
        m = self.create_instance(R, D)
        integral_analytic = m.integrate(
            "(Ax+a)(Bx+b)'", A_mat=A_mat, a_vec=a_vec, B_mat=B_mat, b_vec=b_vec
        )
        d = m.get_density()
        x_sample = d.sample(1000000)
        sample = jnp.einsum("ab,cdb->cda", A_mat, x_sample) + a_vec
        Bx_b_sample = jnp.einsum("ab,cdb->cda", B_mat, x_sample) + b_vec
        integral_sample = (
            jnp.mean(jnp.einsum("abc,abd->abcd", sample, Bx_b_sample), axis=0)
        ) * m.integrate()[:, None, None]
        assert jnp.allclose(integral_sample, integral_analytic, atol=1e-2, rtol=np.inf)

    @pytest.mark.parametrize(
        "R", [1, 2, 3],
    )
    def test_integrate_general_cubic(self, R):
        D, k, l = 1, 1, 1
        # Generate matrices & vectors
        A_mat, a_vec = (
            jnp.array(np.random.rand(k, D)),
            jnp.array(np.random.rand(k)),
        )
        B_mat, b_vec = (
            jnp.array(np.random.rand(k, D)),
            jnp.array(np.random.rand(k)),
        )
        C_mat, c_vec = (
            jnp.array(np.random.rand(l, D)),
            jnp.array(np.random.rand(l)),
        )
        # Integrate with respect to Gaussian density
        m = self.create_instance(R, D)
        integral_analytic_outer = m.integrate(
            "(Ax+a)'(Bx+b)(Cx+c)'",
            A_mat=A_mat,
            a_vec=a_vec,
            B_mat=B_mat,
            b_vec=b_vec,
            C_mat=C_mat,
            c_vec=c_vec,
        )
        integral_analytic_inner = m.integrate(
            "(Ax+a)(Bx+b)'(Cx+c)",
            A_mat=A_mat,
            a_vec=a_vec,
            B_mat=B_mat,
            b_vec=b_vec,
            C_mat=C_mat,
            c_vec=c_vec,
        )
        for r in range(R):

            @jit
            def func(x):
                x_vec = jnp.array([x,]).T
                Ax_a = A_mat * x_vec + a_vec
                Bx_b = B_mat * x_vec + b_vec
                Cx_c = C_mat * x_vec + c_vec
                integral = Ax_a * Bx_b * Cx_c * m(x_vec).T
                return integral[:, r]

            d = m.get_density()
            mu, var = d.mu, d.Sigma
            integral_num = sc_integrate.quadrature(
                func,
                mu[r, 0] - 10 * jnp.sqrt(var[r, 0, 0]),
                mu[r, 0] + 10 * jnp.sqrt(var[r, 0, 0]),
            )[0]

        assert jnp.allclose(integral_num, integral_analytic_outer[r], atol=1e-2)
        assert jnp.allclose(integral_num, integral_analytic_inner[r], atol=1e-2)

    @pytest.mark.parametrize(
        "R", [1, 2, 3],
    )
    def test_integrate_general_quartic(self, R):
        D, k, l = 1, 1, 1
        # Generate matrices & vectors
        A_mat, a_vec = (
            jnp.array(np.random.rand(k, D)),
            jnp.array(np.random.rand(k)),
        )
        B_mat, b_vec = (
            jnp.array(np.random.rand(k, D)),
            jnp.array(np.random.rand(k)),
        )
        C_mat, c_vec = (
            jnp.array(np.random.rand(l, D)),
            jnp.array(np.random.rand(l)),
        )
        D_mat, d_vec = (
            jnp.array(np.random.rand(l, D)),
            jnp.array(np.random.rand(l)),
        )
        # Integrate with respect to Gaussian density
        m = self.create_instance(R, D)
        integral_analytic_outer = m.integrate(
            "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
            A_mat=A_mat,
            a_vec=a_vec,
            B_mat=B_mat,
            b_vec=b_vec,
            C_mat=C_mat,
            c_vec=c_vec,
            D_mat=D_mat,
            d_vec=d_vec,
        )
        integral_analytic_inner = m.integrate(
            "(Ax+a)(Bx+b)'(Cx+c)(Dx+d)'",
            A_mat=A_mat,
            a_vec=a_vec,
            B_mat=B_mat,
            b_vec=b_vec,
            C_mat=C_mat,
            c_vec=c_vec,
            D_mat=D_mat,
            d_vec=d_vec,
        )
        for r in range(R):

            @jit
            def func(x):
                x_vec = jnp.array([x,]).T
                Ax_a = A_mat * x_vec + a_vec
                Bx_b = B_mat * x_vec + b_vec
                Cx_c = C_mat * x_vec + c_vec
                Dx_d = D_mat * x_vec + d_vec
                integral = Ax_a * Bx_b * Cx_c * Dx_d * m(x_vec).T
                return integral[:, r]

            d = m.get_density()
            mu, var = d.mu, d.Sigma
            integral_num = sc_integrate.quadrature(
                func,
                mu[r, 0] - 10 * jnp.sqrt(var[r, 0, 0]),
                mu[r, 0] + 10 * jnp.sqrt(var[r, 0, 0]),
            )[0]

        assert jnp.allclose(integral_num, integral_analytic_outer[r], atol=1e-2)
        assert jnp.allclose(integral_num, integral_analytic_inner[r], atol=1e-2)

    @pytest.mark.parametrize(
        "R", [2, 1,],
    )
    def test_integrate_cubic_outer(self, R):
        D = 1
        A_mat, a_vec = (
            jnp.array(np.random.rand(1, D)),
            jnp.array(np.random.rand(1)),
        )
        # Integrate with respect to Gaussian density
        m = self.create_instance(R, D)
        r1 = m.integrate("x(A'x + a)x'", A_mat=A_mat, a_vec=a_vec)
        r2 = m.integrate("(Ax+a)'(Bx+b)(Cx+c)'", B_mat=A_mat, b_vec=a_vec)
        assert jnp.allclose(r1[:, :, 0], r2,)

    @pytest.mark.parametrize(
        "R, D", [(2, 2,), (1, 1,), (2, 1,), (1, 2,),],
    )
    def test_integrate_xbxx(self, R, D):
        D = 1
        A_mat = jnp.array(np.random.rand(1, D))
        m = self.create_instance(R, D)
        r1 = m.integrate("x(A'x + a)x'", A_mat=A_mat)
        r2 = m.integrate("xb'xx'", b_vec=A_mat[0])
        assert jnp.allclose(r1, r2)

    @pytest.mark.parametrize(
        "R, D", [(2, 2,), (1, 1,), (2, 1,), (1, 2,),],
    )
    def test_integrate_log_factor(self, R, D):
        m = self.create_instance(R, D)
        Lambda_u = jnp.zeros((R, D, D))
        Lambda_u = Lambda_u.at[:].set(jnp.eye(D) * (np.random.rand(R, D))[:, :, None])
        nu_u = jnp.array(1e-4 * np.random.randn(R, D))
        ln_beta_u = jnp.array(np.random.randn(R))
        u = factors.ConjugateFactor(Lambda_u, nu_u, ln_beta_u)
        r_ana = m.integrate("log u(x)", factor=u)
        if D == 1:
            for r in range(R):

                def func(x):
                    x_vec = jnp.array([x,]).T
                    integral = u.evaluate_ln(x_vec).T * m(x_vec).T
                    return integral[:, r]

                d = m.get_density()
                mu, var = d.mu, d.Sigma
                r_num = sc_integrate.quadrature(
                    func,
                    mu[r, 0] - 10 * jnp.sqrt(var[r, 0, 0]),
                    mu[r, 0] + 10 * jnp.sqrt(var[r, 0, 0]),
                )[0]
                assert np.allclose(r_num, r_ana[r], atol=1e-2)
        else:
            p = m.get_density()
            x_sample = p.sample(1000000)
            int_m = m.integrate()
            r_sample = []
            for ridx in range(R):
                r_sample.append(
                    jnp.mean(
                        u.slice(jnp.array([ridx])).evaluate_ln(x_sample[:, ridx])[0],
                        axis=0,
                    )
                )
            r_sample = jnp.array(r_sample) * int_m
            assert np.allclose(r_sample, r_ana, atol=1e-2, rtol=np.inf)


class TestGaussianDiagMeasure(TestGaussianMeasure):
    def setup_class(self):
        self.test_class = measures.GaussianDiagMeasure

    @classmethod
    def create_instance(self, R, D):
        Lambda = jnp.tile(
            (1.0 / objax.random.uniform((D, D)) * jnp.eye(D))[None], [R, 1, 1]
        )
        nu = objax.random.normal((R, D))
        ln_beta = objax.random.normal((R,))
        return measures.GaussianDiagMeasure(Lambda, nu, ln_beta)
