import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from test_factors import TestConjugateFactor
from src_jax import measures, factors, conditionals
import pytest
from jax import numpy as jnp
from jax import scipy as jsc
from jax import config

config.update("jax_enable_x64", True)
import numpy as np
import objax


class TestGaussianMeasure(TestConjugateFactor):
    def setup_class(self):
        self.test_class = measures.GaussianMeasure

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        m = self.test_class(Lambda, nu, ln_beta)
        assert m.Lambda.shape == (m.R, m.D, m.D)
        assert m.Sigma == None
        assert m.mu == None
        assert m.ln_det_Lambda == None
        assert m.ln_det_Sigma == None
        assert m.nu.shape == (m.R, m.D)
        assert m.ln_beta.shape == (m.R,)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
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
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
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
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
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

    @pytest.mark.parametrize(
        "R, D", [(100, 5,), (1, 5,), (100, 1,), (100, 5,),],
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
        assert jnp.alltrue(lnZ == m.lnZ)

    @pytest.mark.parametrize(
        "R1, R2, D", [(10, 50, 5,), (1, 50, 5,), (10, 5, 1,), (10, 1, 10),],
    )
    def test_multiply(self, R1, R2, D):
        Lambda1 = jnp.zeros((R1, D, D))
        Lambda1 = Lambda1.at[:].set(
            jnp.eye(D)[None] * np.random.rand(R1)[:, None, None]
        )
        nu1 = jnp.zeros((R1, D))
        ln_beta1 = jnp.zeros(R1)
        m1 = self.test_class(Lambda1, nu1, ln_beta1)
        Lambda2 = jnp.zeros((R2, D, D))
        Lambda2 = Lambda2.at[:].set(
            jnp.eye(D)[None] * np.random.rand(R2)[:, None, None]
        )
        nu2 = jnp.zeros((R2, D))
        ln_beta2 = jnp.zeros(R2)
        m2 = self.test_class(Lambda2, nu2, ln_beta2)
        m12 = m1.multiply(m2)
        assert m12.R == m1.R * m2.R
        assert m12.D == m1.D == m2.D
        Lambda_new = jnp.reshape(
            (m1.Lambda[:, None] + m2.Lambda[None]), (m1.R * m2.R, m2.D, m2.D),
        )
        nu_new = jnp.reshape((m1.nu[:, None] + m2.nu[None]), (m1.R * m2.R, m2.D))
        ln_beta_new = jnp.reshape(
            (m1.ln_beta[:, None] + m2.ln_beta[None]), (m1.R * m2.R)
        )
        assert jnp.alltrue(m12.Lambda == Lambda_new)
        assert jnp.alltrue(m12.nu == nu_new)
        assert jnp.alltrue(m12.ln_beta == ln_beta_new)
        m12 = m1.multiply(m2, update_full=True)
        # print(jnp.einsum('abc,acd->abd', Lambda_new, m12.Sigma))
        assert jnp.allclose(
            jnp.einsum("abc,acd->abd", Lambda_new, m12.Sigma), jnp.eye(D)[None]
        )
        assert jnp.alltrue(m12.ln_det_Sigma == -m12.ln_det_Lambda)

    @pytest.mark.parametrize(
        "R1, R2, D", [(50, 50, 5,), (1, 50, 5,), (10, 10, 1,), (10, 1, 10),],
    )
    def test_hadamard(self, R1, R2, D):
        Lambda1 = jnp.zeros((R1, D, D))
        Lambda1 = Lambda1.at[:].set(
            jnp.eye(D)[None] * np.random.rand(R1)[:, None, None]
        )
        nu1 = jnp.zeros((R1, D))
        ln_beta1 = jnp.zeros(R1)
        m1 = self.test_class(Lambda1, nu1, ln_beta1)
        Lambda2 = jnp.zeros((R2, D, D))
        Lambda2 = Lambda2.at[:].set(
            jnp.eye(D)[None] * np.random.rand(R2)[:, None, None]
        )
        nu2 = jnp.zeros((R2, D))
        ln_beta2 = jnp.zeros(R2)
        m2 = self.test_class(Lambda2, nu2, ln_beta2)
        m12 = m1.hadamard(m2)
        assert m12.R == jnp.amax(jnp.array([R1, R2]))
        assert m12.D == m1.D == m2.D
        Lambda_new = m1.Lambda + m2.Lambda
        nu_new = m1.nu + m2.nu
        ln_beta_new = m1.ln_beta + m2.ln_beta
        assert jnp.alltrue(m12.Lambda == Lambda_new)
        assert jnp.alltrue(m12.nu == nu_new)
        assert jnp.alltrue(m12.ln_beta == ln_beta_new)
        m12 = m1.hadamard(m2, update_full=True)
        assert jnp.allclose(
            jnp.einsum("abc,acd->abd", Lambda_new, m12.Sigma), jnp.eye(D)[None]
        )
        assert jnp.alltrue(m12.ln_det_Sigma == -m12.ln_det_Lambda)

    @pytest.mark.parametrize(
        "R, D", [(10, 5,), (1, 5,), (10, 1,), (10, 5,),],
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
        integral_analytic = m.integrate("1")
        assert np.allclose(normalization, integral_analytic)

    @pytest.mark.parametrize(
        "R, D", [(10, 5,), (1, 5,), (10, 1,), (10, 5,),],
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
        "R, D", [(10, 5,), (1, 5,), (10, 1,), (10, 5,),],
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
        integral_analytic = m.integrate("Ax_a", A_mat=A, a_vec=a)
        integral = normalization[:, None] * (jnp.einsum("ab,cb->ca", A, mu) + a)
        assert np.allclose(integral, integral_analytic, rtol=1e-4)

    @pytest.mark.parametrize(
        "R, D", [(10, 5,), (1, 5,), (10, 1,), (10, 5,),],
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
        integral_analytic = m.integrate("xx")
        integral = normalization[:, None, None] * (
            jnp.linalg.inv(Lambda) + jnp.einsum("ab,ac->abc", mu, mu)
        )
        assert np.allclose(integral, integral_analytic)

    @pytest.mark.parametrize(
        "R, D", [(10, 5,), (1, 5,), (10, 1,), (10, 5,),],
    )
    def test_integrate_log_factor(self, R, D):
        Lambda_m = jnp.zeros((R, D, D))
        Lambda_m = Lambda_m.at[:].set(
            jnp.eye(D) * (np.random.rand(R, D) + 1)[:, :, None]
        )
        nu_m = jnp.array(1e-4 * np.random.randn(R, D))
        ln_beta_m = jnp.array(np.random.randn(R))
        m = measures.GaussianMeasure(Lambda_m, nu_m, ln_beta_m)
        Lambda_u = jnp.zeros((R, D, D))
        Lambda_u = Lambda_u.at[:].set(jnp.eye(D) * (np.random.rand(R, D))[:, :, None])
        nu_u = jnp.array(1e-4 * np.random.randn(R, D))
        ln_beta_u = jnp.array(np.random.randn(R))
        u = factors.ConjugateFactor(Lambda_u, nu_u, ln_beta_u)
        p = m.get_density()
        x_sample = p.sample(1000000)
        int_m = m.integrate()
        r_sample = []
        for ridx in range(R):
            r_sample.append(
                jnp.mean(
                    u.slice(jnp.array([ridx])).evaluate_ln(x_sample[:, ridx])[0], axis=0
                )
            )
        r_sample = jnp.array(r_sample) * int_m
        r_ana = m.integrate("log_factor", factor=u)
        assert np.allclose(r_sample, r_ana, rtol=1e-2)

    @pytest.mark.parametrize(
        "R, Dx, Dy", [(2, 5, 2), (1, 5, 1), (10, 1, 6), (10, 5, 5),],
    )
    def test_integrate_log_conditional(self, R, Dx, Dy):
        D = Dx + Dy
        Lambda_m = jnp.zeros((R, D, D))
        Lambda_m = Lambda_m.at[:].set(jnp.eye(D) * (np.random.rand(R, D))[:, :, None])
        nu_m = jnp.array(np.random.randn(R, D))
        ln_beta_m = jnp.array(np.random.randn(R))
        m = measures.GaussianMeasure(Lambda_m, nu_m, ln_beta_m)
        M = objax.random.normal((1, Dy, Dx))
        b = objax.random.normal((1, Dy))
        Sigma = jnp.array([jnp.eye(Dy)]) + 0.01
        p_cond = conditionals.ConditionalGaussianDensity(M, b, Sigma)
        p = m.get_density()
        int_m = m.integrate()
        yx_sample = p.sample(1000000)
        r_sample = []
        for ridx in range(R):
            r_sample.append(
                jnp.mean(
                    p_cond(yx_sample[:, ridx, Dy:]).evaluate_ln(
                        yx_sample[:, ridx, :Dy], element_wise=True
                    ),
                    axis=0,
                )
            )
        r_sample = jnp.array(r_sample) * int_m
        r_ana = m.integrate("log_conditional", p_cond=p_cond)
        assert np.allclose(r_sample, r_ana, rtol=1e-2)

    @pytest.mark.parametrize(
        "R, Dx, Dy", [(2, 5, 2), (1, 5, 1), (10, 1, 6), (10, 5, 5),],
    )
    def test_integrate_log_conditional_y(self, R, Dx, Dy):
        Lambda_m = jnp.zeros((R, Dx, Dx))
        Lambda_m = Lambda_m.at[:].set(
            jnp.eye(Dx) * (np.random.rand(R, Dx) + 1)[:, :, None]
        )
        nu_m = jnp.array(np.random.randn(R, Dx))
        ln_beta_m = jnp.array(np.random.randn(R))
        m = measures.GaussianMeasure(Lambda_m, nu_m, ln_beta_m)
        M = objax.random.normal((1, Dy, Dx))
        b = objax.random.normal((1, Dy))
        Sigma = jnp.array([jnp.eye(Dy)]) + 0.01
        p_cond = conditionals.ConditionalGaussianDensity(M, b, Sigma)
        p = m.get_density()
        int_m = m.integrate()
        x_sample = p.sample(100000)
        y = objax.random.normal((8, Dy))
        r_sample = []
        for ridx in range(R):
            r_sample.append(jnp.mean(p_cond(x_sample[:, ridx]).evaluate_ln(y), axis=0))
        print(jnp.array(r_sample).T.shape, int_m.shape)
        r_sample = jnp.array(r_sample).T * int_m[None]
        print(r_sample.shape)
        r_ana = m.integrate("log_conditional_y", p_cond=p_cond, y=y)
        print(r_ana.shape)
        assert np.allclose(r_sample.flatten(), r_ana.flatten(), rtol=1e-2)

