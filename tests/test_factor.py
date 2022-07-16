from gaussian_toolbox import factor, measure
import pytest
from jax import numpy as jnp
import numpy as np
import objax


class TestConjugateFactor:
    def setup_class(self):
        self.test_class = factor.ConjugateFactor

    @classmethod
    def create_instance(self, R, D):
        Lambda = self.get_pd_matrix(R, D)
        nu = objax.random.normal((R, D))
        ln_beta = objax.random.normal((R,))
        return factor.ConjugateFactor(Lambda, nu, ln_beta)

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

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        f = self.create_instance(R, D)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = self.get_pd_matrix(R, D)
        nu = None
        ln_beta = jnp.zeros(R)
        f = self.test_class(Lambda, nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = self.get_pd_matrix(R, D)
        nu = jnp.zeros((R, D))
        ln_beta = None
        f = self.test_class(Lambda, nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = self.get_pd_matrix(R, D)
        nu = None
        ln_beta = None
        f = self.test_class(Lambda, nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = None
        nu = None
        ln_beta = None
        with pytest.raises(AttributeError):
            f = self.test_class(Lambda, nu, ln_beta)

    @pytest.mark.parametrize(
        "R, D, N", [(100, 5, 10), (1, 5, 10), (100, 1, 10), (5, 5, 1)]
    )
    def test_ln_evaluate(self, R, D, N):
        f = self.create_instance(R, D)
        x = jnp.asarray(np.random.randn(N, D))
        ln_eval = f.evaluate_ln(x)
        assert ln_eval.shape == (R, N)
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", f.Lambda, x), x)
        x_nu = jnp.dot(x, f.nu.T).T
        ln_eval_test = -0.5 * x_Lambda_x + x_nu + f.ln_beta[:, None]
        assert jnp.alltrue(ln_eval == ln_eval_test)

    @pytest.mark.parametrize("R, D", [(11, 5), (1, 5), (13, 1), (5, 5)])
    def test_product(self, R, D):
        f = self.create_instance(R, D)
        Lambda_new = jnp.sum(f.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(f.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(f.ln_beta, axis=0, keepdims=True)
        f_prod = f.product()
        assert jnp.allclose(Lambda_new, f_prod.Lambda)
        assert jnp.allclose(nu_new, f_prod.nu)
        assert jnp.allclose(ln_beta_new, f_prod.ln_beta)

    @pytest.mark.parametrize(
        "R, D, N", [(11, 5, 10), (1, 1, 10), (10, 1, 10), (5, 5, 1)]
    )
    def test_evaluate(self, R, D, N):
        f = self.create_instance(R, D)
        x = jnp.asarray(np.random.randn(N, D))
        ln_eval = f(x)
        assert ln_eval.shape == (R, N)
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", f.Lambda, x), x)
        x_nu = jnp.dot(x, f.nu.T).T
        eval_test = jnp.exp(-0.5 * x_Lambda_x + x_nu + f.ln_beta[:, None])
        assert jnp.allclose(ln_eval, eval_test)
        if R == N:
            x_Lambda_x = jnp.einsum(
                "ab,ab->a", jnp.einsum("abc,ac->ab", f.Lambda, x), x
            )
            x_nu = jnp.sum(x * f.nu, axis=1)
            eval_test_ew = jnp.exp(-0.5 * x_Lambda_x + x_nu + f.ln_beta)
            ln_eval_ew = f(x, element_wise=True)
            assert jnp.allclose(ln_eval_ew, eval_test_ew)
        else:
            with pytest.raises(ValueError):
                ln_eval_ew = f(x, element_wise=True)

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
        f = self.create_instance(R, D)
        f_new = f.slice(idx)
        assert jnp.alltrue(f_new.Lambda == f.Lambda[idx])
        assert jnp.alltrue(f_new.nu == f.nu[idx])
        assert jnp.alltrue(f_new.ln_beta == f.ln_beta[idx])


class TestOneRankFactor(TestConjugateFactor):
    def setup_class(self):
        self.test_class = factor.OneRankFactor

    @classmethod
    def create_instance(self, R, D):
        v = objax.random.normal((R, D))
        g = jnp.abs(objax.random.normal((R,)))
        nu = objax.random.normal((R, D))
        ln_beta = objax.random.normal((R,))
        return factor.OneRankFactor(v, g, nu, ln_beta)

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        f = self.create_instance(R, D)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = objax.random.normal((R, D))
        g = None
        nu = objax.random.normal((R, D))
        ln_beta = objax.random.normal((R,))
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = objax.random.normal((R, D))
        g = jnp.abs(objax.random.normal((R,)))
        nu = None
        ln_beta = objax.random.normal((R,))
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = objax.random.normal((R, D))
        g = jnp.abs(objax.random.normal((R,)))
        nu = objax.random.normal((R, D))
        ln_beta = None
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = objax.random.normal((R, D))
        g = jnp.abs(objax.random.normal((R,)))
        nu = None
        ln_beta = None
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = None
        nu = None
        ln_beta = None
        with pytest.raises(AttributeError):
            f = self.test_class(v, g, nu, ln_beta)

    @pytest.mark.parametrize(
        "R, D, N", [(100, 5, 10), (1, 5, 10), (100, 1, 10), (100, 5, 1)]
    )
    def test_ln_evaluate(self, R, D, N):
        f = self.create_instance(R, D)
        x = jnp.asarray(np.random.randn(N, D))
        ln_eval = f.evaluate_ln(x)
        assert ln_eval.shape == (R, N)
        Lambda = jnp.einsum("ab,ac->abc", f.v, jnp.einsum("a,ab->ab", f.g, f.v))
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", Lambda, x), x)
        x_nu = jnp.dot(x, f.nu.T).T
        ln_eval_test = -0.5 * x_Lambda_x + x_nu + f.ln_beta[:, None]
        assert jnp.alltrue(ln_eval == ln_eval_test)

    @pytest.mark.parametrize(
        "R, D, N", [(100, 5, 10), (1, 5, 10), (100, 1, 10), (100, 5, 1)]
    )
    def test_evaluate(self, R, D, N):
        f = self.create_instance(R, D)
        x = jnp.asarray(np.random.randn(N, D))
        evaluate = f.evaluate(x)
        assert evaluate.shape == (R, N)
        Lambda = jnp.einsum("ab,ac->abc", f.v, jnp.einsum("a,ab->ab", f.g, f.v))
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", Lambda, x), x)
        x_nu = jnp.dot(x, f.nu.T).T
        eval_test = jnp.exp(-0.5 * x_Lambda_x + x_nu + f.ln_beta[:, None])
        assert jnp.alltrue(evaluate == eval_test)

    @pytest.mark.parametrize("R, D", [(11, 5), (1, 5), (13, 1), (5, 5)])
    def test_product(self, R, D):
        f = self.create_instance(R, D)
        f_prod = f.product()
        Lambda_new = jnp.sum(f.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(f.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(f.ln_beta, axis=0, keepdims=True)

        assert jnp.allclose(Lambda_new, f_prod.Lambda)
        assert jnp.allclose(nu_new, f_prod.nu)
        assert jnp.allclose(ln_beta_new, f_prod.ln_beta)

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
        f = self.create_instance(R, D)
        f_new = f.slice(idx)
        assert jnp.alltrue(f_new.v == f.v[idx])
        assert jnp.alltrue(f_new.g == f.g[idx])
        assert jnp.alltrue(f_new.Lambda == f.Lambda[idx])
        assert jnp.alltrue(f_new.nu == f.nu[idx])
        assert jnp.alltrue(f_new.ln_beta == f.ln_beta[idx])


class TestLinearFactor(TestConjugateFactor):
    def setup_class(self):
        self.test_class = factor.LinearFactor

    @classmethod
    def create_instance(self, R, D):
        nu = objax.random.normal((R, D))
        ln_beta = objax.random.normal((R,))
        return factor.LinearFactor(nu, ln_beta)

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        f = self.create_instance(R, D)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        nu = jnp.zeros((R, D))
        ln_beta = None
        f = self.test_class(nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        nu = None
        ln_beta = None
        with pytest.raises(AttributeError):
            f = self.test_class(nu, ln_beta)


class TestConstantFactor(TestConjugateFactor):
    def setup_class(self):
        self.test_class = factor.ConstantFactor

    @classmethod
    def create_instance(self, R, D):
        ln_beta = objax.random.normal((R,))
        return factor.ConstantFactor(ln_beta, D)

    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        f = self.create_instance(R, D)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)

