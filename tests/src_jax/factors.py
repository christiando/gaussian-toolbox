import sys

sys.path.append("../../")
from src_jax import factors
import pytest
from jax import numpy as jnp
import numpy as np


class TestConjugateFactor:
    
    def setup_class(self):
        self.test_class = factors.ConjugateFactor
        
    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(Lambda, nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        nu = None
        ln_beta = jnp.zeros(R)
        f = self.test_class(Lambda, nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        nu = jnp.zeros((R, D))
        ln_beta = None
        f = self.test_class(Lambda, nu, ln_beta)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
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
        "R, D, N", [(100, 5, 10), (1, 5, 10), (100, 1, 10), (100, 5, 1)]
    )
    def test_ln_evaluate(self, R, D, N):
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(Lambda, nu, ln_beta)
        x = jnp.asarray(np.random.randn(N, D))
        ln_eval = f.evaluate_ln(x)
        assert ln_eval.shape == (R, N)
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", Lambda, x), x)
        x_nu = jnp.dot(x, nu.T).T
        ln_eval_test = -0.5 * x_Lambda_x + x_nu + ln_beta[:, None]
        assert jnp.alltrue(ln_eval == ln_eval_test)

    @pytest.mark.parametrize(
        "R, D, N", [(100, 5, 10), (1, 5, 10), (100, 1, 10), (100, 5, 1)]
    )
    def test_evaluate(self, R, D, N):
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(Lambda, nu, ln_beta)
        x = jnp.asarray(np.random.randn(N, D))
        ln_eval = f.evaluate(x)
        assert ln_eval.shape == (R, N)
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", Lambda, x), x)
        x_nu = jnp.dot(x, nu.T).T
        eval_test = jnp.exp(-0.5 * x_Lambda_x + x_nu + ln_beta[:, None])
        assert jnp.alltrue(ln_eval == eval_test)

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
        Lambda = jnp.zeros((R, D, D))
        Lambda = Lambda.at[:].set(jnp.eye(D))
        Lambda = Lambda.at[idx].set(
            np.random.rand(len(idx))[:, None, None] * jnp.eye(D)[None]
        )
        nu = jnp.zeros((R, D))
        nu = nu.at[idx].set(np.random.randn(len(idx), D))
        ln_beta = jnp.zeros(R)
        ln_beta = ln_beta.at[idx].set(np.random.randn(len(idx),))
        f = self.test_class(Lambda, nu, ln_beta)
        f_new = f.slice(idx)
        assert jnp.alltrue(f_new.Lambda == f.Lambda[idx])
        assert jnp.alltrue(f_new.nu == f.nu[idx])
        assert jnp.alltrue(f_new.ln_beta == f.ln_beta[idx])


class TestOneRankFactor(TestConjugateFactor):
    
    def setup_class(self):
        self.test_class = factors.OneRankFactor
        
    @pytest.mark.parametrize("R, D", [(100, 5), (1, 5), (100, 1)])
    def test_init(self, R, D):
        v = jnp.ones((R, D,))
        g = jnp.ones((R,))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = jnp.ones((R, D,))
        g = None
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = jnp.ones((R, D,))
        g = jnp.ones((R,))
        nu = None
        ln_beta = jnp.zeros(R)
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = jnp.ones((R, D,))
        g = jnp.ones((R,))
        nu = jnp.zeros((R, D))
        ln_beta = None
        f = self.test_class(v, g, nu, ln_beta)
        assert f.v.shape == (f.R, f.D)
        assert f.g.shape == (f.R,)
        assert f.Lambda.shape == (f.R, f.D, f.D)
        assert f.nu.shape == (f.R, f.D)
        assert f.ln_beta.shape == (f.R,)
        v = jnp.ones((R, D,))
        g = jnp.ones((R,))
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
        v = jnp.ones((R, D,))
        g = jnp.ones((R,))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(v, g, nu, ln_beta)
        x = jnp.asarray(np.random.randn(N, D))
        ln_eval = f.evaluate_ln(x)
        assert ln_eval.shape == (R, N)
        Lambda = jnp.einsum("ab,ac->abc", v, jnp.einsum("a,ab->ab", g, v))
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", Lambda, x), x)
        x_nu = jnp.dot(x, nu.T).T
        ln_eval_test = -0.5 * x_Lambda_x + x_nu + ln_beta[:, None]
        assert jnp.alltrue(ln_eval == ln_eval_test)

    @pytest.mark.parametrize(
        "R, D, N", [(100, 5, 10), (1, 5, 10), (100, 1, 10), (100, 5, 1)]
    )
    def test_evaluate(self, R, D, N):
        v = jnp.ones((R, D,))
        g = jnp.ones((R,))
        nu = jnp.zeros((R, D))
        ln_beta = jnp.zeros(R)
        f = self.test_class(v, g, nu, ln_beta)
        x = jnp.asarray(np.random.randn(N, D))
        evaluate = f.evaluate(x)
        assert evaluate.shape == (R, N)
        Lambda = jnp.einsum("ab,ac->abc", v, jnp.einsum("a,ab->ab", g, v))
        x_Lambda_x = jnp.einsum("adc,dc->ad", jnp.einsum("abc,dc->adb", Lambda, x), x)
        x_nu = jnp.dot(x, nu.T).T
        eval_test = jnp.exp(-0.5 * x_Lambda_x + x_nu + ln_beta[:, None])
        assert jnp.alltrue(evaluate == eval_test)

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
        v = jnp.ones((R, D,))
        v = v.at[idx].set(np.random.randn(len(idx), D))
        g = jnp.ones((R,))
        g = g.at[idx].set(np.random.randn(len(idx),))
        nu = jnp.zeros((R, D))
        nu = nu.at[idx].set(np.random.randn(len(idx), D))
        ln_beta = jnp.zeros(R)
        ln_beta = ln_beta.at[idx].set(np.random.randn(len(idx),))
        f = self.test_class(v, g, nu, ln_beta)
        f_new = f.slice(idx)
        assert jnp.alltrue(f_new.v == f.v[idx])
        assert jnp.alltrue(f_new.g == f.g[idx])
        assert jnp.alltrue(f_new.Lambda == f.Lambda[idx])
        assert jnp.alltrue(f_new.nu == f.nu[idx])
        assert jnp.alltrue(f_new.ln_beta == f.ln_beta[idx])

