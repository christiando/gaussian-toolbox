from gaussian_toolbox import pdf, measure
from gaussian_toolbox.experimental import truncated_measure
from scipy.stats import truncnorm
import pytest
from jax import numpy as jnp
import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
np.random.seed(0)

class TestTruncatedGaussianMeasure:
    
    def create_instance(self, R, lower, upper):
        D = 1
        Lambda = jnp.array(np.random.rand(R, D, D) + 1e-3)
        nu = jnp.array(np.random.rand(R, D))
        m = measure.GaussianMeasure(Lambda=Lambda, nu=nu)
        tm = truncated_measure.TruncatedGaussianMeasure(measure=m, lower_limit=lower, upper_limit=upper)
        return tm, m
    
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (10, None, None), (2, 0, 1), (2, 0, -1), (2, 0, None), (2, 0, 0), (2, None, 0)])
    def test_init(self, R, lower, upper):
        if upper is None:
            upper_num = jnp.inf
        else:
            upper_num = upper
        if lower is None:
            lower_num = -jnp.inf
        else:
            lower_num = lower
        if upper is None and lower is None:
            with pytest.raises(ValueError):
                tm, m = self.create_instance(R, lower, upper)
        elif lower_num >= upper_num:
            with pytest.raises(AssertionError):
                tm, m = self.create_instance(R, lower, upper)
        else:
            tm, m = self.create_instance(R, lower, upper)
            if lower is None:
                assert jnp.all(tm.lower_limit == -jnp.inf)
            elif upper is None:
                assert jnp.all(tm.upper_limit == jnp.inf)
            assert tm.R == R
            assert tm.D == 1
            assert tm.lower_limit.shape == (R, 1)
            assert tm.upper_limit.shape == (R, 1)
            
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_call(self, R, lower, upper):
        tm, m = self.create_instance(R, lower, upper)
        x = jnp.array(np.random.rand(R, 1))
        if lower is None:
            lower = -jnp.inf
        elif upper is None:
            upper = jnp.inf
        in_limits = jnp.logical_and(jnp.greater_equal(x, lower), jnp.less_equal(x, upper))[:,0]
        m_val = m(x, element_wise=True)
        tm_val = tm(x, element_wise=True)
        assert jnp.allclose(m_val[in_limits], tm_val[in_limits])
        assert jnp.all(tm_val[~in_limits] == 0)
        
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_integrate(self, R, lower, upper):
        for int_str in ["1", "x", "x**2"]:
            tm, m = self.create_instance(R, lower, upper)
            if int_str == "1":
                assert jnp.all(jnp.less_equal(tm.integrate(int_str), m.integrate(int_str)))
            if upper is not None:
                tm1 = truncated_measure.TruncatedGaussianMeasure(measure=m, lower_limit=None, upper_limit=upper)
                tm2 = truncated_measure.TruncatedGaussianMeasure(measure=m, lower_limit=upper, upper_limit=None)
                if int_str == "x**2":
                    assert jnp.allclose(tm1.integrate(int_str) + tm2.integrate(int_str), m.integrate("xx'")[:,0])
                else:
                    assert jnp.allclose(tm1.integrate(int_str) + tm2.integrate(int_str), m.integrate(int_str))
                    
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_get_density(self, R, lower, upper):
        tm, m = self.create_instance(R, lower, upper)
        tp = tm.get_density()
        assert jnp.allclose(tp.integrate(), 1)
        
class TestTruncatedGaussianPDF(TestTruncatedGaussianMeasure):
    
    def create_instance(self, R, lower, upper):
        D = 1
        Lambda = jnp.array(np.random.rand(R, D, D) + 1e-3)
        nu = jnp.array(np.random.rand(R, D))
        m = measure.GaussianMeasure(Lambda=Lambda, nu=nu)
        tm = truncated_measure.TruncatedGaussianPDF(measure=m, lower_limit=lower, upper_limit=upper)
        tp = tm.get_density()
        return tp, tp.density
    
    def _get_scipy_params(self, d, lower, upper):
        loc, scale = d.mu[:,0], jnp.sqrt(d.Sigma[:,0,0])
        a, b = (lower - loc) / scale, (upper - loc) / scale
        return loc, scale, a, b
    
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_call(self, R, lower, upper):
        tp, d = self.create_instance(R, lower, upper)
        x = jnp.array(np.random.rand(100, 1))
        if lower is None:
            lower = -jnp.inf
        elif upper is None:
            upper = jnp.inf
        in_limits = jnp.logical_and(jnp.greater_equal(x, lower), jnp.less_equal(x, upper))[:,0]
        d_val = d(x)
        tp_val = tp(x)
        assert jnp.all(jnp.less_equal(d_val[:,in_limits], tp_val[:,in_limits]))
        assert jnp.all(tp_val[:,~in_limits] == 0)
    
        loc, scale, a, b = self._get_scipy_params(d, lower, upper)
        for r in range(R):
            assert jnp.allclose(tp_val[r], truncnorm.pdf(x[:,0], a=a[r], b=b[r], loc=loc[r], scale=scale[r]), atol=1e-5)
            
    
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_mean(self, R, lower, upper):
        tp, d = self.create_instance(R, lower, upper)
        if lower is None:
            lower = -jnp.inf
        elif upper is None:
            upper = jnp.inf
        mu = tp.get_mean()
        loc, scale, a, b = self._get_scipy_params(d, lower, upper)
        for r in range(R):
            assert jnp.allclose(mu[r], truncnorm.mean(a=a[r], b=b[r], loc=loc[r], scale=scale[r]))
            
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_variance(self, R, lower, upper):
        tp, d = self.create_instance(R, lower, upper)
        if lower is None:
            lower = -jnp.inf
        elif upper is None:
            upper = jnp.inf
        sigma2 = tp.get_variance()
        sigma = tp.get_std()
        loc, scale, a, b = self._get_scipy_params(d, lower, upper)
        for r in range(R):
            assert jnp.allclose(sigma2[r], truncnorm.var(a=a[r], b=b[r], loc=loc[r], scale=scale[r]))
            assert jnp.allclose(sigma[r], truncnorm.std(a=a[r], b=b[r], loc=loc[r], scale=scale[r]))
            
    @pytest.mark.parametrize("R, lower, upper", [(1, -1, 2), (2, 0, 1), (10, 0, None)])
    def test_get_moments(self, R, lower, upper):
        tp, d = self.create_instance(R, lower, upper)
        if lower is None:
            lower = -jnp.inf
        elif upper is None:
            upper = jnp.inf
        loc, scale, a, b = self._get_scipy_params(d, lower, upper)
        for order in range(1, 5):
            moments = tp.integrate("x**k", k=order)
            for r in range(R):
                assert jnp.allclose(moments[r], truncnorm.moment(order , a=a[r], b=b[r], loc=loc[r], scale=scale[r]))