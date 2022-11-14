
import pytest
import numpy as np
from jax import numpy as jnp
from gaussian_toolbox.gaussian_process import kernel



class TestKernel:
    def setup_class(self):
        self.kernel = kernel.RBF()
        
    @pytest.mark.parametrize(
        "NA, NB, D",
        [
         (1, 100, 5), (10, 90, 5), (50, 50, 5), (90, 10, 5), (100, 1, 5), (100, 0, 5),       
        ],
    )
    def test_evaluate(self, NA, NB, D):
        np.random.seed(0)
        XA = np.random.randn(NA, D)       
        if NB != 0:
            check_ev = False
            XB = np.random.randn(NB, D)
        else:
            NB, XB, check_ev = NA, XA, True

        r = self.kernel.euclidean_distance(XA, XB)
        kernel = self.kernel.evaluate(XA, XB)
        
        # Check dimenstions and NaNs
        assert kernel.shape == (NA, NB) 
        assert jnp.any(jnp.isnan(kernel)) == False
        
        # Check eigenvalues (kernel must be positive semidefinite)
        if check_ev:
            assert np.all(np.linalg.eigvals(kernel) >= 0)
        
        # Compare with scipy implementation of the distance 
        from scipy.spatial.distance import cdist
        r_scipy = cdist(self.kernel.scale(XA), self.kernel.scale(XB),
                        metric='euclidean')
        assert jnp.allclose(r, r_scipy, atol=1e-5)

                   
class TestMatern12(TestKernel):
    def setup_class(self):
        self.kernel = kernel.Matern12()
        
class TestMatern32(TestKernel):
    def setup_class(self):
        self.kernel = kernel.Matern32()

class TestMatern52(TestKernel):
    def setup_class(self):
        self.kernel = kernel.Matern52()
        
class TestExponential(TestKernel):
    def setup_class(self):
        self.kernel = kernel.Exponential()

class TestRationalQuadratic(TestKernel):
    def setup_class(self):
        self.kernel = kernel.RationalQuadratic()
