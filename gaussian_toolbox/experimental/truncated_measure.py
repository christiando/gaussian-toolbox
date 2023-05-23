from jax import numpy as jnp
from jax.scipy.stats import norm
from ..utils.dataclass import dataclass
from dataclasses import field
from jaxtyping import Array, Float
from typing import Any, Callable, Tuple, Union, Dict
from gaussian_toolbox import pdf, measure


@dataclass(kw_only=True)
class TruncatedGaussianMeasure:
    measure: measure.GaussianMeasure
    lower_limit: Float[Array, "R 1"] = field(default=None)
    upper_limit: Float[Array, "R 1"] = field(default=None)
    density: pdf.GaussianPDF = field(init=False)
    constant: Float[Array, "R"] = field(init=False)
    alpha: Float[Array, "R 1"] = field(init=False)
    beta: Float[Array, "R 1"] = field(init=False)
    
    def __post_init__(self):
        self._check_limits()
        assert self.measure.D == 1, "TruncatedGaussianMeasure only supports 1D measures"
        self.density = self.measure.get_density()
        self.constant = self.measure.integrate()
        self.alpha = (self.lower_limit - self.density.mu) / self.density.Sigma[..., 0]
        self.beta = (self.upper_limit - self.density.mu) / self.density.Sigma[..., 0]
        
    def _check_limits(self):
        if self.lower_limit is None and self.upper_limit is None:
            raise ValueError("At least one of lower_limit and upper_limit must be specified.")
        elif self.lower_limit is None:
            self.lower_limit = -jnp.inf * jnp.ones((self.R, self.D))
        elif self.upper_limit is None:
            self.upper_limit = jnp.inf * jnp.ones((self.R, self.D))
            
        self.lower_limit = self.lower_limit * jnp.ones((self.R, self.D))
        self.upper_limit = self.upper_limit * jnp.ones((self.R, self.D))
        assert jnp.all(self.lower_limit < self.upper_limit), "lower_limit must be smaller than upper_limit"
        
    @property
    def R(self) -> int:
        """Number of factors (leading dimension)."""
        return self.measure.Lambda.shape[0]

    @property
    def D(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.measure.Lambda.shape[1]
        
    def __call__(self, x: Float[Array, "N 1"], element_wise: bool = False) -> Union[Float[Array, "R N"], Float[Array, "R"]]:
        if element_wise:
            print(x.shape)
            if self.R != x.shape[0]:
                raise ValueError("Leading dimension of x must equal R.")
            in_limits = jnp.all(jnp.logical_and(jnp.greater_equal(x, self.lower_limit), jnp.less_equal(x, self.upper_limit)), axis=-1)
        else:
            in_limits = jnp.all(jnp.logical_and(jnp.greater_equal(x[None], self.lower_limit[:,None]), 
                                                jnp.less_equal(x[None], self.upper_limit[:,None])), axis=-1)
        return self.measure(x, element_wise=element_wise) * in_limits
    
    def integrate(self, expr: str = "1", **kwargs) -> Float[Array, "R ..."]:
        r"""Integrate the indicated expression with respect to the Gaussian measure.

        E.g. expr="(Ax+a)" means that :math:`\int (AX + a)u(X){\rm d}X` is computed, and :math:`A` and a can be provided.

        :param expr: Indicates the expression that should be integrated. Check measure's integration dict.
        :return: The integral result.
        """
        return self.integration_dict[expr](**kwargs)
        
    @property
    def integration_dict(self) -> Dict:
        return {
            "1": self.integral,
            "x": self.integrate_x,
            "x^2": self.integrate_x_pow_2,
        }
        
    def _expectation_integral(self) -> Float[Array, "R"]:
        return jnp.squeeze(norm.cdf(self.beta) - norm.cdf(self.alpha), axis=-1)
    
    def integral(self) -> Float[Array, "R"]:
        return self._expectation_integral() * self.constant
    
    def _expectation_x(self) -> Float[Array, "R 1"]:
        Z = self._expectation_integral()
        return self.density.mu + (norm.pdf(self.alpha) - norm.pdf(self.beta)) / Z * self.density.Sigma[..., 0]
        
    def integrate_x(self) -> Float[Array, "R 1"]:
        return self._expectation_x() * self.constant
    
    def _expectation_x_pow_2(self) -> Float[Array, "R 1"]:
        Z = self._expectation_integral()
        mu = self._expectation_x()
        sigma2 = self.density.Sigma[..., 0] * (1 - 
                                              (self.beta * norm.pdf(self.beta) - self.alpha * norm.pdf(self.alpha)) / Z
                                              - (norm.pdf(self.alpha) - norm.pdf(self.beta))**2 / Z**2)
        expected_x_pow_2 = sigma2 + mu**2
        return expected_x_pow_2
    
    def integrate_x_pow_2(self) -> Float[Array, "R 1"]:
        return self._expectation_x_pow_2() * self.constant
    
    def get_density(self) -> "TruncatedGaussianPDF":
        return TruncatedGaussianPDF(measure=self.density, lower_limit=self.lower_limit, upper_limit=self.upper_limit)
    
    
@dataclass(kw_only=True)
class TruncatedGaussianPDF(TruncatedGaussianMeasure):
    measure: measure.GaussianMeasure
    lower_limit: Float[Array, "R 1"] = field(default=None)
    upper_limit: Float[Array, "R 1"] = field(default=None)
    density: pdf.GaussianPDF = field(init=False)
    constant: Float[Array, "R"] = field(init=False)
    alpha: Float[Array, "R 1"] = field(init=False)
    beta: Float[Array, "R 1"] = field(init=False)
    
    def __post_init__(self):
        super(TruncatedGaussianPDF, self).__post_init__()
        self.constant = self._expectation_integral()
        self.constant = 1. / self.constant
        
    def __call__(self, x: Float[Array, "N 1"], element_wise: bool = False) -> Union[Float[Array, "R N"], Float[Array, "R"]]:
        return super(TruncatedGaussianPDF, self).__call__(x, element_wise=element_wise) * self.constant
        
    def get_mean(self) -> Float[Array, "R 1"]:
        return self._expectation_x()
    
    def get_variance(self) -> Float[Array, "R 1"]:
        return self._expectation_x_pow_2() - self.get_mean() ** 2
    
    def get_std(self) -> Float[Array, "R 1"]:
        return jnp.sqrt(self.get_variance())
    
    
        
    
    
    