__all__ = ["TruncatedGaussianMeasure", "TruncatedGaussianPDF"]
from jax import numpy as jnp
from jax.experimental import checkify
from jax.lax import scan
from ..utils.dataclass import dataclass
from dataclasses import field
from jaxtyping import Array, Float
from typing import Any, Callable, Tuple, Union, Dict
from gaussian_toolbox import pdf, measure
from .misc import binom, normal_pdf, normal_cdf


@dataclass(kw_only=True)
class TruncatedGaussianMeasure:
    """Truncated Gaussian measure.

    Args:
        measure: Gaussian measure.
        lower_limit: Lower limit of truncation.
        upper_limit: Upper limit of truncation.
    """

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
        if not isinstance(self.measure, pdf.GaussianPDF):
            self.density = self.measure.get_density()
        else:
            self.density = self.measure
        self.constant = self.measure.integrate()
        self.alpha = jnp.where(
            jnp.isfinite(self.lower_limit),
            (
                jnp.where(jnp.isfinite(self.lower_limit), self.lower_limit, 0)
                - self.density.mu
            )
            * jnp.sqrt(self.density.Lambda[:, :, 0]),
            self.lower_limit,
        )
        self.beta = jnp.where(
            jnp.isfinite(self.upper_limit),
            (
                jnp.where(jnp.isfinite(self.upper_limit), self.upper_limit, 0)
                - self.density.mu
            )
            * jnp.sqrt(self.density.Lambda[:, :, 0]),
            self.upper_limit,
        )

    def _check_limits(self):
        """Check that lower_limit and upper_limit are valid."""
        if self.lower_limit is None and self.upper_limit is None:
            raise ValueError(
                "At least one of lower_limit and upper_limit must be specified."
            )
        elif self.lower_limit is None:
            self.lower_limit = -jnp.inf * jnp.ones((self.R, self.D))
        elif self.upper_limit is None:
            self.upper_limit = jnp.inf * jnp.ones((self.R, self.D))
        # TODO: Find a way to check that lower_limit < upper_limit
        self.lower_limit = self.lower_limit * jnp.ones((self.R, self.D))
        self.upper_limit = self.upper_limit * jnp.ones((self.R, self.D))

    @property
    def R(self) -> int:
        """Number of factors (leading dimension)."""
        return self.measure.Lambda.shape[0]

    @property
    def D(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.measure.Lambda.shape[1]

    def __call__(
        self, x: Float[Array, "N 1"], element_wise: bool = False
    ) -> Union[Float[Array, "R N"], Float[Array, "R"]]:
        """Evaluate the measure at x.

        :raises ValueError: If x has wrong shape.
        :return: The measure evaluated at x.
        """
        if element_wise:
            if self.R != x.shape[0]:
                raise ValueError("Leading dimension of x must equal R.")
            in_limits = jnp.all(
                jnp.logical_and(
                    jnp.greater_equal(x, self.lower_limit),
                    jnp.less_equal(x, self.upper_limit),
                ),
                axis=-1,
            )
        else:
            in_limits = jnp.all(
                jnp.logical_and(
                    jnp.greater_equal(x[None], self.lower_limit[:, None]),
                    jnp.less_equal(x[None], self.upper_limit[:, None]),
                ),
                axis=-1,
            )
        return self.measure(x, element_wise=element_wise) * in_limits

    def integrate(self, expr: str = "1", **kwargs) -> Float[Array, "R ..."]:
        r"""Integrate the indicated expression with respect to the truncated Gaussian measure.

        E.g. expr="x**2" means that :math:`\int X^2 u(X){\rm d}X` is computed, and :math:`A` and a can be provided.

        :param expr: Indicates the expression that should be integrated. Check measure's integration dict.
        :return: The integral result.
        """
        return self.integration_dict[expr](**kwargs)

    @property
    def integration_dict(self) -> Dict:
        return {
            "1": self.integral,
            "x": self.integrate_x,
            "x**2": self.integrate_x_pow_2,
            "x**k": self.integrate_x_pow_k,
        }

    def _expectation_integral(self) -> Float[Array, "R"]:
        """Compute the normalizing constant of the integral of the truncated Gaussian measure.

        Returns:
            The normalizing constant of the measure.
        """
        return jnp.squeeze(normal_cdf(self.beta) - normal_cdf(self.alpha), axis=-1)

    def integral(self) -> Float[Array, "R"]:
        """Compute the integral of the truncated Gaussian measure.

        Returns:
            The integral of the measure.
        """
        return self._expectation_integral() * self.constant

    def _expectation_x(self) -> Float[Array, "R 1"]:
        """Compute the expectation of x under the truncated Gaussian density corresponding to the measure.

        Returns:
            The expectation of x.
        """
        Z = self._expectation_integral()[:, None]
        Z = jnp.where(Z != 0, Z, 1.0)
        mean = self.density.mu + (
            normal_pdf(self.alpha) - normal_pdf(self.beta)
        ) / Z * jnp.sqrt(self.density.Sigma[..., 0])
        mean = jnp.where(Z != 0, mean, 0.0)
        return mean

    def integrate_x(self) -> Float[Array, "R 1"]:
        """Compute the integral of x under the truncated Gaussian measure.

        Returns:
            The integral of x.
        """
        return self._expectation_x() * self.integral()[:, None]

    def _get_variance(self) -> Float[Array, "R 1"]:
        """Compute the variance of x under the truncated Gaussian density corresponding to the measure.

        Returns:
            The variance of x.
        """
        Z = self._expectation_integral()[:, None]
        Z = jnp.where(Z != 0, Z, 1.0)
        beta_pdf = jnp.where(
            jnp.isfinite(self.upper_limit), self.beta, 0.0
        ) * normal_pdf(self.beta)
        alpha_pdf = jnp.where(
            jnp.isfinite(self.lower_limit), self.alpha, 0.0
        ) * normal_pdf(self.alpha)
        variance = self.density.Sigma[..., 0] * (
            1
            - (beta_pdf - alpha_pdf) / Z
            - (normal_pdf(self.alpha) - normal_pdf(self.beta)) ** 2 / Z**2
        )
        variance = jnp.where(Z != 0, variance, 0.0)
        return variance

    def integrate_x_pow_2(self) -> Float[Array, "R 1"]:
        """Compute the integral of x^2 under the truncated Gaussian measure.

        Returns:
            The integral of x^2.
        """
        variance = self._get_variance()
        mu = self._expectation_x()
        return (variance + mu**2) * self.integral()[:, None]

    def _get_moment(
        self, order: int, return_all: bool = False
    ) -> Union[Float[Array, "R order+1"], Float[Array, "R order+1"]]:
        """Compute the moment of order `order` under the truncated Gaussian density corresponding to the measure.

        Returns:
            The moment of `order`.
        """
        denominator = normal_cdf(self.beta[:, 0]) - normal_cdf(self.alpha[:, 0])
        denominator = jnp.where(denominator != 0, denominator, 1.0)

        def scan_function(carry, k):
            L2, L1 = carry
            beta_pdf = jnp.where(
                jnp.isfinite(self.beta[:, 0]),
                self.beta[:, 0] ** (k - 1) * normal_pdf(self.beta[:, 0]),
                0,
            )
            alpha_pdf = jnp.where(
                jnp.isfinite(self.alpha[:, 0]),
                self.alpha[:, 0] ** (k - 1) * normal_pdf(self.alpha[:, 0]),
                0,
            )
            L_new = -(beta_pdf - alpha_pdf) / denominator + (k - 1) * L2
            return (L1, L_new), L_new

        L0, L1 = (
            jnp.ones((self.R,)),
            -(normal_pdf(self.beta[:, 0]) - normal_pdf(self.alpha[:, 0])) / denominator,
        )
        Ls = scan(scan_function, (L0, L1), jnp.arange(2, order + 1))[1]
        Ls = jnp.concatenate([L0[None], L1[None], Ls], axis=0)
        k_range = jnp.arange(0, order + 1)[:, None]
        if return_all:
            moments = jnp.cumsum(
                binom(order, k_range)
                * jnp.sqrt(self.density.Sigma[:, :, 0].T) ** k_range
                * self.density.mu.T ** (order - k_range)
                * Ls,
                axis=0,
            )
        else:
            moments = jnp.sum(
                binom(order, k_range)
                * jnp.sqrt(self.density.Sigma[:, :, 0].T) ** k_range
                * self.density.mu.T ** (order - k_range)
                * Ls,
                axis=0,
            )
        moments = jnp.where(denominator != 0, moments, 0.0)
        return moments

    def integrate_x_pow_k(self, k: int) -> Float[Array, "R 1"]:
        """Compute the integral of x^k under the truncated Gaussian measure.

        Args:
            k: The order of the moment to compute.

        Returns:
            The integral of x^k.
        """
        return self._get_moment(k)[:, None] * self.integral()[:, None]

    def get_density(self) -> "TruncatedGaussianPDF":
        """Return the truncated Gaussian density corresponding to the measure.

        Returns:
            The truncated Gaussian density corresponding to the measure.
        """
        return TruncatedGaussianPDF(
            measure=self.density,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )


@dataclass(kw_only=True)
class TruncatedGaussianPDF(TruncatedGaussianMeasure):
    """Normalized Truncated Gaussian density.

    Args:
        measure: Gaussian measure.
        lower_limit: Lower limit of truncation.
        upper_limit: Upper limit of truncation.
    """

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
        self.constant = 1.0 / self.constant

    def __call__(
        self, x: Float[Array, "N 1"], element_wise: bool = False
    ) -> Union[Float[Array, "R N"], Float[Array, "R"]]:
        """Compute the value of the density at x.

        Args:
            x: The point at which to compute the density.
            element_wise: Whether to return the density at each point of x or the density at x.

        Returns:
            The value of the density at x.
        """
        if element_wise:
            return (
                super(TruncatedGaussianPDF, self).__call__(x, element_wise=element_wise)
                * self.constant
            )
        else:
            return (
                super(TruncatedGaussianPDF, self).__call__(x, element_wise=element_wise)
                * self.constant[:, None]
            )

    def get_mean(self) -> Float[Array, "R 1"]:
        """Compute the mean of the density.

        Returns:
            The mean of the density.
        """
        return self._expectation_x()

    def get_variance(self) -> Float[Array, "R 1"]:
        """Compute the variance of the density.

        Returns:
            The variance of the density.
        """
        variance = self._get_variance()
        return variance

    def get_std(self) -> Float[Array, "R 1"]:
        """Compute the standard deviation of the density.

        Returns:
            The standard deviation of the density.
        """
        return jnp.sqrt(self.get_variance())
