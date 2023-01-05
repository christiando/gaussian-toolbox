##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for the most general form of functions, that are conjugate to    #
# Gaussian densities.                                                                            #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################
__author__ = "Christian Donner"

from jax import numpy as jnp
from .utils import linalg

# from dataclasses import dataclass, field
from .utils.dataclass import dataclass
from dataclasses import field
from typing import Union, Dict
from jaxtyping import Array, Float, Int


@dataclass(kw_only=True)
class ConjugateFactor:
    r"""Object representing a factor which is conjugate to a Gaussian measure.
    A general term, which can be multiplied with a Gaussian and the result is still a Gaussian,
    i.e. has the functional form

    .. math::

        f(X) = \beta \exp\left(- \frac{1}{2} X^\top\Lambda X + X^\top\nu\right),

    D is the dimension, and R the number of Gaussians.
    Note: At least :math:`\Lambda` or :math:`\nu` should be specified!

    Args:
        Lambda: Information (precision) matrix of the Gaussian
            distributions. Must be positive semidefinite.
        nu: Information vector of a Gaussian distribution. If None all
            zeros.
        ln_beta: The log constant factor of the factor. If None all
            zeros.
    """
    Lambda: Float[Array, "R D D"]
    nu: Float[Array, "R D"] = None
    ln_beta: Float[Array, "R"] = None

    def __post_init__(self):
        if self.nu is None:
            self.nu = jnp.zeros((self.R, self.D))
        if self.ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))

    @property
    def R(self) -> int:
        """Number of factors (leading dimension)."""
        return self.Lambda.shape[0]

    @property
    def D(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.Lambda.shape[1]

    def __str__(self) -> str:
        return "Conjugate factor u(x)"

    def __call__(
        self, x: Float[Array, "N D"], element_wise: bool = False
    ) -> Union[Float[Array, "N D"], Float[Array, "N"]]:
        """Evaluate the exponential term at :math:`X=x`.

        Args:
            x: Points where the factor should be evaluated.
            element_wise: Evaluates :math:`x` for only the corresponding density. Requires the N equals R. (Default=None)

        Returns:
            Exponential term.
        """
        return self.evaluate(x, element_wise)

    def evaluate_ln(
        self, x: Float[Array, "N D"], element_wise: bool = False
    ) -> Union[Float[Array, "R N"], Float[Array, "R"]]:
        r"""Evaluate the log-exponential term at :math:`X=x`.

        Args:
            x: Points where the factor should be evaluated.
            element_wise: Evaluates :math:`x` for only the corresponding density. Requires the N equals R.

        Raises:
            ValueError: Raised if N != R, and elementwise is True.

        Returns:
            Log exponential term.
        """

        if element_wise:
            if self.R != x.shape[0]:
                raise ValueError("Leading dimension of x must equal R.")
            x_Lambda_x = jnp.einsum(
                "ab,ab->a", jnp.einsum("abc,ac->ab", self.Lambda, x), x
            )
            x_nu = jnp.sum(x * self.nu, axis=1)
            return -0.5 * x_Lambda_x + x_nu + self.ln_beta

        else:
            x_Lambda_x = jnp.einsum(
                "adc,dc->ad", jnp.einsum("abc,dc->adb", self.Lambda, x), x
            )
            x_nu = jnp.dot(x, self.nu.T).T
            return -0.5 * x_Lambda_x + x_nu + self.ln_beta[:, None]

    def evaluate(
        self, x: Float[Array, "N D"], element_wise: bool = False
    ) -> Union[Float[Array, "R N"], Float[Array, "R"]]:
        r"""Evaluate the exponential term at :math:`X=x`.

        Args:
            x: Points where the factor should be evaluated.
            element_wise: Evaluates :math:`x` for only the corresponding
                density. Requires the N equals R.

        Returns:
            Exponential term.
        """
        return jnp.exp(self.evaluate_ln(x, element_wise))

    def slice(self, indices: Int[Array, "R_new"]) -> "ConjugateFactor":
        """Return an object with only the specified entries.

        Args:
            indices: The entries that should be contained in the
                returned object.

        Returns:
            The resulting Conjugate factor.
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return ConjugateFactor(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)

    def product(self) -> "ConjugateFactor":
        """Compute the product over all factors.

        .. math::

            g(X) = \prod_i f_i(X)

        Returns:
            Product of all factors.
        """
        Lambda_new = jnp.sum(self.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(self.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(self.ln_beta, axis=0, keepdims=True)
        return ConjugateFactor(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = False
    ) -> Dict:
        """Compute the product between the current factor and a Gaussian measure :math:`u(X)`.

        Returns :math:`f(X) * u(X)`.

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log
                determinants of the new Gaussian measure should be
                computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = jnp.reshape(
            (measure.Lambda[:, None] + self.Lambda[None]),
            (measure.R * self.R, self.D, self.D),
        )
        nu_new = jnp.reshape(
            (measure.nu[:, None] + self.nu[None]), (measure.R * self.R, self.D)
        )
        ln_beta_new = jnp.reshape(
            (measure.ln_beta[:, None] + self.ln_beta[None]), (measure.R * self.R)
        )
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
            ln_det_Sigma_new = -ln_det_Lambda_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def _hadamard_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = False
    ) -> Dict:
        """Compute the hadamard (componentwise) product between the current factor and a Gaussian measure :math:`u(X)`.

        Returns :math:`f(X) * u(X)`

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log
                determinants of the new Gaussian measure should be
                computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """

        Lambda_new = measure.Lambda + self.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
            ln_det_Sigma_new = -ln_det_Lambda_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def integrate_log_factor(self, phi_x: "GaussianMeasure") -> Float[Array, "R"]:
        """Integrate over the log factor with respect to a Gaussian measure.

        Args:
            phi_x: The integrating measure.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral.
        """
        if self.R != 1 and self.R != phi_x.R:
            raise NotImplementedError("Only implemented for R=1 or R=phi_x.R.")
        int_phi = phi_x.integrate()
        quadratic_integral = phi_x.integrate("(Ax+a)'(Bx+b)", B_mat=self.Lambda)
        linear_integral = jnp.einsum("ab,ab->a", self.nu, phi_x.integrate("x"))
        int_log_factor = (
            -0.5 * quadratic_integral + linear_integral + self.ln_beta * int_phi
        )
        return int_log_factor

    @staticmethod
    def get_trace(A: Float[Array, "R D D"]) -> Float[Array, "R"]:
        """Get trace of all matrices in A.

        Args:
            A: 3D matrix [_, D, D]

        Returns:
            Returns the trace of all matrices.
        """
        return jnp.sum(A.diagonal(axis1=-1, axis2=-2), axis=1)

    def to_dict(self) -> Dict:
        """Write Factor into dict.

        Returns:
            Dictionary with relevant parameters.
        """
        factor_dict = {"Lambda": self.Lambda, "nu": self.nu, "ln_beta": self.ln_beta}
        return factor_dict

    @classmethod
    def from_dict(cls, cls_dict: dict) -> "ConjugateFactor":
        """Creates class from dictionary

        Args:
            cls_dict: Dictionary with relevant parameters.

        Returns:
            The corresponding conjugate factor.
        """
        return cls(**cls_dict)


@dataclass(kw_only=True)
class LowRankFactor(ConjugateFactor):
    # TODO implement low rank updates with Woodbury inversion.
    Lambda: Float[Array, "R D D"]
    nu: Float[Array, "R D"] = None
    ln_beta: Float[Array, "R"] = None


@dataclass(kw_only=True)
class OneRankFactor(ConjugateFactor):
    r"""A low rank term, which can be multiplied with a Gaussian and the result is still a Gaussian.

    It has the functional form

    .. math::

        f(X) = \beta \exp\left(- \frac{1}{2}X^\top\Lambda X + X^\top\nu\right),

    but :math:`\Lambda` is of rank 1 and has the form :math:`\Lambda=g * vv^\top`.

    D is the dimension, and R the number of Gaussians.

    Args:
        v: Rank one vector for the constructing :math:`\Lambda`.
        g: Factor for :math:`\Lambda`. If None, it is assumed to be 1.
        nu: Information vector of a Gaussian distribution. If None all
            zeros.
        ln_beta: The log constant factor of the factor. If None all
            zeros.
    """
    v: Float[Array, "R D"]
    g: Float[Array, "R"] = None
    Lambda: Float[Array, "R D D"] = field(init=False)
    nu: Float[Array, "R D"] = None
    ln_beta: Float[Array, "R"] = None

    def __post_init__(self):
        if self.v is None:
            raise AttributeError("v must be defined!")
        if self.g is None:
            self.g = jnp.ones(self.R)
        self.Lambda = self._get_Lambda()
        if self.nu is None:
            self.nu = jnp.zeros((self.R, self.D))
        if self.ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))

    @property
    def R(self) -> int:
        """Number of factors (leading dimension)."""
        return self.v.shape[0]

    @property
    def D(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.v.shape[1]

    def slice(self, indices: Int[Array, "R_new"]) -> "OneRankFactor":
        """Return an object with only the specified entries.

        Args:
            indices: The entries that should be contained in the
                returned object.

        Returns:
            The resulting OneRankFactor.
        """
        v_new = jnp.take(self.v, indices, axis=0)
        g_new = jnp.take(self.g, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return OneRankFactor(v=v_new, g=g_new, nu=nu_new, ln_beta=ln_beta_new)

    def _get_Lambda(self) -> Float[Array, "R D D"]:
        r"""Compute the rank one matrix :math:`Lambda=g* vv^\top`

        Returns:
            The low rank matrix. Dimensions are [R, D, D].
        """
        return jnp.einsum("ab,ac->abc", self.v, jnp.einsum("a,ab->ab", self.g, self.v))

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> Dict:
        r"""Compute the product between the current factor and a Gaussian measure :math:`u(X)`.

        Returns :math:`f(X) * u(X)`. In contrast to full rank updates, the updated covariances and
        log determinants can be computed efficiently, using Woodbury matrix inversion and matrix deteriminant lemma.

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log
                determinants of the new Gaussian measure should be
                computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = jnp.reshape(
            (measure.Lambda[:, None] + self.Lambda[None]),
            (measure.R * self.R, self.D, self.D),
        )
        nu_new = jnp.reshape(
            (measure.nu[:, None] + self.nu[None]), (measure.R * self.R, self.D)
        )
        ln_beta_new = jnp.reshape(
            (measure.ln_beta[:, None] + self.ln_beta[None]), (measure.R * self.R)
        )
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
                ln_det_Sigma_new = -ln_det_Lambda_new
            else:
                # Sherman morrison inversion
                Sigma_v = jnp.einsum("abc,dc->adb", measure.Sigma, self.v)
                v_Sigma_v = jnp.einsum("abc,bc->ab", Sigma_v, self.v)
                denominator = 1.0 + self.g[None] * v_Sigma_v
                nominator = self.g[None, :, None, None] * jnp.einsum(
                    "abc,abd->abcd", Sigma_v, Sigma_v
                )
                Sigma_new = (
                    measure.Sigma[:, None] - nominator / denominator[:, :, None, None]
                )
                Sigma_new = Sigma_new.reshape((measure.R * self.R, self.D, self.D))
                # Matrix determinant lemma
                ln_det_Sigma_new = measure.ln_det_Sigma[:, None] - jnp.log(denominator)
                ln_det_Sigma_new = ln_det_Sigma_new.reshape((measure.R * self.R))
                ln_det_Lambda_new = -ln_det_Sigma_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def _hadamard_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> Dict:
        r"""Compute the hadamard (componentwise) product between the current factor and a Gaussian measure :math:`u(X)`.

        Returns :math:`f(x) * u(x)`. In contrast to full rank updates, the updated covariances and
        log determinants can be computed efficiently, using Woodbury matrix inversion and matrix deteriminant lemma.

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log
                determinants of the new Gaussian measure should be
        computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = measure.Lambda + self.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
                ln_det_Sigma_new = -ln_det_Lambda_new
            else:
                # Sherman morrison inversion
                Sigma_v = jnp.einsum("abc,ac->ab", measure.Sigma, self.v)
                v_Sigma_v = jnp.einsum("ab,ab->a", Sigma_v, self.v)
                denominator = 1.0 + self.g * v_Sigma_v
                nominator = self.g[:, None, None] * jnp.einsum(
                    "ab,ac->abc", Sigma_v, Sigma_v
                )
                Sigma_new = measure.Sigma - nominator / denominator[:, None, None]
                # Matrix determinant lemma
                ln_det_Sigma_new = measure.ln_det_Sigma - jnp.log(denominator)
                ln_det_Lambda_new = -ln_det_Sigma_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def to_dict(self) -> Dict:
        """Write Factor into dict.

        Returns:
            Dictionary with relevant parameters.
        """
        factor_dict = {"v": self.v, "g": self.g, "nu": self.nu, "ln_beta": self.ln_beta}
        return factor_dict


@dataclass(kw_only=True)
class LinearFactor(ConjugateFactor):
    r"""Object representing a factor which is conjugate to a Gaussian measure.

    A general term, which can be multiplied with a Gaussian and the result is still a Gaussian,
    i.e. has the functional form

    .. math::

        f(X) = \beta \exp\left(X^\top\nu\right),

    D is the dimension, and R the number of Gaussians.

    Note: At least :math:`\Lambda` or :math:`nu` should be specified!

    Args:
        Lambda: Is ignored.
        nu: Information vector of a Gaussian distribution. If None all
            zeros.
        ln_beta: The log constant factor of the factor. If None all
            zeros.
    """
    nu: Float[Array, "R D"]
    ln_beta: Float[Array, "R"] = None
    Lambda: Float[Array, "R D D"] = field(init=False)

    def __post_init__(self):
        self.Lambda = jnp.zeros((self.R, self.D, self.D))
        if self.ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))

    @property
    def R(self) -> int:
        """Number of factors (leading dimension)."""
        return self.nu.shape[0]

    @property
    def D(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.nu.shape[1]

    def slice(self, indices: Int[Array, "R_new"]) -> "LinearFactor":
        """Return an object with only the specified entries.

        Args:
            indices: The entries that should be contained in the
                returned object.

        Returns:
            The resulting LinearFactor.
        """
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return LinearFactor(nu=nu_new, ln_beta=ln_beta_new)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> Dict:
        """Compute the product between the current factor and a Gaussian measure :math:`u(X)`.

        Returns :math:`f(X) * u(X)`. For the linear term, we do not need to update the covariances.

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log
                determinants of the new Gaussian measure should be
                computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = jnp.tile(measure.Lambda[:, None], (1, self.R, 1, 1)).reshape(
            measure.R * self.R, self.D, self.D
        )
        nu_new = (measure.nu[:, None] + self.nu[None]).reshape(
            (measure.R * self.R, self.D)
        )
        ln_beta_new = (measure.ln_beta[:, None] + self.ln_beta[None]).reshape(
            (measure.R * self.R)
        )
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
                ln_det_Sigma_new = -ln_det_Lambda_new
            else:
                Sigma_new = jnp.tile(measure.Sigma[:, None], (1, self.R, 1, 1)).reshape(
                    measure.R * self.R, self.D, self.D
                )
                ln_det_Sigma_new = jnp.tile(
                    measure.ln_det_Sigma[:, None], (1, self.R)
                ).reshape(measure.R * self.R)
                ln_det_Lambda_new = -ln_det_Sigma_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def _hadamard_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> Dict:
        r"""Compute the hadamard (componentwise) product between the current factor and a Gaussian measure :math:`u(X)`.

             Returns :math:`f(X) * u(X)`. For the linear term, we do not need to update the covariances.

             :param measure: The gaussian measure the factor is multiplied with.
             :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be
        computed.
             :return: Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = measure.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(measure.Lambda)
                ln_det_Sigma_new = -ln_det_Lambda_new
            else:
                Sigma_new = measure.Sigma
                ln_det_Sigma_new = measure.ln_det_Sigma
                ln_det_Lambda_new = -ln_det_Sigma_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def to_dict(self) -> Dict:
        """Write Factor into dict.

        Returns:
            Dictionary with relevant parameters.
        """
        factor_dict = {"nu": self.nu, "ln_beta": self.ln_beta}
        return factor_dict


@dataclass(kw_only=True)
class ConstantFactor(ConjugateFactor):
    r"""A term, which can be multiplied with a Gaussian and the result is still a Gaussian.

    It has the functional form :math:`f(X) = \beta`.

    D is the dimension, and R the number of Gaussians.

    Args:
        Lambda: Is ignored.
        nu: Is ignored.
        ln_beta: The log constant factor of the factor.
        num_dim: The dimension of the Gaussian.
    """
    ln_beta: Float[Array, "R"]
    num_dim: int
    Lambda: Float[Array, "R D D"] = field(init=False)
    nu: Float[Array, "R D"] = field(init=False)

    def __post_init__(self):
        self.Lambda = jnp.zeros((self.R, self.D, self.D))
        self.nu = jnp.zeros((self.R, self.D))
        if self.nu is None:
            self.nu = jnp.zeros((self.R, self.D))
        if self.ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))

    @property
    def D(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.num_dim

    @property
    def R(self) -> int:
        """Number of factors (leading dimension)."""
        return self.ln_beta.shape[0]

    def slice(self, indices: Int[Array, "R_new"]) -> "ConstantFactor":
        """Return an object with only the specified entries.

        Args:
            indices: The entries that should be contained in the
                returned object.

        Returns:
            The resulting ConstantFactor.
        """
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return ConstantFactor(ln_beta=ln_beta_new, num_dim=self.D)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> Dict:
        """Compute the product between the current factor and a Gaussian measure :math:`u(X)`.

        Returns :math:`f(X) * u(X)`. For the linear term, we do not need to update the covariances.

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log
                determinants of the new Gaussian measure should be
                computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = jnp.tile(measure.Lambda[:, None], (1, self.R, 1, 1)).reshape(
            measure.R * self.R, self.D, self.D
        )
        nu_new = jnp.tile(measure.nu[:, None], (1, self.R, 1)).reshape(
            (measure.R * self.R, self.D)
        )
        ln_beta_new = (measure.ln_beta[:, None] + self.ln_beta[None]).reshape(
            (measure.R * self.R)
        )
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(Lambda_new)
                ln_det_Sigma_new = -ln_det_Lambda_new
            else:
                Sigma_new = jnp.tile(measure.Sigma[:, None], (1, self.R, 1, 1)).reshape(
                    measure.R * self.R, self.D, self.D
                )
                ln_det_Sigma_new = jnp.tile(
                    measure.ln_det_Sigma[:, None], (1, self.R)
                ).reshape(measure.R * self.R)
                ln_det_Lambda_new = -ln_det_Sigma_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def _hadamard_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> dict:
        """Computes the hadamard (componentwise) product between the current factor and a Gaussian measure :math:`u(X)`

                Returns :math:`f(X) * u(X)`. For the linear term, we do not need to update the covariances.

        Args:
            measure: The gaussian measure the factor is multiplied with.
            update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be computed.

        Returns:
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = measure.Lambda
        nu_new = measure.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(measure.Lambda)
                ln_det_Sigma_new = -ln_det_Lambda_new
            else:
                Sigma_new = measure.Sigma
                ln_det_Sigma_new = measure.ln_det_Sigma
                ln_det_Lambda_new = -ln_det_Sigma_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    def to_dict(self) -> Dict:
        """Write Factor into dict.

        Returns:
            Dictionary with relevant parameters.
        """
        factor_dict = {"ln_beta": self.ln_beta, "D": self.D}
        return factor_dict
