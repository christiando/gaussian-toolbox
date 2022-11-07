##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for kernels construction.                                        #
#                                                                                                #
# Author: Maurizio Di Lucente                                                                    #
##################################################################################################

__author__ = "Maurizio Di Lucente"

from jax.config import config

config.update("jax_enable_x64", True)
from jax import numpy as jnp
import objax


class Kernel(objax.Module):
    def __init__(
        self,
        length_scale: float = 0.1,
        variance: float = 1.0,
        alpha: float = 1.0,
        regularization_const: float = 1e-5,
    ) -> None:
        """
        Base class for isotropic and stationary kernels, i.e., kernels that only
        depend on r = ‖x - x'‖.

        :param length_scale: value for the lengthscale parameter, defaults to .1
        :type length_scale: float, optional

        :param variance: value for the variance parameter, defaults to 1
        :type variance: float, optional

        :param alpha: alpha parameter for relative weighting, defaults to 1
        :type alpha: float, optional (only for RationalQuadratic kernels)

        :param regularization_const: constant to stabilize matrix if squared, defaults to 1e-5
        :type regularization_const: float, optional
        """

        self.log_length_scale = objax.TrainVar(jnp.array(jnp.log(length_scale)))
        self.log_variance = objax.TrainVar(jnp.array(jnp.log(variance)))
        self.alpha = alpha
        self.regularization_const = regularization_const

    @property
    def length_scale(self):
        return jnp.exp(self.log_length_scale.value)

    @property
    def variance(self):
        return jnp.exp(self.log_variance.value)

    def scale(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Scale the input by length scale ℓ.

        :param X: Input data [N x D] matrix
        :type X: jnp.ndarray

        :return: scaled input data.
        :rtype: jnp.ndarray
        """

        D = X.shape[-1]
        if D != self.length_scale.size and self.length_scale.size != 1:
            raise ValueError("Wrong dimensions of data and length scale(s)")

        X_scaled = X / self.length_scale
        return X_scaled

    def euclidean_distance(self, XA: jnp.array, XB: jnp.array = None) -> jnp.ndarray:
        """
        Calculate the euclidean distance between the scaled matrix XA and the
        scaled matrix XB.
        If XB is not provided, the distance is calculated on XA itself.

        :param XA: Scaled input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Scaled input data [M x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: matrix with pairwise scaled euclidean distances
        :rtype: jnp.ndarray
        """

        if XB is None:
            XB = XA

        delta_X = self.scale(XA[None] - XB[:, None])
        r2 = jnp.transpose(jnp.sum(jnp.square(delta_X), axis=2))

        floor_val = jnp.sqrt(1e-36)  # In order to avoid gradient at 0
        r = jnp.sqrt(jnp.maximum(r2, jnp.square(floor_val)))
        r -= floor_val * jnp.equal(r, floor_val)
        return r

    def regularize_kernel(self, K: jnp.array, r: jnp.array) -> jnp.ndarray:
        """
        If the kernel matrix is quadratic with zeros on the diagonal,
        a constant is added to the diagonal to stabilize the matrix.

        :param K: computed kernel matrix
        :type K: jnp.ndarray

        :param r: matrix with pairwise scaled euclidean distances
        :type r: jnp.ndarray

        :return: regularized kernel matrix
        :rtype: jnp.ndarray
        """

        K = K + self.regularization_const * jnp.equal(r, 0)
        return K

    def eval_diag(self, XA: jnp.ndarray) -> jnp.ndarray:
        """_summary_

        :param XA: Input data [N x D]
        :type XA: jnp.ndarray
        :return: Diagonal of kernel matrix [1 x N]
        :rtype: jnp.ndarray
        """
        N = XA.shape[0]
        return (self.variance + self.regularization_const)* jnp.ones((1, N))


class RBF(Kernel):
    """
    The radial basis function (RBF) or squared exponential kernel.
    The kernel equation is:

        k(r) = σ² exp{-½ r²}

    where:
    r is the Euclidean distance between the input points, scaled by the
    lengthscale parameter ℓ. σ²  is the variance parameter.
    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def evaluate(self, XA, XB=None) -> jnp.ndarray:
        """
        :param XA: Input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: RBF kernel computed.
        :rtype: jnp.ndarray
        """

        r = self.euclidean_distance(XA, XB)
        K_unreg = self.variance * jnp.exp(-0.5 * jnp.square(r))
        return self.regularize_kernel(K_unreg, r)


class Matern12(Kernel):
    """
    The Matern 1/2 kernel. The kernel equation is:

        k(r) =  σ² * exp(-r)

    where:
    r is the Euclidean distance between the input points, scaled by the
    lengthscale parameter ℓ. σ²  is the variance parameter.
    """

    def evaluate(self, XA, XB=None) -> jnp.ndarray:
        """
        :param XA: Input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: Matern 1/2 kernel computed.
        :rtype: jnp.ndarray
        """

        r = self.euclidean_distance(XA, XB)
        K_unreg = self.variance * jnp.exp(-r)
        return self.regularize_kernel(K_unreg, r)


class Matern32(Kernel):
    """
    The Matern 3/2 kernel. The kernel equation is:

        k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r is the Euclidean distance between the input points, scaled by the
    lengthscale parameter ℓ. σ²  is the variance parameter.
    """

    def evaluate(self, XA: jnp.ndarray, XB: jnp.ndarray = None) -> jnp.ndarray:
        """
        :param XA: Input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: Matern 3/2 kernel computed.
        :rtype: jnp.ndarray
        """

        r = self.euclidean_distance(XA, XB)
        K_unreg = self.variance * (1 + jnp.sqrt(3) * r) * jnp.exp(-jnp.sqrt(3) * r)
        return self.regularize_kernel(K_unreg, r)


class Matern52(Kernel):
    """
    The Matern 5/2 kernel. The kernel equation is:

        k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r is the Euclidean distance between the input points, scaled by the
    lengthscale parameter ℓ. σ²  is the variance parameter.
    """

    def evaluate(self, XA: jnp.ndarray, XB: jnp.ndarray = None) -> jnp.ndarray:
        """
        :param XA: Input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: Matern 5/2 kernel computed.
        :rtype: jnp.ndarray
        """

        r = self.euclidean_distance(XA, XB)
        K_unreg = (
            self.variance
            * (1 + jnp.sqrt(5) * r + 5 / 3 * jnp.square(r))
            * jnp.exp(-jnp.sqrt(5) * r)
        )
        return self.regularize_kernel(K_unreg, r)


class Exponential(Kernel):
    """
    The Exponential kernel. The kernel equation is:

        k(r) =  σ² * exp(-½ r)

    where:
    r is the Euclidean distance between the input points, scaled by the
    lengthscale parameter ℓ. σ²  is the variance parameter.
    The Exponential kernel is equivalent to a Matern12 kernel with
    doubled lengthscales.
    """

    def evaluate(self, XA: jnp.ndarray, XB: jnp.ndarray = None) -> jnp.ndarray:
        """
        :param XA: Input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: Exponential kernel computed.
        :rtype: jnp.ndarray
        """

        r = self.euclidean_distance(XA, XB)
        K_unreg = self.variance * jnp.exp(-0.5 * r)
        return self.regularize_kernel(K_unreg, r)


class RationalQuadratic(Kernel):
    """
    Rational Quadratic kernel. The kernel equation is:

        k(r) = σ² (1 + r² / 2α)^(-α)

    where:
    r is the Euclidean distance between the input points, scaled by the
    lengthscale parameter ℓ. σ²  is the variance parameter.
    α determines relative weighting of small-scale and large-scale fluctuations.
    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
    """

    def evaluate(self, XA: jnp.ndarray, XB: jnp.ndarray = None) -> jnp.ndarray:
        """
        :param XA: Input data [N x D] where the kernel is evaluated
        :type XA: jnp.ndarray

        :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
        :type XB: jnp.ndarray, optional

        :return: Exponential kernel computed.
        :rtype: jnp.ndarray
        """

        r = self.euclidean_distance(XA, XB)
        K_unreg = self.variance * (1 + 0.5 * jnp.square(r) / self.alpha) ** (
            -self.alpha
        )
        return self.regularize_kernel(K_unreg, r)


def ConstructKernel(
    XA: jnp.ndarray,
    XB: jnp.ndarray = None,
    kernel_type: str = "RBF",
    length_scale: float = 0.1,
    variance: float = 1,
    alpha: float = 1,
) -> jnp.ndarray:
    """
    Construct kernel depending on the type.

    :param XA: Input data [N x D] where the kernel is evaluated
    :type XA: jnp.ndarray

    :param XB: Input data [N x D] where the kernel is evaluated, defaults to XA
    :type XB: jnp.ndarray, optional

    :param kernel_type: type of kernel, defaults to "RBF"
    :type kernel_type: char, optional

    :param length_scale: characteristic length-scale of the process, defaults to .1
    :type length_scale: float, optional

    :param variance: variance parameter of the process, defaults to 1
    :type variance: float, optional

    :param alpha: alpha parameter for relative weighting, defaults to 1
    :type alpha: float, optional

    :return K: Required kernel computed.
    :rtype K: jnp.ndarray

    :return kernel: kernel class.
    :rtype kernel: kernel_type class
    """

    options = {
        "RBF": RBF(length_scale=length_scale, variance=variance),
        "Matern12": Matern12(length_scale=length_scale, variance=variance),
        "Matern32": Matern32(length_scale=length_scale, variance=variance),
        "Matern52": Matern52(length_scale=length_scale, variance=variance),
        "Exponential": Exponential(length_scale=length_scale, variance=variance),
        "RationalQuadratic": RationalQuadratic(
            length_scale=length_scale, variance=variance, alpha=alpha
        ),
    }

    kernel = options[kernel_type]
    K = kernel.evaluate(XA, XB)
    return K, kernel
