##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for the most general form of functions, that are conjugate to    #
# Gaussian densities.                                                                            #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################
__author__ = "Christian Donner"

__all__ = ["ConjugateFactor", "OneRankFactor", "LinearFactor", "ConstantFactor"]
from jax import numpy as jnp
from utils import linalg


class ConjugateFactor:
    def __init__(
        self, Lambda: jnp.ndarray, nu: jnp.ndarray = None, ln_beta: jnp.ndarray = None
    ):
        """Object representing a factor which is conjugate to a Gaussian measure.
        
        A general term, which can be multiplied with a Gaussian and the result is still a Gaussian, 
        i.e. has the functional form
        
        f(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),

        D is the dimension, and R the number of Gaussians.

        Note: At least Lambda or nu should be specified!

        :param Lambda: Information (precision) matrix of the Gaussian distributions. Must be postive semidefinite. 
            Dimensions should be [R, D, D].
        :type Lambda: jnp.ndarray
        :param nu: Information vector of a Gaussian distribution. If None all zeros. Dimensions should be [R, D], 
            defaults to None
        :type nu: jnp.ndarray, optional
        :param ln_beta: The log constant factor of the factor. If None all zeros. Dimensions should be [R], 
            defaults to None
        :type ln_beta: jnp.ndarray, optional
        """

        self.R, self.D = Lambda.shape[0], Lambda.shape[1]
        self.Lambda = Lambda

        if nu is None:
            self.nu = jnp.zeros((self.R, self.D))
        else:
            self.nu = nu
        if ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))
        else:
            self.ln_beta = ln_beta

    def __str__(self) -> str:
        return "Conjugate factor u(x)"

    def __call__(self, x: jnp.ndarray, element_wise: bool = False):
        """ Evaluate the exponential term at x.
        
        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.
        :param element_wise: bool
            Evaluates x for only the corresponding density. Requires the N equals R. (Default=None)
            
        :return: jnp.ndarray [R, N], [N]
            Exponential term.
        """
        return self.evaluate(x, element_wise)

    def evaluate_ln(self, x: jnp.ndarray, element_wise: bool = False) -> jnp.ndarray:
        """Evaluate the log-exponential term at x.

        :param x: Points where the factor should be evaluated. Dimensions should be [N, D].
        :type x: jnp.ndarray
        :param element_wise: Evaluates x for only the corresponding density. Requires the N equals R., defaults to False
        :type element_wise: bool, optional
        :raises ValueError: Raised if N != R, and elemntwise is True.
        :return: Log exponential term. Dimensions are [R, N], or [N]
        :rtype: jnp.ndarray 
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

    def evaluate(self, x: jnp.ndarray, element_wise: bool = False) -> jnp.ndarray:
        """Evaluate the exponential term at x.

        :param x: Points where the factor should be evaluated. Dimensions should be [N,D]
        :type x: jnp.ndarray
        :param element_wise: Evaluates x for only the corresponding density. Requires the N equals R, defaults to False
        :type element_wise: bool, optional
        :return: Exponential term. Dimensions are [R, N], or [N].
        :rtype: jnp.ndarray
        """
        return jnp.exp(self.evaluate_ln(x, element_wise))

    def slice(self, indices: jnp.ndarray) -> "ConjugateFactor":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting Conjugate factor.
        :rtype: ConjugateFactor
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return ConjugateFactor(Lambda_new, nu_new, ln_beta_new)

    def product(self) -> "ConjugateFactor":
        """Compute the product over all factors.
        
        g(x) = \prod_i f_i(x)

        :return: Product of all factors.
        :rtype: ConjugateFactor
        """
        Lambda_new = jnp.sum(self.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(self.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(self.ln_beta, axis=0, keepdims=True)
        return ConjugateFactor(Lambda_new, nu_new, ln_beta_new)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = False
    ) -> dict:
        """Compute the product between the current factor and a Gaussian measure u.
        
        Returns f(x) * u(x).

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to False
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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
    ) -> dict:
        """Compute the hadamard (componentwise) product between the current factor and a Gaussian measure u.A
        
        Returns f(x) * u(x)

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to False
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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

    def intergate_log_factor(self, phi_x: "GaussianMeasure") -> jnp.ndarray:
        """Integrate over the log factor with respect to a Gaussian measure.

        :param phi_x: The intergating measure.
        :type phi_x: GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral.
        :rtype: jnp.ndarray
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
    def get_trace(A: jnp.ndarray) -> jnp.ndarray:
        """Get trace of all matrices in A.

        :param A: 3D matrix [_, D, D]
        :type A: jnp.ndarray
        :return: Returns the trace of all matrices.
        :rtype: jnp.ndarray
        """
        return jnp.sum(A.diagonal(axis1=-1, axis2=-2), axis=1)


class LowRankFactor(ConjugateFactor):
    # TODO implement low rank updates with Woodbury inversion.
    def __init__(
        self,
        Lambda: jnp.ndarray = None,
        nu: jnp.ndarray = None,
        ln_beta: jnp.ndarray = None,
    ):
        """So far only place-holder.
        """
        super().__init__(Lambda, nu, ln_beta)


class OneRankFactor(LowRankFactor):
    def __init__(
        self,
        v: jnp.ndarray,
        g: jnp.ndarray = None,
        nu: jnp.ndarray = None,
        ln_beta: jnp.ndarray = None,
    ):
        """A low rank term, which can be multiplied with a Gaussian and the result is still a Gaussian.
    
        It has the functional form
        
        f(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),
        
        but Lambda is of rank 1 and has the form Lambda=g * vv'.

        D is the dimension, and R the number of Gaussians.

        :param v: Rank one vector for the constructing the Lambda matrix. Dimensions should be [R, D].
        :type v: jnp.ndarray, optional
        :param g: Factor for the Lambda matrix. If None, it is assumed to be 1. Dimensions should be [R], 
            defaults to None
        :type g: jnp.ndarray, optional
        :param nu: Information vector of a Gaussian distribution. Dimensions should be [R, D], defaults to None
        :type nu: jnp.ndarray, optional
        :param ln_beta: The log constant factor of the factor. If None all zeros. Dimensions should be [R], 
            defaults to None
        :type ln_beta: jnp.ndarray, optional
        """
        self.R, self.D = v.shape
        self.v = v
        if g is None:
            self.g = jnp.ones(self.R)
        else:
            self.g = g

        Lambda = self._get_Lambda()
        super().__init__(Lambda, nu, ln_beta)

    def slice(self, indices: jnp.ndarray) -> "OneRankFactor":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting OneRankFactor.
        :rtype: OneRankFactor
        """
        v_new = jnp.take(self.v, indices, axis=0)
        g_new = jnp.take(self.g, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return OneRankFactor(v_new, g_new, nu_new, ln_beta_new)

    def _get_Lambda(self) -> jnp.ndarray:
        """Compute the rank one matrix Lambda=g* vv'

        :return: The low rank matrix. Dimensions are [R, D, D].
        :rtype: jnp.ndarray
        """
        return jnp.einsum("ab,ac->abc", self.v, jnp.einsum("a,ab->ab", self.g, self.v))

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = True
    ) -> dict:
        """Compute the product between the current factor and a Gaussian measure u.
        
        Returns f(x) * u(x). In contrast to full rank updates, the updated covariances and 
        log determinants can be computed efficiently, using Woodbury matrix inversion and matrix deteriminant lemma.

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to True
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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
        self, measure: "GaussianMeasure", update_full=True
    ) -> dict:
        """Compute the hadamard (componentwise) product between the current factor and a Gaussian measure u.
        
        Returns f(x) * u(x). In contrast to full rank updates, the updated covariances and 
        log determinants can be computed efficiently, using Woodbury matrix inversion and matrix deteriminant lemma.

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
        computed, defaults to True
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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


class LinearFactor(ConjugateFactor):
    def __init__(self, nu: jnp.ndarray, ln_beta: jnp.ndarray = None):
        """A term, which can be multiplied with a Gaussian and the result is still a Gaussian.
        
        It has the functional form
        
        f(x) = beta * exp(x'nu),

        D is the dimension, and R the number of Gaussians.

        Note: At least Lambda or nu should be specified!

        :param nu: Information vector. Dimensions should be [R, D].
        :type nu: jnp.ndarray
        :param ln_beta: The log constant factor of the factor. If None all zeros. Dimensions should be [R], 
            defaults to None
        :type ln_beta: jnp.ndarray, optional
        """

        self.R, self.D = nu.shape[0], nu.shape[1]
        self.nu = nu
        self.Lambda = jnp.zeros((self.R, self.D))
        if ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))
        else:
            self.ln_beta = ln_beta

    def slice(self, indices: jnp.ndarray) -> "LinearFactor":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting LinearFactor.
        :rtype: LinearFactor
        """
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return LinearFactor(nu_new, ln_beta_new)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full=True
    ) -> dict:
        """Compute the product between the current factor and a Gaussian measure u.
        
        Returns f(x) * u(x). For the linear term, we do not need to update the covariances.

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to True
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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
        self, measure: "GaussianMeasure", update_full=True
    ) -> dict:
        """ Compute the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
        Returns f(x) * u(x). For the linear term, we do not need to update the covariances.

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to True
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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


class ConstantFactor(ConjugateFactor):
    def __init__(self, ln_beta: jnp.ndarray, D: int):
        """A term, which can be multiplied with a Gaussian and the result is still a Gaussian.
        
        It has the functional form f(x) = beta.

        D is the dimension, and R the number of Gaussians.

        :param ln_beta: The log constant factor of the factor. Dimensions should be [R]
        :type ln_beta: jnp.ndarray
        :param D: The dimension of the Gaussian.
        :type D: int
        """
        self.R, self.D = ln_beta.shape[0], D
        Lambda = jnp.zeros((self.R, self.D, self.D))
        nu = jnp.zeros((self.R, self.D))
        ln_beta = ln_beta
        super().__init__(Lambda, nu, ln_beta)

    def slice(self, indices: jnp.ndarray) -> "ConstantFactor":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting ConstantFactor.
        :rtype: ConstantFactor
        """
        ln_beta_new = jnp.array(self.ln_beta, indices, axis=0)
        return ConstantFactor(ln_beta_new, self.D)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full=True
    ) -> dict:
        """Compute the product between the current factor and a Gaussian measure u.
        
        Returns f(x) * u(x). For the linear term, we do not need to update the covariances.

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to True
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
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
        self, measure: "GaussianMeasure", update_full=True
    ) -> dict:
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
        Returns f(x) * u(x). For the linear term, we do not need to update the covariances.

        :param measure: The gaussian measure the factor is multiplied with.
        :type measure: GaussianMeasure
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be 
            computed, defaults to True
        :type update_full: bool, optional
        :return: Returns the resulting dictionary to create GaussianMeasure.
        :rtype: dict
        """
        Lambda_new = measure.Lambda
        nu_new = measure.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = linalg.invert_matrix(measure.Sigma)
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

