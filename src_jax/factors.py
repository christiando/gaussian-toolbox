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
from jax import scipy as jsc
from typing import Tuple


class ConjugateFactor:
    def __init__(self, Lambda, nu: jnp.ndarray = None, ln_beta: jnp.ndarray = None):
        """ A general term, which can be multiplied with a Gaussian and the result is still a Gaussian, 
            i.e. has the functional form
        
            f(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),

            D is the dimension, and R the number of Gaussians.

            Note: At least Lambda or nu should be specified!
            
        :param Lambda: jnp.ndarray [R, D, D]
            Information (precision) matrix of the Gaussian distributions. Must be postive semidefinite.
        :param nu: jnp.ndarray [R, D]
            Information vector of a Gaussian distribution. If None all zeros. (Default=None)
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
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

    def evaluate_ln(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the log-exponential term at x.
        
        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.
        :param r: list
            Indices of densities that need to be evaluated. If empty, all densities are evaluated. (Default=[])
            
        :return: jnp.ndarray [N, R]
            Log exponential term.
        """
        x_Lambda_x = jnp.einsum(
            "adc,dc->ad", jnp.einsum("abc,dc->adb", self.Lambda, x), x
        )
        x_nu = jnp.dot(x, self.nu.T).T
        return -0.5 * x_Lambda_x + x_nu + self.ln_beta[:, None]

    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the exponential term at x.
        
        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.
            
        :return: jnp.ndarray [N, R]
            Exponential term.
        """
        return jnp.exp(self.evaluate_ln(x))

    def slice(self, indices: list) -> "ConjugateFactor":
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: ConjugateFactor
            The resulting Conjugate factor.
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return ConjugateFactor(Lambda_new, nu_new, ln_beta_new)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full: bool = False
    ) -> "GaussianMeasure":
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=False)
            
        :return: dict
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
            Sigma_new, ln_det_Lambda_new = self.invert_matrix(Lambda_new)
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
    ) -> "GaussianMeasure":
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=False)
            
        :return: dict
            Returns the resulting dictionary to create GaussianMeasure.
        """

        Lambda_new = measure.Lambda + self.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            Sigma_new, ln_det_Lambda_new = self.invert_matrix(Lambda_new)
            ln_det_Sigma_new = -ln_det_Lambda_new
            new_density_dict.update(
                {
                    "Sigma": Sigma_new,
                    "ln_det_Lambda": ln_det_Lambda_new,
                    "ln_det_Sigma": ln_det_Sigma_new,
                }
            )
        return new_density_dict

    @staticmethod
    def invert_matrix(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        L = jsc.linalg.cho_factor(A)
        # TODO: Check whether we can make it mor efficienty with solve_triangular.
        # L_inv = solve_triangular(L, jnp.eye(L.shape[0]), lower=True,
        #                         check_finite=False)
        # L_inv = vmap(lambda B: jsc.linalg.solve_triangular(B, jnp.eye(B.shape[0])))(A)
        A_inv = jsc.linalg.cho_solve(L, jnp.eye(A.shape[1])[None].tile((len(A), 1, 1)))
        # A_inv = jnp.einsum('acb,acd->abd', L_inv, L_inv)
        ln_det_A = 2.0 * jnp.sum(jnp.log(L[0].diagonal(axis1=-1, axis2=-2)), axis=1)
        return A_inv, ln_det_A

    @staticmethod
    def get_trace(A: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(A.diagonal(axis1=-1, axis2=-2), axis=1)


class LowRankFactor(ConjugateFactor):
    # TODO implement low rank updates with Woodbury inversion.
    def __init__(
        self,
        Lambda: jnp.ndarray = None,
        nu: jnp.ndarray = None,
        ln_beta: jnp.ndarray = None,
    ):
        super().__init__(Lambda, nu, ln_beta)


class OneRankFactor(LowRankFactor):
    def __init__(
        self,
        v: jnp.ndarray = None,
        g: jnp.ndarray = None,
        nu: jnp.ndarray = None,
        ln_beta: jnp.ndarray = None,
    ):
        """ A term, which can be multiplied with a Gaussian and the result is still a Gaussian, 
            i.e. has the functional form
        
            f(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),
            
            but Lambda is of rank 1 and has the form Lambda=g * vv'.

            D is the dimension, and R the number of Gaussians.
            
        :param v: jnp.ndarray [R, D]
            Rank one vector for the constructing the Lambda matrix.
        :param g: jnp.narray [R]
            Factor for the Lambda matrix. If None, it is assumed to be 1. (Default=None)
        :param nu: jnp.ndarray [R, D]
            Information vector of a Gaussian distribution. If None all zeros. (Default=None)
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        """
        self.R, self.D = v.shape
        self.v = v
        if g is None:
            self.g = jnp.ones(self.R)
        else:
            self.g = g

        Lambda = self._get_Lambda()
        super().__init__(Lambda, nu, ln_beta)

    def slice(self, indices: list) -> "OneRankFactor":
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: OneRankFactor
            The resulting OneRankFactor.
        """
        v_new = jnp.take(self.v, indices, axis=0)
        g_new = jnp.take(self.g, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return OneRankFactor(v_new, g_new, nu_new, ln_beta_new)

    def _get_Lambda(self) -> jnp.ndarray:
        """ Computes the rank one matrix
        
            Lambda=g* vv'
            
        :return: jnp.ndarray [R, D, D]
            The low rank matrix.
        """
        return jnp.einsum("ab,ac->abc", self.v, jnp.einsum("a,ab->ab", self.g, self.v))

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full=True
    ) -> "GaussianMeasure":
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. In contrast to full rank updates, the updated covariances and 
            log determinants can be computed efficiently.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)
            
        :return: dict
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
                Sigma_new, ln_det_Lambda_new = self.invert_matrix(Lambda_new)
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
    ) -> "GaussianMeasure":
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. In contrast to full rank updates, the updated covariances and 
            log determinants can be computed efficiently.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)
            
        :return: dict
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = measure.Lambda + self.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = self.invert_matrix(Lambda_new)
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
        """ A term, which can be multiplied with a Gaussian and the result is still a Gaussian and it has the form
            i.e. has the functional form
        
            f(x) = beta * exp(x'nu),

            D is the dimension, and R the number of Gaussians.

            Note: At least Lambda or nu should be specified!
            
        :param nu: jnp.ndarray [R, D]
            Information vector of a Gaussian distribution.
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        """

        self.R, self.D = nu.shape[0], nu.shape[1]
        self.nu = nu
        self.Lambda = jnp.zeros((self.R, self.D))
        if ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))
        else:
            self.ln_beta = ln_beta

    def slice(self, indices: list) -> "LinearFactor":
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        return LinearFactor(nu_new, ln_beta_new)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full=True
    ) -> "GaussianMeasure":
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)
            
        :return: dict
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
                Sigma_new, ln_det_Lambda_new = self.invert_matrix(Lambda_new)
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
    ) -> "GaussianMeasure":
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)
            
        :return: dict
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = measure.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = self.invert_matrix(measure.Lambda)
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
        """ A term, which can be multiplied with a Gaussian and the result is still a Gaussian and it has the form
            i.e. has the functional form
        
            f(x) = beta,

            D is the dimension, and R the number of Gaussians.
            
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor.
        :param D: int
            The dimension of the Gaussian.
        """

        self.R, self.D = ln_beta.shape[0], D
        Lambda = jnp.zeros((self.R, self.D, self.D))
        nu = jnp.zeros((self.R, self.D))
        ln_beta = ln_beta
        super().__init__(Lambda, nu, ln_beta)

    def slice(self, indices: list) -> "ConstantFactor":
        ln_beta_new = jnp.array(self.ln_beta, indices, axis=0)
        return ConstantFactor(ln_beta_new, self.D)

    def _multiply_with_measure(
        self, measure: "GaussianMeasure", update_full=True
    ) -> "GaussianMeasure":
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)
            
        :return: dict
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
                Sigma_new, ln_det_Lambda_new = self.invert_matrix(Lambda_new)
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
    ) -> "GaussianMeasure":
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)
            
        :return: dict
            Returns the resulting dictionary to create GaussianMeasure.
        """
        Lambda_new = measure.Lambda
        nu_new = measure.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        new_density_dict = {"Lambda": Lambda_new, "nu": nu_new, "ln_beta": ln_beta_new}
        if update_full:
            if measure.Sigma is None:
                Sigma_new, ln_det_Lambda_new = self.invert_matrix(measure.Sigma)
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

