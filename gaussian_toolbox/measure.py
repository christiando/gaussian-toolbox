##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) measures.                                 #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

from jax import numpy as jnp
from . import factor
from typing import Tuple
from .utils.linalg import invert_matrix, invert_diagonal

from .utils.dataclass import dataclass

@dataclass(kw_only=True)
class GaussianMeasure(factor.ConjugateFactor):
    r"""A measure with a Gaussian form.

    .. math::
    
        u(X) = \beta * \exp(- 0.5 * X^\top\Lambda X + X^\top\nu),

    D is the dimension, and R the number of Gaussians. 

    :param Lambda: Information (precision) matrix of the Gaussian distributions. Needs to be postive definite. 
        Dimensions should be [R, D, D].
    :type Lambda: jnp.ndarray
    :param nu: Information vector of a Gaussian distribution. If None all zeros. Dimensions should be [R, D], 
        defaults to None
    :type nu: jnp.ndarray, optional
    :param ln_beta: The log constant factor of the factor. If None all zeros. Dimensions should be [R], 
        defaults to None
    :type ln_beta: jnp.ndarray, optional
    :param Sigma: Covariance matrix of the Gaussian distributions. Needs to be positive definite. 
        Dimensions should be [R, D, D], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param ln_det_Lambda: Log determinant of Lambda. Dimensions should be [R], defaults to None
    :type ln_det_Lambda: jnp.ndarray, optional
    :param ln_det_Sigma: Log determinant of Sigma. Dimensions should be [R], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    """
    Lambda: jnp.ndarray
    nu: jnp.ndarray = None
    ln_beta: jnp.ndarray = None
    Sigma: jnp.ndarray = None
    ln_det_Lambda: jnp.ndarray = None
    ln_det_Sigma: jnp.ndarray = None

    def __post_init__(self):
        if self.nu is None:
            self.nu = jnp.zeros((self.R, self.D))
        if self.ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))
        self.Sigma = self.Sigma
        self.ln_det_Lambda = self.ln_det_Lambda
        self.ln_det_Sigma = self.ln_det_Sigma
        self.lnZ = None
        self.mu = None
       
    @property 
    def integration_dict(self) -> dict:
        return {
            "1": self.integral,
            "x": self.integrate_x,
            "(Ax+a)": self.integrate_general_linear,
            "xx'": self.integrate_xxT,
            "(Ax+a)'(Bx+b)": self.integrate_general_quadratic_inner,
            "(Ax+a)(Bx+b)'": self.integrate_general_quadratic_outer,
            "(Ax+a)(Bx+b)'(Cx+c)": self.integrate_general_cubic_inner,
            "(Ax+a)'(Bx+b)(Cx+c)'": self.integrate_general_cubic_outer,
            "x(A'x + a)x'": self.integrate_cubic_outer,  # Rename
            "xb'xx'": self.integrate_xbxx,
            "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)": self.integrate_general_quartic_inner,
            "(Ax+a)(Bx+b)'(Cx+c)(Dx+d)'": self.integrate_general_quartic_outer,
            "log u(x)": self.integrate_log_factor,
        }

    def __str__(self) -> str:
        return "Gaussian measure phi(x)"

    def slice(self, indices: jnp.ndarray) -> "GaussianMeasure":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting Gaussian measure.
        :rtype: GaussianMeasure
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        new_measure = GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if self.Sigma is not None:
            new_measure.Sigma = jnp.take(self.Sigma, indices, axis=0)
            new_measure.ln_det_Sigma = jnp.take(self.ln_det_Sigma, indices, axis=0)
            new_measure.ln_det_Lambda = jnp.take(self.ln_det_Lambda, indices, axis=0)
        return new_measure

    def _prepare_integration(self):
        """Compute the log normalization and :math:`\mu`. (Requires inversion of precision matrix.)
        """
        if self.lnZ is None:
            self.compute_lnZ()
        if self.mu is None:
            self.compute_mu()

    def compute_lnZ(self):
        """ Compute the log partition function.
        """
        if self.Sigma is None:
            self.invert_lambda()
        nu_Lambda_nu = jnp.einsum(
            "ab,ab->a", self.nu, jnp.einsum("abc,ac->ab", self.Sigma, self.nu)
        )
        self.lnZ = 0.5 * (
            nu_Lambda_nu + self.D * jnp.log(2.0 * jnp.pi) + self.ln_det_Sigma
        )

    def invert_lambda(self):
        """Invert precision matrix.
        """
        self.Sigma, self.ln_det_Lambda = invert_matrix(self.Lambda)
        self.ln_det_Sigma = -self.ln_det_Lambda

    def __mul__(self, factor: factor.ConjugateFactor,) -> "GaussianMeasure":
        r"""Compute the product between the measure :math:`u(X)` and a conjugate factor :math:`f(X)`.

        Returns :math:`f(X) * u(X)`.

        :param factor: The conjugate factor the measure is multiplied with.
        :type factor: factor.ConjugateFactor
        :return: Returns the resulting GaussianMeasure.
        :rtype: GaussianMeasure
        """
        return self.multiply(factor)

    def multiply(
        self, factor: factor.ConjugateFactor, update_full: bool = False
    ) -> "GaussianMeasure":
        """Compute the product between the measure :math:`u(X)` and a conjugate factor :math:`f(X)`.

        Returns :math:`f(X) * u(X)`.

        :param factor: The conjugate factor the measure is multiplied with.
        :type factor: factor.ConjugateFactor
        :return: Resulting GaussianMeasure.
        :rtype: GaussianMeasure
        """
        new_measure_dict = factor._multiply_with_measure(self, update_full=update_full)
        return GaussianMeasure(**new_measure_dict)

    def hadamard(
        self, factor: factor.ConjugateFactor, update_full: bool = False
    ) -> "GaussianMeasure":
        """Compute the hadamard (componentwise) product between the measure :math:`u(X)` and a conjugate factor :math:`f(X)`.

        Returns :math:`f(X) * u(X)`.

        :param factor: The conjugate factor the measure is multiplied with.
        :type factor: factor.ConjugateFactor
        :param update_full: Whether also the covariance and the log determinants of the new Gaussian measure should be computed. , defaults to False
        :type update_full: bool, optional
        :return: Resulting GaussianMeasure.
        :rtype: GaussianMeasure
        """
        new_measure_dict = factor._hadamard_with_measure(self, update_full=update_full)
        return GaussianMeasure(**new_measure_dict)

    def product(self) -> "GaussianMeasure":
        """Compute the product over all factor.
        
        .. math::
        
            v(X) = \prod_i u_i(X)

        :return: Factor of all factor.
        :rtype: GaussianMeasure
        """
        Lambda_new = jnp.sum(self.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(self.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(self.ln_beta, axis=0, keepdims=True)
        new_measure = GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if self.Sigma is not None:
            new_measure._prepare_integration()
        return new_measure

    def integrate(self, expr: str = "1", **kwargs) -> jnp.ndarray:
        r""" Integrate the indicated expression with respect to the Gaussian measure.
        
        E.g. expr="(Ax+a)" means that :math:`\int (AX + a)u(X){\rm d}X` is computed, and :math:`A` and a can be provided.

        :param expr: Indicates the expression that should be integrated. Check measure's integration dict, 
            defaults to "1"
        :type expr: str, optional
        :return: The integral result.
        :rtype: jnp.ndarray
        """
        return self.integration_dict[expr](**kwargs)

    def log_integral_light(self) -> jnp.ndarray:
        r"""Compute the log integral of the exponential term.

        .. math::
        
            \log \int u(X) {\rm d}X.

        :return: Log integral. Dimensions are [R].
        :rtype: jnp.ndarray
        """
        if self.lnZ is None:
            self.compute_lnZ()
        return self.lnZ + self.ln_beta

    def log_integral(self) -> jnp.ndarray:
        r"""Compute the log integral of the exponential term.

        .. math::
        
            \log \int u(X) {\rm d}X.

        :return: Log integral. Dimensions are [R].
        :rtype: jnp.ndarray
        """
        self._prepare_integration()
        return self.lnZ + self.ln_beta

    def integral_light(self) -> jnp.ndarray:
        r""" Compute the log integral of the exponential term.
        
        .. math::

            \int u(X) dX.

        :return:  Integral. Dimensions are [R].jnp.ndarray [R]
        :rtype: jnp.ndarray
        """
        return jnp.exp(self.log_integral_light())

    def integral(self) -> jnp.ndarray:
        r""" Compute the log integral of the exponential term.

        .. math::
        
            \int u(X) {\rm d}X.

        :return:  Integral. Dimensions are [R].jnp.ndarray [R]
        :rtype: jnp.ndarray
        """
        return jnp.exp(self.log_integral())

    def normalize(self):
        r""" Normalize the term such that
        
        .. math::

            \int u(X) {\rm d}X = 1.
        """
        self.compute_lnZ()
        self.ln_beta = -self.lnZ

    def is_normalized(self) -> jnp.ndarray:
        """Check whether measure is normalized

        :return: Boolean area indicating which measure is normalized.
        :rtype: jnp.ndarray
        """
        return jnp.equal(self.lnZ, -self.ln_beta)

    def compute_mu(self):
        """Converts from information to mean vector.
        """
        if self.Sigma is None:
            self.invert_lambda()
        self.mu = jnp.einsum("abc,ac->ab", self.Sigma, self.nu)

    def get_density(self) -> "GaussianPDF":
        """Return the corresponing normalised density object.

        :return: Corresponding density object.
        :rtype: GaussianPDF
        """
        from . import pdf

        self._prepare_integration()
        return pdf.GaussianPDF(
            Sigma=self.Sigma,
            mu=self.mu,
            Lambda=self.Lambda,
            ln_det_Sigma=self.ln_det_Sigma,
        )

    def _get_default(
        self, mat: jnp.array = None, vec: jnp.array = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make matrices and vectors right dimensions for integration.

        :param mat: Matrix or matrices, 2D or 3D. If None, it returns identity, defaults to None
        :type mat: jnp.array, optional
        :param vec: Vector or vectors, 1D or 2D. If None returns 0s. , defaults to None
        :type vec: jnp.array, optional
        :return: Returns matrix/matrices and vectors with dimensionality ready for integration.
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        if mat is None:
            mat = jnp.eye(self.D)
        if vec is None:
            vec = jnp.zeros(self.D)

        if vec.ndim == 1:
            if mat.ndim == 2:
                mat = jnp.tile(mat[None], [1, 1, 1])
            vec = jnp.tile(vec[None], [mat.shape[0], 1])
        else:
            if mat.ndim == 2:
                mat = jnp.tile(mat[None], [vec.shape[0], 1, 1])
        return mat, vec

    # Linear integals

    def _expectation_x(self) -> jnp.ndarray:
        r""" Compute the expectation.
        
        .. math::

            int X {\rm d}u(X) / \int {\rm d}u(X)

        :return: The solved intergal. Dimension is [R, D].
        :rtype: jnp.ndarray
        """
        return self.mu

    def integrate_x(self) -> jnp.ndarray:
        r""" Compute the integral.
        
        .. math::

            \int X {\rm d}u(X)

        :return: The solved intergal. Dimension is [R, D].
        :rtype: jnp.ndarray
        """
        constant = self.integral()
        return jnp.einsum("a,ab->ab", constant, self._expectation_x())

    def _expectation_general_linear(
        self, A_mat: jnp.ndarray, a_vec: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Compute the linear expectation.
        
        .. math::

            \int (AX+a) {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D]
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K]
        :rtype: jnp.ndarray
        """
        return jnp.einsum("abc,ac->ab", A_mat, self.mu) + a_vec

    def integrate_general_linear(
        self, A_mat: jnp.ndarray = None, a_vec: jnp.ndarray = None
    ) -> jnp.ndarray:
        r"""Compute the linear expectation.

        .. math::
        
            \int (AX+a) {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. If None, it is assumed identity. Dimensions should be  [K,D] or [R,K,D], 
            defaults to None
        :type A_mat: jnp.ndarray, optional
        :param a_vec: Real valued vector. Dimensions should be [K] or [R,K], defaults to None
        :type a_vec: jnp.ndarray, optional
        :return: The solved intergal. Dimensions are [R, K]
        :rtype: jnp.ndarray
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        constant = self.integral()
        return jnp.einsum(
            "a,ab->ab", constant, self._expectation_general_linear(A_mat, a_vec)
        )

    # Quadratic integrals

    def _expectation_xxT(self) -> jnp.ndarray:
        r"""Compute the expectation.
        
        .. math::

            \int XX^\top {\rm d}u(X) / \int {\rm d}u(X)

        :return: The solved intergal. Dimensions are [R, D, D]
        :rtype: jnp.ndarray
        """
        return self.Sigma + jnp.einsum("ab,ac->acb", self.mu, self.mu)

    def integrate_xxT(self) -> jnp.ndarray:
        r"""Compute the integral
        
        .. math::
        
            \int XX^\top {\rm d}u(X)
            
        :return:  The solved intergal. Dimensions are [R, D, D]
        :rtype: jnp.ndarray
        """
        constant = self.integral()
        return jnp.einsum("a,abc->abc", constant, self._expectation_xxT())

    def _expectation_general_quadratic_inner(
        self,
        A_mat: jnp.ndarray,
        a_vec: jnp.ndarray,
        B_mat: jnp.ndarray,
        b_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Computes the quartic expectation.

        .. math::
        
            \int (AX+a)^\top(BX+b) {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D]
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type b_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R].
        :rtype: jnp.ndarray
        """
        AB = jnp.einsum("abc,abd->acd", A_mat, B_mat)
        ABSigma_trace = self.get_trace(jnp.einsum("cab,cbd->cad", AB, self.Sigma))
        mu_AB_mu = jnp.einsum(
            "ab,ab->a", jnp.einsum("ab, abc-> ac", self.mu, AB), self.mu
        )
        muAb = jnp.einsum("ab,ab->a", jnp.einsum("ab,acb->ac", self.mu, A_mat), b_vec)
        aBm_b = jnp.einsum(
            "ab, ab->a", a_vec, self._expectation_general_linear(B_mat, b_vec)
        )
        return ABSigma_trace + mu_AB_mu + muAb + aBm_b

    def integrate_general_quadratic_inner(
        self,
        A_mat: jnp.ndarray = None,
        a_vec: jnp.ndarray = None,
        B_mat: jnp.ndarray = None,
        b_vec: jnp.ndarray = None,
    ) -> jnp.ndarray:
        r""" Compute the quadratic expectation.

        int (AX+a)'(BX+b) {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D]
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type b_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R].
        :rtype: jnp.ndarray
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return constant * self._expectation_general_quadratic_inner(
            A_mat, a_vec, B_mat, b_vec
        )

    def _expectation_general_quadratic_outer(
        self,
        A_mat: jnp.ndarray,
        a_vec: jnp.ndarray,
        B_mat: jnp.ndarray,
        b_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Compute the quadratic expectation.

        .. math::
        
            \int (AX+a)(BX+b)^\top {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D]
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type b_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K, L].
        :rtype: jnp.ndarray
        """
        Exx = self._expectation_xxT()
        AxxB = jnp.einsum("cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", Exx, B_mat))
        Axb = jnp.einsum("ab,ac->abc", jnp.einsum("cab,cb->ca", A_mat, self.mu), b_vec)
        aBx_b = jnp.einsum(
            "ba, bc->bac", a_vec, self._expectation_general_linear(B_mat, b_vec)
        )
        return AxxB + Axb + aBx_b

    def integrate_general_quadratic_outer(
        self,
        A_mat: jnp.ndarray = None,
        a_vec: jnp.ndarray = None,
        B_mat: jnp.ndarray = None,
        b_vec: jnp.ndarray = None,
    ) -> jnp.ndarray:
        r""" Compute the quadratic expectation.
        
        .. math::

            \int (AX+a)(BX+b)' {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D]
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type b_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K, L].
        :rtype: jnp.ndarray
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return jnp.einsum(
            "a,abc->abc",
            constant,
            self._expectation_general_quadratic_outer(A_mat, a_vec, B_mat, b_vec),
        )

    # Cubic integrals

    def _expectation_xbxx(self, b_vec: jnp.ndarray) -> jnp.ndarray:
        r""" Compute the cubic expectation.

        .. math::
        
            \int XbXX^\top {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.


        :param b_vec: Real avlued vector. Dimensions should be [1, D] or [R, D].
        :type b_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, D, D]
        :rtype: jnp.ndarray
        """
        constant = self.integral()
        Exx = self._expectation_xxT()
        mub_outer = jnp.einsum("ab,ac->abc", self.mu, b_vec)
        mbExx = jnp.einsum("abc,acd->abd", mub_outer, Exx)
        bmu_inner = jnp.einsum("ab,ab->a", self.mu, b_vec)
        bmSigma = jnp.einsum("a,abc->abc", bmu_inner, self.Sigma)
        bmu_outer = jnp.einsum("ab,ac->abc", b_vec, self.mu)
        Sigmabm = jnp.einsum("abd,ade->abe", self.Sigma, bmu_outer)
        return mbExx + bmSigma + Sigmabm

    def _expectation_cubic_outer(
        self, A_mat: jnp.ndarray, a_vec: jnp.ndarray
    ) -> jnp.ndarray:
        r""" Compute the cubic expectation.

        .. math::
        
            \int X(A^\top X + a)X^\top {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. If None, it is assumed identity. Dimensions should be [1,1,D] or [R,1,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. If None, it is assumed zeros. Dimensions should be [1,1] or [R,1].
        :type a_vec: jnp.ndarray
        :return: Solved integral. Dimensions are [R, D, D].
        :rtype: jnp.ndarray
        """
        # xAxx
        xAxx = self._expectation_xbxx(b_vec=A_mat)
        axx = a_vec[:, None, None] * self._expectation_xxT()
        return xAxx + axx

    def integrate_cubic_outer(
        self, A_mat: jnp.ndarray = None, a_vec: jnp.ndarray = None
    ) -> jnp.ndarray:
        r"""Compute the cubic integration.

        .. math::
        
            \int X(A^\top X + a)X^\top {\rm d}u(X).

        :param A_mat: Real valued matrix. If None, it is assumed identity. Dimensions should be [1,D] or [R,1,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. If None, it is assumed zeros. Dimensions should be [1] or [R,1].
        :type a_vec: jnp.ndarray
        :return: Solved integral. Dimensions are [R, D, D].
        :rtype: jnp.ndarray
        """
        if A_mat is None:
            A_mat = jnp.ones((1, self.D))
        if a_vec is None:
            a_vec = jnp.zeros(1)
        if A_mat.ndim == 2:
            A_mat = jnp.tile(A_mat[None], [1, 1, 1])
        if a_vec.ndim == 1:
            a_vec = jnp.tile(a_vec[None], [1, 1])
        constant = self.integral()
        return constant[:, None, None] * self._expectation_cubic_outer(
            A_mat=A_mat[:, 0], a_vec=a_vec[:, 0]
        )

    def integrate_xbxx(self, b_vec: jnp.ndarray) -> jnp.ndarray:
        r"""Compute the cubic integral.

        .. math::
        
            \int Xb^\top XX^\top {\rm d}u(X)

        :param b_vec: Real valued vector. Dimensions should be  [D,] or [1, D,].
        :type b_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, D, D].
        :rtype: jnp.ndarray
        """
        constant = self.integral()
        if b_vec is None:
            b_vec = jnp.zeros(1)
        if b_vec.ndim == 1:
            b_vec = jnp.tile(b_vec[None], [1, 1])
        return constant[:, None, None] * self._expectation_xbxx(b_vec)

    def _expectation_general_cubic_inner(
        self,
        A_mat: jnp.ndarray,
        a_vec: jnp.ndarray,
        B_mat: jnp.ndarray,
        b_vec: jnp.ndarray,
        C_mat: jnp.ndarray,
        c_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Compute the quartic expectation.

        .. math::
        
            \int (AX+a)(BX+b)^\top(CX+c) {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type c_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K].
        :rtype: jnp.ndarray
        """
        Amu_a = jnp.einsum("cab,cb-> ca", A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum("cab,cb-> ca", B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum("cab,cb-> ca", C_mat, self.mu) + c_vec
        BSigmaC = jnp.einsum(
            "cab,cbd->cad", B_mat, jnp.einsum("abc,adc->abd", self.Sigma, C_mat)
        )
        BmubCmuc = jnp.einsum("ab,ab->a", Bmu_b, Cmu_c)

        BCm_c = jnp.einsum("cab,ca->cb", B_mat, Cmu_c)
        CBm_b = jnp.einsum("cab,ca->cb", C_mat, Bmu_b)
        first_term = jnp.einsum(
            "abc,ac->ab", jnp.einsum("cab,cbd->cad", A_mat, self.Sigma), BCm_c + CBm_b
        )
        second_term = Amu_a * (self.get_trace(BSigmaC) + BmubCmuc)[:, None]
        return first_term + second_term

    def integrate_general_cubic_inner(
        self,
        A_mat: jnp.ndarray = None,
        a_vec: jnp.ndarray = None,
        B_mat: jnp.ndarray = None,
        b_vec: jnp.ndarray = None,
        C_mat: jnp.ndarray = None,
        c_vec: jnp.ndarray = None,
    ) -> jnp.ndarray:
        r"""Compute the quadratic integration.

        .. math::
        
            \int (AX+a)(BX+b)^\top(CX+c)  {\rm d}u(X).

        :param A_mat: Real valued matrix. Dimensions should be [K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type c_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K].
        :rtype: jnp.ndarray
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:, None] * self._expectation_general_cubic_inner(
            A_mat, a_vec, B_mat, b_vec, C_mat, c_vec
        )

    def _expectation_general_cubic_outer(
        self,
        A_mat: jnp.ndarray,
        a_vec: jnp.ndarray,
        B_mat: jnp.ndarray,
        b_vec: jnp.ndarray,
        C_mat: jnp.ndarray,
        c_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Compute the cubic expectation.

        .. math::
        
            \int (AX+a)^\top(BX+b)(CX+c)^\top {\rm d}\phi(X),

        with :math:`\phi(x) = u(X) / \int {\rm d}u(X)`.
        
        # REMARK: Does the same thing as inner transposed.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,L] or [R,K].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type c_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, L].
        :rtype: jnp.ndarray
        """
        Amu_a = jnp.einsum("cab,cb-> ca", A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum("cab,cb-> ca", B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum("cab,cb-> ca", C_mat, self.mu) + c_vec
        BSigmaC = jnp.einsum(
            "cab,cbd->cad", B_mat, jnp.einsum("abc,adc->abd", self.Sigma, C_mat)
        )
        ASigmaC = jnp.einsum(
            "cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", self.Sigma, C_mat)
        )
        ASigmaB = jnp.einsum(
            "cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", self.Sigma, B_mat)
        )
        BmubCmuc = jnp.einsum("ab,ac->abc", Bmu_b, Cmu_c)
        AmuaCmuc = jnp.einsum("ab,ac->abc", Amu_a, Cmu_c)
        AmuaBmub = jnp.einsum("ab,ab->a", Amu_a, Bmu_b)
        first_term = jnp.einsum("ab,abc->ac", Amu_a, BSigmaC + BmubCmuc)
        second_term = jnp.einsum("ab,abc->ac", Bmu_b, ASigmaC + AmuaCmuc)
        third_term = -AmuaBmub[:, None] * Cmu_c
        fourth_term = self.get_trace(ASigmaB)[:, None] * Cmu_c
        return first_term + second_term + third_term + fourth_term

    def integrate_general_cubic_outer(
        self,
        A_mat: jnp.ndarray = None,
        a_vec: jnp.ndarray = None,
        B_mat: jnp.ndarray = None,
        b_vec: jnp.ndarray = None,
        C_mat: jnp.ndarray = None,
        c_vec: jnp.ndarray = None,
    ) -> jnp.ndarray:
        r""" Compute the quadratic integration
        
        .. math::

           \int (AX+a)^\top(Bx+b)(Cx+c)^\top {\rm d}u(X),

        :param A_mat: Real valued matrix. Dimensions should be [K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [K,D] or [R,K,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [L] or [R,K].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type c_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, L].
        :rtype: jnp.ndarray
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:, None] * self._expectation_general_cubic_outer(
            A_mat, a_vec, B_mat, b_vec, C_mat, c_vec
        )

    # Quartic integrals

    def _expectation_general_quartic_outer(
        self,
        A_mat: jnp.ndarray,
        a_vec: jnp.ndarray,
        B_mat: jnp.ndarray,
        b_vec: jnp.ndarray,
        C_mat: jnp.ndarray,
        c_vec: jnp.ndarray,
        D_mat: jnp.ndarray,
        d_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Compute the quartic expectation
        
        .. math::

            \int (AX+a)(BX+b)^\top(CX+c)(DX+d)^\top {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type c_vec: jnp.ndarray
        :param D_mat: Real valued matrix. Dimensions should be [1,M,D] or [R,M,D].
        :type D_mat: jnp.ndarray
        :param d_vec: Real valued vector. Dimensions should be [1,M] or [R,M].
        :type d_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K, M].
        :rtype: jnp.ndarray
        """
        Amu_a = jnp.einsum("cab,cb-> ca", A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum("cab,cb-> ca", B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum("cab,cb-> ca", C_mat, self.mu) + c_vec
        Dmu_d = jnp.einsum("cab,cb-> ca", D_mat, self.mu) + d_vec
        ASigmaB = jnp.einsum(
            "cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", self.Sigma, B_mat)
        )
        CSigmaD = jnp.einsum(
            "cab,cbd->cad", C_mat, jnp.einsum("abc,adc->abd", self.Sigma, D_mat)
        )
        ASigmaC = jnp.einsum(
            "cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", self.Sigma, C_mat)
        )
        BSigmaD = jnp.einsum(
            "cab,cbd->cad", B_mat, jnp.einsum("abc,adc->abd", self.Sigma, D_mat)
        )
        ASigmaD = jnp.einsum(
            "cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", self.Sigma, D_mat)
        )
        BSigmaC = jnp.einsum(
            "cab,cbd->cad", B_mat, jnp.einsum("abc,adc->abd", self.Sigma, C_mat)
        )
        AmuaBmub = jnp.einsum("ab,ac->abc", Amu_a, Bmu_b)
        CmucDmud = jnp.einsum("ab,ac->abc", Cmu_c, Dmu_d)
        AmuaCmuc = jnp.einsum("ab,ac->abc", Amu_a, Cmu_c)
        BmubDmud = jnp.einsum("ab,ac->abc", Bmu_b, Dmu_d)
        BmubCmuc = jnp.einsum("ab,ab->a", Bmu_b, Cmu_c)
        AmuaDmud = jnp.einsum("ab,ac->abc", Amu_a, Dmu_d)
        first_term = jnp.einsum("abc,acd->abd", ASigmaB + AmuaBmub, CSigmaD + CmucDmud)
        second_term = jnp.einsum("abc,acd->abd", ASigmaC + AmuaCmuc, BSigmaD + BmubDmud)
        third_term = BmubCmuc[:, None, None] * (ASigmaD - AmuaDmud)
        fourth_term = self.get_trace(BSigmaC)[:, None, None] * (ASigmaD + AmuaDmud)
        return first_term + second_term + third_term + fourth_term

    def integrate_general_quartic_outer(
        self,
        A_mat: jnp.ndarray = None,
        a_vec: jnp.ndarray = None,
        B_mat: jnp.ndarray = None,
        b_vec: jnp.ndarray = None,
        C_mat: jnp.ndarray = None,
        c_vec: jnp.ndarray = None,
        D_mat: jnp.ndarray = None,
        d_vec: jnp.ndarray = None,
    ) -> jnp.ndarray:
        r""" Compute the quartic integral.
        
        .. math::

            \int (AX+a)(BX+b)^\top(CX+c)(DX+d)^\top {\rm d}u(X).

        :param A_mat: Real valued matrix. Dimensions should be [K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type c_vec: jnp.ndarray
        :param D_mat: Real valued matrix. Dimensions should be [M,D] or [R,M,D].
        :type D_mat: jnp.ndarray
        :param d_vec: Real valued vector. Dimensions should be [M] or [R,M].
        :type d_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R, K, M].
        :rtype: jnp.ndarray
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        D_mat, d_vec = self._get_default(D_mat, d_vec)
        constant = self.integral()
        return constant[:, None, None] * self._expectation_general_quartic_outer(
            A_mat, a_vec, B_mat, b_vec, C_mat, c_vec, D_mat, d_vec
        )

    def _expectation_general_quartic_inner(
        self,
        A_mat: jnp.ndarray,
        a_vec: jnp.ndarray,
        B_mat: jnp.ndarray,
        b_vec: jnp.ndarray,
        C_mat: jnp.ndarray,
        c_vec: jnp.ndarray,
        D_mat: jnp.ndarray,
        d_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Compute the quartic expectation.

        .. math::
        
            \int (AX+a)'(BX+b)(CX+c)'(DX+d) {\rm d}\phi(X),

        with :math:`\phi(X) = u(X) / \int {\rm d}u(X)`.

        :param A_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [1,K,D] or [R,K,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [1,K] or [R,K].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type c_vec: jnp.ndarray
        :param D_mat: Real valued matrix. Dimensions should be [1,L,D] or [R,L,D].
        :type D_mat: jnp.ndarray
        :param d_vec: Real valued vector. Dimensions should be [1,L] or [R,L].
        :type d_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R].
        :rtype: jnp.ndarray
        """
        Amu_a = jnp.einsum("cab,cb-> ca", A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum("cab,cb-> ca", B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum("cab,cb-> ca", C_mat, self.mu) + c_vec
        Dmu_d = jnp.einsum("cab,cb-> ca", D_mat, self.mu) + d_vec
        ASigmaB = jnp.einsum(
            "cab,cbd->cad", A_mat, jnp.einsum("abc,adc->abd", self.Sigma, B_mat)
        )
        CSigmaD = jnp.einsum(
            "cab,cbd->cad", C_mat, jnp.einsum("abc,adc->abd", self.Sigma, D_mat)
        )

        AmuaBmub = jnp.einsum("ab,ab->a", Amu_a, Bmu_b)
        CmucDmud = jnp.einsum("ab,ab->a", Cmu_c, Dmu_d)
        CD = jnp.einsum("abc,abd->acd", C_mat, D_mat)
        CD_DC = CD + jnp.swapaxes(CD, axis1=1, axis2=2)
        SCD_DCS = jnp.einsum(
            "abc,acd->abd", jnp.einsum("abc,acd->abd", self.Sigma, CD_DC), self.Sigma
        )
        ASCD_DCSB = jnp.einsum(
            "abc,adc->abd", jnp.einsum("cab,cbd->cad", A_mat, SCD_DCS), B_mat
        )
        Am_aB = jnp.einsum("ab,abc->ac", Amu_a, B_mat)
        Bm_bA = jnp.einsum("ab,abc->ac", Bmu_b, A_mat)
        CDm_d = jnp.einsum("cab,ca->cb", C_mat, Dmu_d)
        DCm_c = jnp.einsum("cab,ca->cb", D_mat, Cmu_c)
        first_term = self.get_trace(ASCD_DCSB)
        second_term = jnp.einsum(
            "ab,ab->a",
            jnp.einsum("ab,abc->ac", Am_aB + Bm_bA, self.Sigma),
            CDm_d + DCm_c,
        )
        third_term = (self.get_trace(ASigmaB) + AmuaBmub) * (
            self.get_trace(CSigmaD) + CmucDmud
        )
        return first_term + second_term + third_term

    def integrate_general_quartic_inner(
        self,
        A_mat: jnp.ndarray = None,
        a_vec: jnp.ndarray = None,
        B_mat: jnp.ndarray = None,
        b_vec: jnp.ndarray = None,
        C_mat: jnp.ndarray = None,
        c_vec: jnp.ndarray = None,
        D_mat: jnp.ndarray = None,
        d_vec: jnp.ndarray = None,
    ) -> jnp.ndarray:
        r""" Compute the quartic integral.
        
        .. math::

            \int (AX+a)(BX+b)'(CX+c)(DX+d)' {\rm d}u(X).

        :param A_mat: Real valued matrix. Dimensions should be [K,D] or [R,K,D].
        :type A_mat: jnp.ndarray
        :param a_vec: Real valued vector. Dimensions should be [K] or [R,K].
        :type a_vec: jnp.ndarray
        :param B_mat: Real valued matrix. Dimensions should be [K,D] or [R,K,D].
        :type B_mat: jnp.ndarray
        :param b_vec: Real valued vector. Dimensions should be [K] or [R,K].
        :type b_vec: jnp.ndarray
        :param C_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type C_mat: jnp.ndarray
        :param c_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type c_vec: jnp.ndarray
        :param D_mat: Real valued matrix. Dimensions should be [L,D] or [R,L,D].
        :type D_mat: jnp.ndarray
        :param d_vec: Real valued vector. Dimensions should be [L] or [R,L].
        :type d_vec: jnp.ndarray
        :return: The solved intergal. Dimensions are [R].
        :rtype: jnp.ndarray
        """

        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        D_mat, d_vec = self._get_default(D_mat, d_vec)
        constant = self.integral()
        return constant * self._expectation_general_quartic_inner(
            A_mat, a_vec, B_mat, b_vec, C_mat, c_vec, D_mat, d_vec
        )

    def integrate_log_factor(self, factor: factor.ConjugateFactor) -> jnp.array:
        """Integrates over a log factor.

        :param factor: The factor, which will be intergrated.
        :type factor: factor.ConjugateFactor
        :return: The integral
        :rtype: jnp.array
        """
        return factor.intergate_log_factor(self)

@dataclass(kw_only=True)
class GaussianDiagMeasure(GaussianMeasure):
    r"""A measure with a Gaussian form.

    .. math::
    
        u(X) = \beta * \exp(- 0.5 * X^\top\Lambda X + X^\top\nu),
\
    D is the dimension, and R the number of Gaussians. 

    :param Lambda: Information (precision) matrix of the Gaussian distributions. Needs to be postive definite and diagonal.
        Dimensions should be [R, D, D].
    :type Lambda: jnp.ndarray
    :param nu: Information vector of a Gaussian distribution. If None all zeros. Dimensions should be [R, D], 
        defaults to None
    :type nu: jnp.ndarray, optional
    :param ln_beta: The log constant factor of the factor. If None all zeros. Dimensions should be [R], 
        defaults to None
    :type ln_beta: jnp.ndarray, optional
    :param Sigma: Covariance matrix of the Gaussian distributions. Needs to be positive definite. 
        Dimensions should be [R, D, D], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param ln_det_Lambda: Log determinant of Lambda. Dimensions should be [R], defaults to None
    :type ln_det_Lambda: jnp.ndarray, optional
    :param ln_det_Sigma: Log determinant of Sigma. Dimensions should be [R], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    """

    def invert_lambda(self):
        self.Sigma, self.ln_det_Lambda = invert_diagonal(self.Lambda)
        self.ln_det_Sigma = -self.ln_det_Lambda

    def slice(self, indices: jnp.ndarray) -> "GaussianDiagMeasure":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting Gaussian diagonal measure.
        :rtype: GaussianDiagMeasure
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        nu_new = jnp.take(self.nu, indices, axis=0)
        ln_beta_new = jnp.take(self.ln_beta, indices, axis=0)
        new_measure = GaussianDiagMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if self.Sigma is not None:
            new_measure.Sigma = jnp.take(self.Sigma, indices, axis=0)
            new_measure.ln_det_Sigma = jnp.take(self.ln_det_Sigma, indices, axis=0)
            new_measure.ln_det_Lambda = jnp.take(self.ln_det_Lambda, indices, axis=0)
        return new_measure

    def product(self) -> "GaussianDiagMeasure":
        """Computes the product over all factor.
        
        .. math::
        
            v(X) = \prod_i u_i(X)

        :return: Factor of all factor.
        :rtype: GaussianDiagMeasure
        """
        Lambda_new = jnp.sum(self.Lambda, axis=0, keepdims=True)
        nu_new = jnp.sum(self.nu, axis=0, keepdims=True)
        ln_beta_new = jnp.sum(self.ln_beta, axis=0, keepdims=True)
        new_measure = GaussianDiagMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if self.Sigma is not None:
            new_measure._prepare_integration()
        return new_measure
