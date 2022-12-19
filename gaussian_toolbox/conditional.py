##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for conditional Gaussian densities, that can be seen as          #
# operators.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
__all__ = ["ConditionalGaussianPDF", "NNControlGaussianConditional"]

# import jnp
# from pdf import GaussianPDF
from jax import numpy as jnp
from typing import Tuple
from . import pdf, factor, measure
from .utils.linalg import invert_matrix, invert_diagonal

from .utils.dataclass import dataclass
from dataclasses import field

@dataclass(kw_only=True)
class ConditionalGaussianPDF:
    """A conditional Gaussian density

    .. math::

        p(Y|X) = {\cal N}(\mu(X), \Sigma),

    with the conditional mean function :math:`\mu(X) = M X + b`.

    :param M: Matrix in the mean function. Dimensions should be [R, Dy, Dx]
    :type M: jnp.ndarray
    :param b: Vector in the conditional mean function. If None all entries are 0. Dimensions should be [R, Dy], defaults to None
    :type b: jnp.ndarray, optional
    :param Sigma: The covariance matrix of the conditional. Dimensions should be [R, Dy, Dy], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [R, Dy, Dy],
        defaults to None
    :type Lambda: jnp.ndarray, optional
    :param ln_det_Sigma: Log determinant of the covariance matrix. Dimensions should be [R], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    :raises RuntimeError: Raised if neiterh Sigma nor Lambda are provided
    """
    M: jnp.ndarray
    b: jnp.ndarray = None
    Sigma: jnp.ndarray = None
    Lambda: jnp.ndarray = None
    ln_det_Sigma: jnp.ndarray = None
    
    def __post_init__(self):
        if self.b is None:
            self.b = jnp.zeros((self.R, self.Dy))
        if self.Sigma is None and self.Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif self.Sigma is not None:
            if self.Lambda is None or self.ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        else:
            self.Sigma, ln_det_Lambda = invert_matrix(self.Lambda)
            self.ln_det_Sigma = -ln_det_Lambda
        
    @property
    def R(self):
        return self.M.shape[0]
    
    @property
    def Dy(self):
        return self.M.shape[1]
    
    @property
    def Dx(self):
        return self.M.shape[2]

    def __str__(self) -> str:
        return "Conditional Gaussian density p(y|x)"

    def __call__(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :amt:`x`, i.e.

        .. math::

            p(Y\\vert X=x) =  {\cal N}(\mu(X=x), \Sigma)

        :param x: Instances, the :math:`\mu` should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on x.
        :rtype: pdf.GaussianPDF
        """
        return self.condition_on_x(x)

    def slice(self, indices: jnp.ndarray) -> "ConditionalGaussianPDF":
        """Return the conditional with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting conditional Gaussian diagonal density.
        :rtype: ConditionalGaussianPDF
        """
        M_new = jnp.take(self.M, indices, axis=0)
        b_new = jnp.take(self.b, indices, axis=0)
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = ConditionalGaussianPDF(
            M=M_new, b=b_new, Sigma=Sigma_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        return new_measure

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the conditional :math:`\mu` function :math:`\mu(X=x) = M x + b`.

        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional means. Dimensions are [R, N, Dy].
        :rtype: jnp.ndarray
        """
        mu_y = jnp.einsum("abc,dc->adb", self.M, x) + self.b[:, None]
        return mu_y

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :math:`x`.

        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on :math:`x`.
        :rtype: pdf.GaussianPDF
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((self.R * N, self.Dy))
        Sigma_new = jnp.tile(self.Sigma[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        Lambda_new = jnp.tile(self.Lambda[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        ln_det_Sigma_new = jnp.tile(self.ln_det_Sigma[:, None], (1, N)).reshape(
            self.R * N
        )
        return pdf.GaussianPDF(
            Sigma=Sigma_new,
            mu=mu_new,
            Lambda=Lambda_new,
            ln_det_Sigma=ln_det_Sigma_new,
        )

    def set_y(self, y: jnp.ndarray, **kwargs) -> factor.ConjugateFactor:
        """Set a specific value for :math:`y` in :math:`p(Y=y|X)` and returns the corresponding conjugate factor.

        :param y: Data for :math:`y`, where the rth entry is associated with the rth conditional density.
        :type y: jnp.ndarray [R, Dy]
        :return: The conjugate factor where the first dimension is R.
        :rtype: factor.ConjugateFactor
        """
        try:
            assert self.R == 1 or y.shape[0] == self.R
        except AssertionError:
            raise RuntimeError(
                "Either R should be one or the leading dimension should be equal to R"
            )

        y_minus_b = y - self.b
        Lambda_new = jnp.einsum(
            "abc,acd->abd",
            jnp.einsum("abd, abc -> adc", self.M, self.Lambda),
            self.M,
        )
        if self.R == 1:
            Lambda_new = jnp.tile(Lambda_new, (y.shape[0], 1, 1))
        nu_new = jnp.einsum(
            "abc, ab -> ac",
            jnp.einsum("abc, acd -> abd", self.Lambda, self.M),
            y_minus_b,
        )
        yb_Lambda_yb = jnp.einsum(
            "ab, ab-> a",
            jnp.einsum("ab, abc -> ac", y_minus_b, self.Lambda),
            y_minus_b,
        )
        ln_beta_new = -0.5 * (
            yb_Lambda_yb + self.Dx * jnp.log(2 * jnp.pi) + self.ln_det_Sigma
        )
        factor_new = factor.ConjugateFactor(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        return factor_new

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Return the joint density.

        .. math::

            p(X,Y) = p(Y|X)p(X),

        where :math:`p(Y|X)` is the object itself.

        :param p_x: Marginal density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :raises RuntimeError: Only works if one of the densities involved have R==1.
        :return: The joint density.
        :rtype: pdf.GaussianPDF
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditional is not implemented."
            )
        R = p_x.R * self.R
        D_xy = p_x.D + self.Dy
        # Mean
        mu_x = jnp.tile(
            p_x.mu[None],
            (
                self.R,
                1,
                1,
            ),
        ).reshape((R, p_x.D))
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        mu_xy = jnp.hstack([mu_x, mu_y])
        # Sigma
        Sigma_x = jnp.tile(p_x.Sigma[None], (self.R, 1, 1, 1)).reshape(R, p_x.D, p_x.D)
        MSigma_x = jnp.einsum("abc,dce->adbe", self.M, p_x.Sigma)  # [R1,R,Dy,D]
        MSigmaM = jnp.einsum("abcd,aed->abce", MSigma_x, self.M)
        Sigma_y = (self.Sigma[:, None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        C_xy = MSigma_x.reshape((R, self.Dy, p_x.D))
        Sigma_xy = jnp.block([[Sigma_x, jnp.swapaxes(C_xy, 1, 2)], [C_xy, Sigma_y]])
        # Sigma_xy = jnp.empty((R, D_xy, D_xy))
        # Sigma_xy[:,:p_x.D,:p_x.D] = Sigma_x
        # Sigma_xy[:,p_x.D:,p_x.D:] = Sigma_y
        # Sigma_xy[:,p_x.D:,:p_x.D] = C_xy
        # Sigma_xy[:,:p_x.D,p_x.D:] = jnp.swapaxes(C_xy, 1, 2)
        # Lambda
        Lambda_y = jnp.tile(self.Lambda[:, None], (1, p_x.R, 1, 1)).reshape(
            (R, self.Dy, self.Dy)
        )
        Lambda_yM = jnp.einsum("abc,abd->acd", self.Lambda, self.M)  # [R1,Dy,D]
        MLambdaM = jnp.einsum("abc,abd->acd", self.M, Lambda_yM)
        Lambda_x = (p_x.Lambda[None] + MLambdaM[:, None]).reshape((R, p_x.D, p_x.D))
        L_xy = jnp.tile(-Lambda_yM[:, None], (1, p_x.R, 1, 1)).reshape(
            (R, self.Dy, p_x.D)
        )
        Lambda_xy = jnp.block([[Lambda_x, jnp.swapaxes(L_xy, 1, 2)], [L_xy, Lambda_y]])
        # Log determinant
        if p_x.D > self.Dy:
            CLambda_x = jnp.einsum(
                "abcd,bde->abce", MSigma_x, p_x.Lambda
            )  # [R1,R,Dy,D]
            CLambdaC = jnp.einsum(
                "abcd,abed->abce", CLambda_x, MSigma_x
            )  # [R1,R,Dy,Dy]
            delta_ln_det = jnp.linalg.slogdet(Sigma_y[:, None] - CLambdaC)[1].reshape(
                (R,)
            )
            ln_det_Sigma_xy = p_x.ln_det_Sigma + delta_ln_det
        else:
            # [R1,Dy,Dy] x [R1, Dy, D] = [R1, Dy, D]
            Sigma_yL = jnp.einsum("abc,acd->abd", self.Sigma, -Lambda_yM)
            # [R1, Dy, D] x [R1, Dy, D] = [R1, D, D]
            LSigmaL = jnp.einsum("abc,abd->acd", -Lambda_yM, Sigma_yL)
            LSigmaL = jnp.tile(LSigmaL[:, None], (1, p_x.R)).reshape((R, p_x.D, p_x.D))
            delta_ln_det = jnp.linalg.slogdet(Lambda_x - LSigmaL)[1]
            ln_det_Sigma_xy = -(
                jnp.tile(-self.ln_det_Sigma[:, None], (1, p_x.R)).reshape((R,))
                + delta_ln_det
            )
        return pdf.GaussianPDF(Sigma=Sigma_xy, mu=mu_xy, Lambda=Lambda_xy, ln_det_Sigma=ln_det_Sigma_xy)

    def affine_marginal_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Return the marginal density :math:`p(X)` given :math:`p(Y|X)` and :math:`p(X)`, where :math:`p(Y|X)`
        is the object itself.

        :param p_x: Marginal density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :raises RuntimeError: Only works if one of the densities involved have R==1.
        :return: The marginal density.
        :rtype: pdf.GaussianPDF
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditional is not implemented."
            )
        R = p_x.R * self.R
        # Mean
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        # Sigma
        MSigma_x = jnp.einsum("abc,dce->adbe", self.M, p_x.Sigma)  # [R1,R,Dy,D]
        MSigmaM = jnp.einsum("abcd,aed->abce", MSigma_x, self.M)
        Sigma_y = (self.Sigma[:, None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        return pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)

    def affine_conditional_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> "ConditionalGaussianPDF":
        """Return the conditional density :math:`p(X|Y)`, given :math:`p(Y|X)` and :math:`p(X)`, where :math:`p(Y|X)`
        is the object itself.

        :param p_x: Marginal density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :raises RuntimeError: Only works if one of the densities involved have R==1.
        :return: The conditional density :math:`p(Y|X)`.
        :rtype: ConditionalGaussianPDF
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditional is not implemented."
            )
        R = p_x.R * self.R
        # TODO: Could be flexibly made more effiecient here.
        # Marginal Sigma y
        # MSigma_x = jnp.einsum('abc,dce->adbe', self.M, p_xSigma) # [R1,R,Dy,D]
        # MSigmaM = jnp.einsum('abcd,aed->abce', MSigma_x, self.M)
        # Sigma_y = (self.Sigma[:,None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        # Lambda_y, ln_det_Sigma_y = p_x.invert_matrix(Sigma_y)
        # Lambda
        Lambda_yM = jnp.einsum("abc,abd->acd", self.Lambda, self.M)  # [R1,Dy,D]
        MLambdaM = jnp.einsum("abc,abd->acd", self.M, Lambda_yM)
        Lambda_x = (p_x.Lambda[None] + MLambdaM[:, None]).reshape((R, p_x.D, p_x.D))
        # Sigma
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        # M_x
        M_Lambda_y = jnp.einsum("abc,abd->acd", self.M, self.Lambda)  # [R1, D, Dy]
        M_x = jnp.einsum(
            "abcd,ade->abce", Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), M_Lambda_y
        )  # [R1, R, D, Dy]
        # [R1, R, D, Dy] x [R1, Dy] = [R1, R, D]
        b_x = -jnp.einsum("abcd,ad->abc", M_x, self.b)
        b_x += jnp.einsum(
            "abcd,bd->abc", Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), p_x.nu
        )
        b_x = b_x.reshape((R, p_x.D))
        M_x = M_x.reshape((R, p_x.D, self.Dy))
        return ConditionalGaussianPDF(
            M=M_x,
            b=b_x,
            Sigma=Sigma_x,
            Lambda=Lambda_x,
            ln_det_Sigma=-ln_det_Lambda_x,
        )

    def integrate_log_conditional(self, p_yx: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Integrates over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.

        .. math::

            \int \log(p(Y|X))p(Y,X){\\rm d}Y{\\rm d}X.

        :param p_yx: Probability density function (first dimensions are :math:`Y`, last ones are :math:`X`).
        :type p_yx: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density :math:`p(Y,X)`.
        :rtype: jnp.ndarray
        """
        if self.R != 1 and self.R != p_yx.R:
            raise NotImplementedError("Only implemented for R=1.")
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(-self.M)
        b = -self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = p_yx.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        log_expectation = -0.5 * (
            quadratic_integral + (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        return log_expectation

    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: jnp.ndarray=None, **kwargs) -> callable:
        """Computes the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

           f(Y) = \int \log(p(Y|X))p(X){\\rm d}x.

        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = self.M
        b = self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = p_x.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde)
        log_expectation_constant = -0.5 * (
            quadratic_integral + (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_integral)
            + log_expectation_constant
        )
        if y == None:
            return log_expectation_y
        else:
            return log_expectation_y(y)

    def conditional_entropy(self, p_x: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Computes the conditional entropy

        .. math::

            H_{Y|X} = H_{Y,X} - H_X = -\int p(X,Y)\ln p(Y|X) {\\rm d}X {\\rm d}Y

        :param p_x: Marginal over condtional variable
        :type p_x: pdf.GaussianPDF
        :return: Conditional entropy
        :rtype: jnp.ndarray [R]
        """
        p_xy = self.affine_joint_transformation(p_x)
        cond_entropy = p_xy.entropy() - p_x.entropy()
        return cond_entropy

    def mutual_information(self, p_x: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Computes the mutual information

        .. math::

            I_{Y,X} = H_{Y,X} - H_X - H_Y

        :param p_x: Marginal over condtional variable
        :type p_x: pdf.GaussianPDF
        :return: Mututal information
        :rtype: jnp.ndarray [R]
        """
        cond_entropy = self.conditional_entropy(p_x, **kwargs)
        p_y = self.affine_marginal_transformation(p_x, **kwargs)
        mutual_info = cond_entropy - p_y.entropy()
        return mutual_info

    def update_Sigma(self, Sigma_new: jnp.ndarray):
        """Updates the covariance matrix :math:`\Sigma`.

        :param Sigma_new: The new covariance matrix.
        :type Sigma_new: jnp.ndarray [R, Dy, Dy]
        :raises ValueError: Raised when dimension of old and new covariance do not match.
        """
        if self.Sigma.shape != Sigma_new.shape:
            raise ValueError("Dimensions of the new Sigma don't match.")
        self.Sigma = Sigma_new
        self.Lambda, self.ln_det_Sigma = invert_matrix(Sigma_new)

@dataclass(kw_only=True)
class ConditionalGaussianDiagPDF(ConditionalGaussianPDF):
    """A conditional Gaussian density

    .. math::

        p(Y|X) = {\cal N}(\mu(X), \Sigma),

    with the conditional mean function :math:`\mu(X) = M X + b` and :math:`\Sigma` is diagonal.

    :param M: Matrix in the mean function. Dimensions should be [R, Dy, Dx]
    :type M: jnp.ndarray
    :param b: Vector in the conditional mean function. If None all entries are 0. Dimensions should be [R, Dy], defaults to None
    :type b: jnp.ndarray, optional
    :param Sigma: The (diagonal) covariance matrix of the conditional. Dimensions should be [R, Dy, Dy], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param Lambda: (Diagonal) Information (precision) matrix of the Gaussians. Dimensions should be [R, Dy, Dy],
        defaults to None
    :type Lambda: jnp.ndarray, optional
    :param ln_det_Sigma: Log determinant of the covariance matrix. Dimensions should be [R], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    :raises RuntimeError: Raised if neiterh Sigma nor Lambda are provided
    """
    
    def __post_init__(self):
        if self.b is None:
            self.b = jnp.zeros((self.R, self.Dy))
        if self.Sigma is None and self.Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif self.Sigma is not None:
            if self.Lambda is None or self.ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = invert_diagonal(self.Sigma)
        else:
            self.Sigma, ln_det_Lambda = invert_diagonal(self.Lambda)
            self.ln_det_Sigma = -ln_det_Lambda

@dataclass(kw_only=True)
class NNControlGaussianConditional(ConditionalGaussianPDF):
    """A conditional Gaussian density, where the transition model is determined through a (known) control variable u.

        .. math::

            p(Y|X, u) = N(\mu(X|u), \Sigma)

        with the conditional mean function ::math:`\mu(X|u) = M(u) X + b(u)`,

        where :math:`M(u)` and :math:`b(u)` come from the same neural network.

    :param Sigma: Covariance matrix [1, Dy, Dy]
    :type Sigma: jnp.ndarray
    :param Dx: Dimension of the conditional variable.
    :type Dx: int
    :param Du: Dimension of the control variable
    :type Du: int
    :param hidden_units: Determines how many hidden layers and how many units in each layer, defaults to [16,]
    :type hidden_units: list, optional
    :param non_linearity: Non linearity after each layer, defaults to objax.functional.tanh
    :type non_linearity: callable, optional
    :raises NotImplementedError: Raised when the leading dimension of Sigma is not 1.
    """
    Sigma: jnp.ndarray
    num_cond_dim: int
    num_control_dim: int
    control_func: callable
    M: jnp.ndarray = field(init=False)

    def __post_init__(
        self,

    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        dummy_input = jnp.zeros([1, self.Du])
        dummy_output = self.control_func(dummy_input)
        assert dummy_output.shape == (1, self.Dy * (self.Dx + 1))
        
    @property
    def R(self):
        return self.Sigma.shape[0]
    
    @property
    def Dy(self):
        return self.Sigma.shape[1]
    
    @property
    def Du(self):
        return self.num_control_dim
    
    @property
    def Dx(self):
        return self.num_cond_dim
    
    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :amt:`x`, i.e.

        .. math::

            p(Y\\vert X=x) =  {\cal N}(\mu(X=x), \Sigma)

        :param x: Instances, the :math:`\mu` should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on x.
        :rtype: pdf.GaussianPDF
        """
        return self.condition_on_x_u(x, u)

    def get_M_b(self, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Construct :math:`M(u)` and :math:`b(u)` from the output.

        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Returns :math:`M(u)` [R, Dy, Dx] and :math:`b(u)` [R, Dy]
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        output = self.control_func(u)
        M = output[:, : self.Dy * self.Dx].reshape((-1, self.Dy, self.Dx))
        b = output[:, self.Dy * self.Dx :]
        return M, b

    def set_control_variable(self, u: jnp.ndarray) -> ConditionalGaussianPDF:
        """Create the conditional for a given control variable u,

        .. math::

            p(Y|X, u).

        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: The conditional
        :rtype: ConditionalGaussianPDF
        """
        R = u.shape[0]
        M, b = self.get_M_b(u)
        tile_dims = (R, 1, 1)
        return ConditionalGaussianPDF(
            M=M,
            b=b,
            Sigma=jnp.tile(self.Sigma, tile_dims),
            Lambda=jnp.tile(self.Lambda, tile_dims),
            ln_det_Sigma=jnp.tile(self.ln_det_Sigma, (R,)),
        )

    def get_conditional_mu(self, x: jnp.ndarray, u: jnp.array, **kwargs) -> jnp.ndarray:
        """Compute the conditional mean given an :math:`x` and an :math:`u`,

        .. math::

            \mu(X=x|u) = M(u)x + b(u)

        :param x: Conditional variable [N, Dx]
        :type x: jnp.ndarray
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Conditional mean [R, N, Dy]
        :rtype: jnp.ndarray
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.get_conditional_mu(x)

    def condition_on_x_u(self, x: jnp.ndarray, u: jnp.array, **kwargs) -> pdf.GaussianPDF:
        """Return the Gaussian density

        .. math::

            p(Y|X=x, u)

        :param x: Conditional variable [N, Dx]
        :type x: jnp.ndarray
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Gaussian density conditioned on instances x, and u.
        :rtype: pdf.GaussianPDF
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.condition_on_x(x)

    def set_y(self, y: jnp.ndarray, u: jnp.array, **kwargs) -> factor.ConjugateFactor:
        """Set an instance of Y and U and returns

        .. math:

            p(Y=y|X, u)

        :param y: Random variable [R, Dy]
        :type y: jnp.ndarray
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: The factor with the instantiation.
        :rtype: factor.ConjugateFactor
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.set_y(y)

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, u: jnp.array, **kwargs
    ) -> pdf.GaussianPDF:
        """Perform the affine joint transformation with a given control variable

        .. math::

            p(X,Y|u) = p(Y|X,u)p(X),

        where :math:`p(Y|X,u)` is the object itself.

        :param p_x: Marginal over :math:`X`
        :type p_x: pdf.GaussianPDF
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: The joint density
        :rtype: pdf.GaussianPDF
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.affine_joint_transformation(p_x)

    def affine_marginal_transformation(
        self, p_x: pdf.GaussianPDF, u: jnp.array, **kwargs
    ) -> pdf.GaussianPDF:
        """Return the marginal density :math:`p(Y)` given  :math:`p(Y|X,u)` and :math:`p(X)`,
        where p(Y|X,u) is the object itself.

        :param p_x: Marginal over :math:`X`
        :type p_x: pdf.GaussianPDF
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Marginal density :math:`p(Y|u)`.
        :rtype: pdf.GaussianPDF
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.affine_marginal_transformation(p_x)

    def affine_conditional_transformation(
        self, p_x: pdf.GaussianPDF, u: jnp.array, **kwargs
    ) -> "ConditionalGaussianPDF":
        """Return the conditional density :math:`p(X|Y, u)`, given :math:`p(Y|X,u)` and :math:`p(X)`,
        where :math:`p(Y|X,u)` is the object itself.

        :param p_x: Marginal over :math:`X`
        :type p_x: pdf.GaussianPDF
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Conditional density :math:`p(X|Y, u)`.
        :rtype: ConditionalGaussianPDF
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.affine_conditional_transformation(p_x)

    def conditional_entropy(
        self, p_x: pdf.GaussianPDF, u: jnp.array, **kwargs
    ) -> jnp.ndarray:
        """Compute the conditional entropy

        .. math:

            H_{y|x,u} = H_{y,x|u} - H_x = -\int p(X,Y\\vert u)\ln p(Y|X,u) {\\rm d}x {\\rm d}y

        :param p_x: Marginal over condtional variable
        :type p_x: pdf.GaussianPDF
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Conditional entropy
        :rtype: jnp.ndarray [R]
        """
        p_xy = self.affine_joint_transformation(p_x, u)
        cond_entropy = p_xy.entropy() - p_x.entropy()
        return cond_entropy

    def integrate_log_conditional(
        self, phi_yx: measure.GaussianMeasure, u: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Integrate over the log conditional with respect to the pdf :math:`p(Y,X)`, i.e.

        .. math::

            \int \log(p(Y|X,u))p(Y,X){\\rm d}Y{\\rm d}X.

        :param p_yx: Probability density function (first dimensions are :math:`Y`, last ones are :math:`X`).
        :type p_yx: measure.GaussianMeasure
        :param u: Control variables [1, Du]
        :type u: jnp.ndarray
        :raises NotImplementedError: Only one network input allowed.
        :return: Returns the integral with respect to density :math:`p(Y,X)`.
        :rtype: jnp.ndarray
        """
        if u.shape[0] != 1:
            raise NotImplementedError("Only implemented for a single input.")
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.integrate_log_conditional(phi_yx)

    def integrate_log_conditional_y(
        self, phi_x: measure.GaussianMeasure, u: jnp.ndarray, y: jnp.ndarray=None, **kwargs
    ) -> callable:
        """Computes the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

           f(Y) = \int \log(p(Y|X,u))p(X)dX.

        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianMeasure
        :param u: Control variables [1, Du]
        :type u: jnp.ndarray
        :raises NotImplementedError: Only one network input allowed.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        if u.shape[0] != 1:
            raise NotImplementedError("Only implemented for a single input.")

        cond_gauss = self.set_control_variable(u)
        return cond_gauss.integrate_log_conditional_y(phi_x, y=y, **kwargs)

@dataclass(kw_only=True)
class ConditionalIdentityGaussianPDF(ConditionalGaussianPDF):
    """A conditional Gaussian density

    .. math::

        p(Y|X) = {\cal N}(\mu(X), \Sigma),

    with the conditional mean function :math:`\mu(X) = X`.

    :param Sigma: The covariance matrix of the conditional. Dimensions should be [R, Dy, Dy], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [R, Dy, Dy],
        defaults to None
    :type Lambda: jnp.ndarray, optional
    :param ln_det_Sigma: Log determinant of the covariance matrix. Dimensions should be [R], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    :raises RuntimeError: Raised if neiterh Sigma nor Lambda are provided
    """
    Sigma: jnp.ndarray = None
    Lambda: jnp.ndarray = None
    ln_det_Sigma: jnp.ndarray = None
    num_cond_dim: int = field(init=False)
    num_control_dim: int = field(init=False)
    M: jnp.ndarray = field(init=False)
    b: jnp.ndarray = field(init=False)

    def __post_init__(
        self,
    ):
        if self.Sigma is None and self.Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif self.Sigma is not None:
            if self.Lambda is None or self.ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        else:
            self.Sigma, ln_det_Lambda = invert_matrix(self.Lambda)
            self.ln_det_Sigma = -ln_det_Lambda
      
    
    @property
    def R(self):
        return self.Sigma.shape[0]
    
    @property
    def Dx(self):
        return self.Sigma.shape[1]
    
    @property
    def Dy(self):
        return self.Sigma.shape[1]

    def __str__(self) -> str:
        return "Conditional Gaussian density p(y|x)"

    def __call__(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :amt:`x`, i.e.

        .. math::

            p(Y\\vert X=x) =  {\cal N}(\mu(X=x), \Sigma)

        :param x: Instances, the :math:`\mu` should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on x.
        :rtype: pdf.GaussianPDF
        """
        return self.condition_on_x(x)

    def slice(self, indices: jnp.ndarray) -> "ConditionalGaussianPDF":
        """Return the conditional with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting conditional Gaussian diagonal density.
        :rtype: ConditionalGaussianPDF
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = ConditionalIdentityGaussianPDF(
            Sigma=Sigma_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        return new_measure

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the conditional :math:`\mu` function :math:`\mu(X=x) = M x + b`.

        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional means. Dimensions are [R, N, Dy].
        :rtype: jnp.ndarray
        """
        mu_y = jnp.tile(x[None], (self.R, 1, 1))
        return mu_y

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :math:`x`.

        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on :math:`x`.
        :rtype: pdf.GaussianPDF
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((self.R * N, self.Dy))
        Sigma_new = jnp.tile(self.Sigma[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        Lambda_new = jnp.tile(self.Lambda[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        ln_det_Sigma_new = jnp.tile(self.ln_det_Sigma[:, None], (1, N)).reshape(
            self.R * N
        )
        return pdf.GaussianPDF(
            Sigma=Sigma_new,
            mu=mu_new,
            Lambda=Lambda_new,
            ln_det_Sigma=ln_det_Sigma_new,
        )

    def set_y(self, y: jnp.ndarray, **kwargs) -> factor.ConjugateFactor:
        """Set a specific value for :math:`y` in :math:`p(Y=y|X)` and returns the corresponding conjugate factor.

        :param y: Data for :math:`y`, where the rth entry is associated with the rth conditional density.
        :type y: jnp.ndarray [R, Dy]
        :return: The conjugate factor where the first dimension is R.
        :rtype: factor.ConjugateFactor
        """
        try:
            assert self.R == 1 or y.shape[0] == self.R
        except AssertionError:
            raise RuntimeError(
                "Either R should be one or the leading dimension should be equal to R"
            )
        nu_new = jnp.einsum(
            "abc, ab -> ac",
            self.Lambda,
            y,
        )
        y_Lambda_y = jnp.einsum(
            "ab, ab-> a",
            jnp.einsum("ab, abc -> ac", y, self.Lambda),
            y,
        )
        ln_beta_new = -0.5 * (
            y_Lambda_y + self.Dx * jnp.log(2 * jnp.pi) + self.ln_det_Sigma
        )
        factor_new = factor.ConjugateFactor(Lambda=self.Lambda, nu=nu_new, ln_beta=ln_beta_new)
        return factor_new

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Return the joint density.

        .. math::

            p(X,Y) = p(Y|X)p(X),

        where :math:`p(Y|X)` is the object itself.

        :param p_x: Marginal density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :raises RuntimeError: Only works if one of the densities involved have R==1.
        :return: The joint density.
        :rtype: pdf.GaussianPDF
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditional is not implemented."
            )
        R = p_x.R * self.R
        D_xy = p_x.D + self.Dy
        # Mean
        mu_x = jnp.tile(
            p_x.mu[None],
            (
                self.R,
                1,
                1,
            ),
        ).reshape((R, p_x.D))
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        mu_xy = jnp.hstack([mu_x, mu_y])
        # Sigma
        Sigma_x = jnp.tile(p_x.Sigma[None], (self.R, 1, 1, 1)).reshape(R, p_x.D, p_x.D)
        Sigma_y = (self.Sigma[:, None] + p_x.Sigma).reshape((R, self.Dy, self.Dy))
        C_xy = p_x.Sigma
        Sigma_xy = jnp.block([[Sigma_x, jnp.swapaxes(C_xy, 1, 2)], [C_xy, Sigma_y]])
        # Sigma_xy = jnp.empty((R, D_xy, D_xy))
        # Sigma_xy[:,:p_x.D,:p_x.D] = Sigma_x
        # Sigma_xy[:,p_x.D:,p_x.D:] = Sigma_y
        # Sigma_xy[:,p_x.D:,:p_x.D] = C_xy
        # Sigma_xy[:,:p_x.D,p_x.D:] = jnp.swapaxes(C_xy, 1, 2)
        # Lambda
        Lambda_y = jnp.tile(self.Lambda[:, None], (1, p_x.R, 1, 1)).reshape(
            (R, self.Dy, self.Dy)
        )
        Lambda_x = (p_x.Lambda[None] + self.Lambda[:, None]).reshape((R, p_x.D, p_x.D))
        L_xy = jnp.tile(-self.Lambda[:, None], (1, p_x.R, 1, 1)).reshape(
            (R, self.Dy, p_x.D)
        )
        Lambda_xy = jnp.block([[Lambda_x, jnp.swapaxes(L_xy, 1, 2)], [L_xy, Lambda_y]])
        # Log determinant
        if p_x.D > self.Dy:
            CLambdaC = jnp.tile(
                p_x.Sigma[None], (R, self.R, p_x.D, p_x.D)
            )  # [R1,R,Dy,Dy]
            delta_ln_det = jnp.linalg.slogdet(Sigma_y[:, None] - CLambdaC)[1].reshape(
                (R,)
            )
            ln_det_Sigma_xy = p_x.ln_det_Sigma + delta_ln_det
        else:
            # [R1, Dy, D] x [R1, Dy, D] = [R1, D, D]
            LSigmaL = jnp.tile(self.Lambda[:, None], (1, p_x.R)).reshape(
                (R, p_x.D, p_x.D)
            )
            delta_ln_det = jnp.linalg.slogdet(Lambda_x - LSigmaL)[1]
            ln_det_Sigma_xy = -(
                jnp.tile(-self.ln_det_Sigma[:, None], (1, p_x.R)).reshape((R,))
                + delta_ln_det
            )
        return pdf.GaussianPDF(Sigma=Sigma_xy, mu=mu_xy, Lambda=Lambda_xy, ln_det_Sigma=ln_det_Sigma_xy)

    def affine_marginal_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Return the marginal density :math:`p(X)` given :math:`p(Y|X)` and :math:`p(X)`, where :math:`p(Y|X)`
        is the object itself.

        :param p_x: Marginal density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :raises RuntimeError: Only works if one of the densities involved have R==1.
        :return: The marginal density.
        :rtype: pdf.GaussianPDF
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditional is not implemented."
            )
        R = p_x.R * self.R
        # Mean
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        # Sigma
        Sigma_y = (self.Sigma[:, None] + p_x.Sigma[:, None]).reshape(
            (R, self.Dy, self.Dy)
        )
        return pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)

    def affine_conditional_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> "ConditionalGaussianPDF":
        """Return the conditional density :math:`p(X|Y)`, given :math:`p(Y|X)` and :math:`p(X)`, where :math:`p(Y|X)`
        is the object itself.

        :param p_x: Marginal density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :raises RuntimeError: Only works if one of the densities involved have R==1.
        :return: The conditional density :math:`p(Y|X)`.
        :rtype: ConditionalGaussianPDF
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditional is not implemented."
            )
        R = p_x.R * self.R
        # TODO: Could be flexibly made more effiecient here.
        # Marginal Sigma y
        # MSigma_x = jnp.einsum('abc,dce->adbe', self.M, p_xSigma) # [R1,R,Dy,D]
        # MSigmaM = jnp.einsum('abcd,aed->abce', MSigma_x, self.M)
        # Sigma_y = (self.Sigma[:,None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        # Lambda_y, ln_det_Sigma_y = p_x.invert_matrix(Sigma_y)
        # Lambda
        Lambda_x = (p_x.Lambda[None] + self.Lambda[:, None]).reshape((R, p_x.D, p_x.D))
        # Sigma
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        M_x = jnp.einsum(
            "abcd,ade->abce",
            Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)),
            self.Lambda,
        )  # [R1, R, D, Dy]
        # [R1, R, D, Dy] x [R1, Dy] = [R1, R, D]
        b_x = jnp.einsum(
            "abcd,bd->abc", Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), p_x.nu
        )
        b_x = b_x.reshape((R, p_x.D))
        M_x = M_x.reshape((R, p_x.D, self.Dy))
        return ConditionalGaussianPDF(
            M=M_x,
            b=b_x,
            Sigma=Sigma_x,
            Lambda=Lambda_x,
            ln_det_Sigma=-ln_det_Lambda_x,
        )

    def integrate_log_conditional(self, p_yx: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Integrates over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.

        .. math::

            \int \log(p(Y|X))p(Y,X){\\rm d}Y{\\rm d}X.

        :param p_yx: Probability density function (first dimensions are :math:`Y`, last ones are :math:`X`).
        :type p_yx: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density :math:`p(Y,X)`.
        :rtype: jnp.ndarray
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(jnp.eye(self.Dy, self.Dy)[None])
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        quadratic_integral = p_yx.integrate("(Ax+a)'(Bx+b)", A_mat=A, B_mat=A_tilde)
        log_expectation = -0.5 * (
            quadratic_integral + (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        return log_expectation

    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: jnp.ndarray=None, **kwargs) -> callable:
        """Computes the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

           f(Y) = \int \log(p(Y|X))p(X){\\rm d}x.

        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = jnp.tile(jnp.eye(self.Dy)[None], (self.R, 1, 1))
        A_tilde = self.Lambda
        quadratic_integral = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=A, B_mat=A_tilde)
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde)
        log_expectation_constant = -0.5 * (
            quadratic_integral + (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_integral)
            + log_expectation_constant
        )
        if y == None:
            return log_expectation_y
        else:
            return log_expectation_y(y)

    def conditional_entropy(self, p_x: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Computes the conditional entropy

        .. math::

            H_{Y|X} = H_{Y,X} - H_X = -\int p(X,Y)\ln p(Y|X) {\\rm d}X {\\rm d}Y

        :param p_x: Marginal over condtional variable
        :type p_x: pdf.GaussianPDF
        :return: Conditional entropy
        :rtype: jnp.ndarray [R]
        """
        p_xy = self.affine_joint_transformation(p_x)
        cond_entropy = p_xy.entropy() - p_x.entropy()
        return cond_entropy

    def mutual_information(self, p_x: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Computes the mutual information

        .. math::

            I_{Y,X} = H_{Y,X} - H_X - H_Y

        :param p_x: Marginal over condtional variable
        :type p_x: pdf.GaussianPDF
        :return: Mututal information
        :rtype: jnp.ndarray [R]
        """
        cond_entropy = self.conditional_entropy(p_x, **kwargs)
        p_y = self.affine_marginal_transformation(p_x, **kwargs)
        mutual_info = cond_entropy - p_y.entropy()
        return mutual_info

    def update_Sigma(self, Sigma_new: jnp.ndarray):
        """Updates the covariance matrix :math:`\Sigma`.

        :param Sigma_new: The new covariance matrix.
        :type Sigma_new: jnp.ndarray [R, Dy, Dy]
        :raises ValueError: Raised when dimension of old and new covariance do not match.
        """
        if self.Sigma.shape != Sigma_new.shape:
            raise ValueError("Dimensions of the new Sigma don't match.")
        self.Sigma = Sigma_new
        self.Lambda, self.ln_det_Sigma = invert_matrix(Sigma_new)

@dataclass(kw_only=True)
class ConditionalIdentityDiagGaussianPDF(ConditionalIdentityGaussianPDF):
    """A conditional Gaussian density

    .. math::

        p(Y|X) = {\cal N}(\mu(X), \Sigma),

    with the conditional mean function :math:`\mu(X) = X`.

    :param Sigma: The covariance matrix of the conditional. Dimensions should be [R, Dy, Dy], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [R, Dy, Dy],
        defaults to None
    :type Lambda: jnp.ndarray, optional
    :param ln_det_Sigma: Log determinant of the covariance matrix. Dimensions should be [R], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    :raises RuntimeError: Raised if neiterh Sigma nor Lambda are provided
    """

    def __post_init__(
        self,
    ):
        if self.Sigma is None and self.Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif self.Sigma is not None:
            if self.Lambda is None or self.ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = invert_diagonal(self.Sigma)
        else:
            self.Sigma, ln_det_Lambda = invert_diagonal(self.Lambda)
            self.ln_det_Sigma = -ln_det_Lambda

    def __str__(self) -> str:
        return "Conditional Gaussian density p(y|x)"

    def __call__(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :amt:`x`, i.e.

        .. math::

            p(Y\\vert X=x) =  {\cal N}(\mu(X=x), \Sigma)

        :param x: Instances, the :math:`\mu` should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on x.
        :rtype: pdf.GaussianPDF
        """
        return self.condition_on_x(x)

    def slice(self, indices: jnp.ndarray) -> "ConditionalGaussianPDF":
        """Return the conditional with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting conditional Gaussian diagonal density.
        :rtype: ConditionalGaussianPDF
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = ConditionalIdentityGaussianPDF(
            Sigma=Sigma_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        return new_measure

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the conditional :math:`\mu` function :math:`\mu(X=x) = M x + b`.

        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional means. Dimensions are [R, N, Dy].
        :rtype: jnp.ndarray
        """
        mu_y = jnp.tile(x[None], (self.R, 1, 1))
        return mu_y

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :math:`x`.

        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: The density conditioned on :math:`x`.
        :rtype: pdf.GaussianPDF
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((self.R * N, self.Dy))
        Sigma_new = jnp.tile(self.Sigma[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        Lambda_new = jnp.tile(self.Lambda[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        ln_det_Sigma_new = jnp.tile(self.ln_det_Sigma[:, None], (1, N)).reshape(
            self.R * N
        )
        return pdf.GaussianPDF(
            Sigma=Sigma_new,
            mu=mu_new,
            Lambda=Lambda_new,
            ln_det_Sigma=ln_det_Sigma_new,
        )

    def set_y(self, y: jnp.ndarray, **kwargs) -> factor.ConjugateFactor:
        """Set a specific value for :math:`y` in :math:`p(Y=y|X)` and returns the corresponding conjugate factor.

        :param y: Data for :math:`y`, where the rth entry is associated with the rth conditional density.
        :type y: jnp.ndarray [R, Dy]
        :return: The conjugate factor where the first dimension is R.
        :rtype: factor.ConjugateFactor
        """
        try:
            assert self.R == 1 or y.shape[0] == self.R
        except AssertionError:
            raise RuntimeError(
                "Either R should be one or the leading dimension should be equal to R"
            )
        nu_new = self.Lambda.diagonal(axis1=1, axis2=2) * y

        y_Lambda_y = jnp.sum(y**2 * self.Lambda.diagonal(axis1=1, axis2=2), axis=1)
        ln_beta_new = -0.5 * (
            y_Lambda_y + self.Dx * jnp.log(2 * jnp.pi) + self.ln_det_Sigma
        )
        factor_new = factor.ConjugateFactor(Lambda=self.Lambda, nu=nu_new, ln_beta=ln_beta_new)
        return factor_new

    def integrate_log_conditional(self, p_yx: pdf.GaussianPDF, **kwargs) -> jnp.ndarray:
        """Integrates over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.

        .. math::

            \int \log(p(Y|X))p(Y,X){\\rm d}Y{\\rm d}X.

        :param p_yx: Probability density function (first dimensions are :math:`Y`, last ones are :math:`X`).
        :type p_yx: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density :math:`p(Y,X)`.
        :rtype: jnp.ndarray
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(jnp.eye(self.Dy, self.Dy)[None])
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        quadratic_integral = p_yx.integrate("(Ax+a)'(Bx+b)", A_mat=A, B_mat=A_tilde)
        log_expectation = -0.5 * (
            quadratic_integral + (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        return log_expectation

    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: jnp.ndarray=None, **kwargs) -> callable:
        """Computes the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

           f(Y) = \int \log(p(Y|X))p(X){\\rm d}x.

        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = jnp.tile(jnp.eye(self.Dy)[None], (self.R, 1, 1))
        A_tilde = self.Lambda
        quadratic_integral = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=A, B_mat=A_tilde)
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde)
        log_expectation_constant = -0.5 * (
            quadratic_integral + (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_integral)
            + log_expectation_constant
        )
        if y == None:
            return log_expectation_y
        else:
            return log_expectation_y(y)
