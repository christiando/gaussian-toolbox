__author__ = "Christian Donner"

from jax import numpy as jnp
from typing import Tuple
from . import pdf, factor, measure, conditional
from .utils.linalg import invert_matrix

from .utils.dataclass import dataclass
from jax import lax
from jax import jit

@dataclass(kw_only=True)
class LConjugateFactorMGaussianConditional(conditional.ConditionalGaussianPDF):
    """ Base class for approximate conditional.
    """

    def evaluate_phi(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the feature vector
        
        .. math::

            \phi(X=x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x)))^\\top.

        :param x: Points where phi should be evaluated. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Feature vector. Dimensions are [N, Dphi].
        :rtype: jnp.ndarray
        """
        phi_x = jnp.block([x, self.k_func.evaluate(x).T])
        return phi_x

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the conditional mu function :math:`\mu(X=x) = M \phi(x) + b`.


        :param x: Points where :math:`\phi` should be evaluated. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional mu. Dimensions are [1, N, Dy].
        :rtype: jnp.ndarray
        """
        phi_x = self.evaluate_phi(x)
        mu_y = jnp.einsum("ab,cb->ca", self.M[0], phi_x) + self.b[0][None]
        return mu_y

    def set_y(self, y: jnp.ndarray, **kwargs):
        """Not valid function for this model class.

        :param y: Data for :math:`Y`, where the rth entry is associated with the rth conditional density. 
        :type y: jnp.ndarray [R, Dy]
        :raises AttributeError: Raised because doesn't :math:`p(Y|X)` is not a ConjugateFactor for :math:`X`. 
        """
        raise NotImplementedError("This class doesn't have the function set_y.")

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the expected covariance :math:`\Sigma_Y = \mathbb{E}[YY^\\top] - \mathbb{E}[Y]\mathbb{E}[Y]^\\top`.

        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: Returns the expected mean and covariance. Dimensions are [p_R, Dy], and  [p_R, Dy, Dy].
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        #### E[f(x)] ####
        # E[x] [R, Dx]
        Ex = p_x.integrate("x")
        # E[k(x)] [R, Dphi - Dx]
        p_k = p_x.multiply(self.k_func, update_full=True)
        Ekx = p_k.integrate().reshape((p_x.R, self.Dphi - self.Dx))
        # E[f(x)]
        Ef = jnp.concatenate([Ex, Ekx], axis=1)

        #### E[f(x)f(x)'] ####
        # Linear terms E[xx']
        Exx = p_x.integrate("xx'")
        # Cross terms E[x k(x)']
        Ekx = p_k.integrate("x").reshape((p_x.R, self.Dk, self.Dx))
        # kernel terms E[k(x)k(x)']
        Ekk = (
            p_k.multiply(self.k_func, update_full=True)
            .integrate()
            .reshape((p_x.R, self.Dk, self.Dk))
        )
        # Eff[:,self.Dx:,self.Dx:] = Ekk
        Eff = jnp.block([[Exx, jnp.swapaxes(Ekx, axis1=1, axis2=2)], [Ekx, Ekk]])

        ### mu_y = E[mu(x)] = ME[f(x)] + b ###
        mu_y = jnp.einsum("ab,cb->ca", self.M[0], Ef) + self.b[0][None]

        # Sigma_y = E[yy'] - mu_ymu_y' = Sigma + E[mu(x)mu(x)'] - mu_ymu_y'
        #                                = Sigma + ME[f(x)f(x)']M' + bE[f(x)']M' + ME[f(x)]b' + bb' - mu_ymu_y'
        Sigma_y = jnp.tile(self.Sigma, (p_x.R, 1, 1))
        Sigma_y += jnp.einsum(
            "ab,cbd->cad", self.M[0], jnp.einsum("abc,dc->abd", Eff, self.M[0])
        )
        MEfb = jnp.einsum(
            "ab,c->abc", jnp.einsum("ab,cb->ca", self.M[0], Ef), self.b[0]
        )
        Sigma_y += MEfb + jnp.swapaxes(MEfb, axis1=1, axis2=2)
        Sigma_y += (self.b[0, None] * self.b[0, :, None])[None]
        Sigma_y -= mu_y[:, None] * mu_y[:, :, None]
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> jnp.ndarray:
        """Compute :math:`\mathbb{E}[YX^\\top] = \int\int YX^\\top p(Y|X)p(X) {\\rm d}Y{\\rm d}x = \int (M f(X) + b)X^\\top p(X) {\\rm d}X`.

        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: Returns the cross expectations. Dimensions are [p_R, Dx, Dy].
        :rtype: jnp.ndarray
        """
        # E[xx']
        Exx = p_x.integrate("xx'")
        # E[k(x)x']
        Ekx = (
            p_x.multiply(self.k_func, update_full=True)
            .integrate("x")
            .reshape((p_x.R, self.Dk, self.Dx))
        )
        # E[f(x)x']
        Ef_x = jnp.concatenate([Exx, Ekx], axis=1)
        # M E[f(x)x']
        MEf_x = jnp.einsum("ab,cbd->cad", self.M[0], Ef_x)
        # bE[x']
        bEx = self.b[0][None, :, None] * p_x.integrate("x")[:, None]
        # E[yx']
        Eyx = MEf_x + bEx
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density

        .. math: 
        
            p(X,Y) \approx N(\mu_{XY},\Sigma_{XY}),

        The mean is given by

        .. math::
        
            \mu_{XY} = (\mu_X, \mu_Y)^\top

        with :math:`\mu_Y = \mathbb{E}[\mu_Y(X)]`. The covariance is given by
        
        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                        \Sigma_X  &                                \mathbb{E}[XY^\top] - \mu_X\mu_Y^\top \\
                        \mathbb{E}[YX^\top] - \mu_Y\mu_X^\top & \mathbb{E}[YY^\top] - \mu_Y\mu_Y^\top
                        \end{pmatrix}.


        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: The joint distribution p(x,y).
        :rtype: pdf.GaussianPDF
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        mu_xy = jnp.concatenate([mu_x, mu_y], axis=1)

        Sigma_xy = jnp.block(
            [[p_x.Sigma, jnp.swapaxes(cov_yx, axis1=1, axis2=2)], [cov_yx, Sigma_y]]
        )
        p_xy = pdf.GaussianPDF(Sigma=Sigma_xy, mu=mu_xy)
        return p_xy

    def affine_conditional_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> conditional.ConditionalGaussianPDF:
        r"""Get an approximation of the joint density via moment matching
        
        .. math::

            p(X|Y) \approx {\cal N}(\mu_{X|Y},Sigma_{X|Y}),

        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: The conditional density ::math:`p(X|Y)`.
        :rtype: conditional.ConditionalGaussianPDF
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = conditional.ConditionalGaussianPDF(
            M=M_new, b=b_new, Sigma=Sigma_new,
        )
        return cond_p_xy
    
    def integrate_log_conditional(
        self, p_yx: measure.GaussianMeasure, **kwargs
    ) -> jnp.ndarray:
        raise NotImplementedError("Log integral not implemented!")

    def integrate_log_conditional_y(
        self, p_x: measure.GaussianMeasure, **kwargs
    ) -> callable:
        raise NotImplementedError("Log integral not implemented!")

    def affine_marginal_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r""" Get an approximation of the marginal density
        
        .. math::

            p(Y)\aprox N(\mu_Y,\Sigma_y),

        The mean is given by
        
        .. math::

            \mu_Y = \mathbb{E}[\mu_Y(X)]. 

        The covariance is given by

        .. math::
        
            \Sigma_Y = E[YY^\top] - \mu_Y\mu_Y^\top.

        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: The joint distribution p(y).
        :rtype: pdf.GaussianPDF
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y,)
        return p_y

@dataclass(kw_only=True)
class LRBFGaussianConditional(LConjugateFactorMGaussianConditional):
    r"""A conditional Gaussian density, with a linear RBF mean (LRBFM) function,

    .. math::
        
        p(Y|X) = {\cal N}(\mu(X), \Sigma)

    with the conditional mean function :math:`\mu(X) = M \phi(X) + b`. 
    :math:`\phi(X)` is a feature vector of the form

    .. math:: 
    
        \phi(X) = (1,X_1,...,X_m,k(h_1(X)),...,k(h_n(X)))^\top,

    with

    .. math::
    
        k(h) = \exp(-h^2 / 2) \text{ and  } h_i(X) = (X_i - s_{i}) / l_i.

    Note, that the affine transformations will be approximated via moment matching.


    :param M: Matrix in the mean function. Dimensions should be [1, Dy, Dphi]
    :type M: jnp.ndarray
    :param b: Vector in the conditional mean function. Dimensions should be [1, Dy]
    :type b: jnp.ndarray
    :param mu: Parameters for linear mapping in the nonlinear functions. Dimensions should be [Dk, Dx]
    :type mu: jnp.ndarray
    :param length_scale: Length-scale of the kernels. Dimensions should be [Dk, Dx]
    :type length_scale: jnp.ndarray
    :param Sigma: The covariance matrix of the conditional. Dimensions should be [1, Dy, Dy], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [1, Dy, Dy], defaults to None
    :type Lambda: jnp.ndarray, optional
    :param ln_det_Sigma:  Log determinant of the covariance matrix. Dimensions should be [1], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    """
    M: jnp.ndarray
    b: jnp.ndarray
    mu: jnp.ndarray
    length_scale: jnp.ndarray
    Sigma: jnp.ndarray = None
    Lambda: jnp.ndarray = None
    ln_det_Sigma: jnp.ndarray = None

    def __post_init__(
        self,

    ):
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
        self.update_phi()
        
    @property
    def Dk(self):
        return self.mu.shape[0]
    
    @property
    def Dx(self):
        return self.mu.shape[1]
    
    @property
    def Dphi(self):
        return self.Dk + self.Dx

    def update_phi(self):
        """ Set up the non-linear kernel function in :math:`\phi(x)`.
        """
        Lambda = jnp.eye(self.Dx)[None] / self.length_scale[:, None] ** 2
        nu = self.mu / self.length_scale ** 2
        ln_beta = -0.5 * jnp.sum((self.mu / self.length_scale) ** 2, axis=1)
        self.k_func = measure.GaussianDiagMeasure(Lambda=Lambda, nu=nu, ln_beta=ln_beta)

    def integrate_log_conditional(
        self, p_yx: pdf.GaussianPDF, p_x: pdf.GaussianPDF = None, **kwargs
    ) -> jnp.ndarray:
        r"""Integrate over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.
        
        .. math::
        
            \int \log(p(Y|X))p(Y,X){\rm d}Y{\rm d}X.

        :param p_yx: Probability density function (first dimensions are :math:`Y`, last ones are :math:`X`).
        :type p_yx: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density :math:`p(Y,X)`.
        :rtype: jnp.ndarray
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        # E[(y - Mx - b)' Lambda (y - Mx - b)]
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(-self.M[:, :, : self.Dx])
        b = -self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = p_yx.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        # E[(y - Mx - b) Lambda Mk phi(x)]

        Lambda_joint = jnp.zeros((self.Dk, self.Dy + self.Dx, self.Dy + self.Dx))
        Lambda_joint = Lambda_joint.at[:, self.Dy :, self.Dy :].set(self.k_func.Lambda)
        nu_joint = jnp.zeros([self.Dk, self.Dy + self.Dx])
        nu_joint = nu_joint.at[:, self.Dy :].set(self.k_func.nu)
        joint_k_func = factor.ConjugateFactor(
            Lambda=Lambda_joint, nu=nu_joint, ln_beta=self.k_func.ln_beta
        )
        p_yx_k = p_yx.multiply(joint_k_func, update_full=True)
        E_k_lin_term = jnp.reshape(
            p_yx_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_yx.R, self.Dk, self.Dy),
        )
        Mk = self.M[:, :, self.Dx :]
        lin_kernel_integral = jnp.einsum("abc,acb->a", Mk, E_k_lin_term)

        # E[phi(x)' Mk'  Lambda Mk phi(x)]
        if p_x is None:
            p_x = p_yx.get_marginal(jnp.arange(self.Dy, self.Dy + self.Dx))
        p_x_kk = p_x.multiply(self.k_func).multiply(self.k_func, update_full=True)
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant = self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi)
        log_expectation = -0.5 * (
            quadratic_integral
            - 2 * lin_kernel_integral
            + kernel_kernel_integral
            + constant
        )
        return log_expectation

    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: jnp.ndarray=None, **kwargs) -> callable:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::
        
            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.
    
        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianPDF
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = self.M[:, :, : self.Dx]
        b = self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)

        Mk = self.M[:, :, self.Dx :]
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde)
        p_x_k = p_x.multiply(self.k_func, update_full=True)
        E_k = jnp.reshape(p_x_k.integrate(), (p_x.R, self.Dk))
        E_Mk = jnp.einsum("abc,ac->ab", self.Lambda, jnp.einsum("abc,ac->ab", Mk, E_k))
        linear_term = linear_integral + E_Mk
        quadratic_integral = p_x.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        E_k_lin = jnp.reshape(
            p_x_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_x.R, self.Dk, self.Dy),
        )
        E_Mk_lin = jnp.einsum("abc,acb->a", Mk, E_k_lin)
        p_x_kk = p_x_k.multiply(self.k_func, update_full=True)
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant_term = -0.5 * (
            quadratic_integral
            + 2 * E_Mk_lin
            + kernel_kernel_integral
            + self.ln_det_Sigma
            + self.Dy * jnp.log(2.0 * jnp.pi)
        )

        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_term)
            + constant_term
        )
        if y == None:
            return log_expectation_y
        else:
            return log_expectation_y(y)
@dataclass(kw_only=True)
class LSEMGaussianConditional(LConjugateFactorMGaussianConditional):
    r"""A conditional Gaussian density, with a linear squared exponential mean (LSEM) function,

    .. math::
    
        p(Y|X) = {\cal N}(\mu(X), \Sigma)

    with the conditional mean function :math:`mu(X) = M \phi(X) + b`. 
    :math:`\phi(X)` is a feature vector of the form

    .. math::
    
        \phi(X) = (1,X_1,...,X_m,k(h_1(X)),...,k(h_n(X)))^\top,

    with
    
    .. math::

        k(h) = exp(-h^2 / 2) \text{ and } h_i(x) = w_i^\top x + w_{i,0}.

    Note, that the affine transformations will be approximated via moment matching.

    :param M: Matrix in the mean function. Dimensions should be [1, Dy, Dphi]
    :type M: jnp.ndarray
    :param b: Vector in the conditional mean function. Dimensions should be [1, Dy]
    :type b: jnp.ndarray
    :param W: Parameters for linear mapping in the nonlinear functions. Dimensions should be [Dk, Dx + 1]
    :type W: jnp.ndarray
    :param Sigma: The covariance matrix of the conditional. Dimensions should be [1, Dy, Dy], defaults to None
    :type Sigma: jnp.ndarray, optional
    :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [1, Dy, Dy], defaults to None
    :type Lambda: jnp.ndarray, optional
    :param ln_det_Sigma:  Log determinant of the covariance matrix. Dimensions should be [1], defaults to None
    :type ln_det_Sigma: jnp.ndarray, optional
    """

    M: jnp.ndarray
    b: jnp.ndarray
    W: jnp.ndarray
    Sigma: jnp.ndarray = None
    Lambda: jnp.ndarray = None
    ln_det_Sigma: jnp.ndarray = None
    
    def __post_init__(
        self,

    ):
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
        self.w0 = self.W[:, 0]
        self.W = self.W[:, 1:]
        self.update_phi()
        
    @property
    def Dk(self):
        return self.W.shape[0]
    
    @property
    def Dx(self):
        return self.W.shape[1]
    
    @property
    def Dphi(self):
        return self.Dk + self.Dx
    

    def update_phi(self):
        """ Set up the non-linear kernel function in :math:`\phi(x)`.
        """
        v = self.W
        nu = self.W * self.w0[:, None]
        ln_beta = -0.5 * self.w0 ** 2
        self.k_func = factor.OneRankFactor(v=v, nu=nu, ln_beta=ln_beta)

    def integrate_log_conditional(
        self, p_yx: pdf.GaussianPDF, p_x: pdf.GaussianPDF = None, **kwargs
    ) -> jnp.ndarray:
        r"""Integrate over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.
        
        .. math::
        
            \int \log(p(Y|X))p(Y,X){\rm d}Y{\rm d}X.

        :param p_yx: Probability density function (first dimensions are :math:`Y`, last ones are :math:`X`).
        :type p_yx: measure.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density :math:`p(Y,X)`.
        :rtype: jnp.ndarray
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        # E[(y - Mx - b)' Lambda (y - Mx - b)]
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(-self.M[:, :, : self.Dx])
        b = -self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = p_yx.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        # E[(y - Mx - b) Lambda Mk phi(x)]
        zero_arr = jnp.zeros([self.Dk, self.Dy + self.Dx])
        v_joint = zero_arr.at[:, self.Dy :].set(self.k_func.v)
        nu_joint = zero_arr.at[:, self.Dy :].set(self.k_func.nu)
        joint_k_func = factor.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=self.k_func.ln_beta
        )
        p_yx_k = p_yx.multiply(joint_k_func, update_full=True)
        E_k_lin_term = jnp.reshape(
            p_yx_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_yx.R, self.Dk, self.Dy),
        )
        Mk = self.M[:, :, self.Dx :]
        lin_kernel_integral = jnp.einsum("abc,acb->a", Mk, E_k_lin_term)

        # E[phi(x)' Mk'  Lambda Mk phi(x)]
        if p_x is None:
            p_x = p_yx.get_marginal(jnp.arange(self.Dy, self.Dy + self.Dx))
        p_x_kk = p_x.multiply(self.k_func, update_full=True).multiply(
            self.k_func, update_full=True
        )
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant = self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi)
        log_expectation = -0.5 * (
            quadratic_integral
            - 2 * lin_kernel_integral
            + kernel_kernel_integral
            + constant
        )
        return log_expectation

    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: jnp.ndarray=None, **kwargs) -> callable:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::
        
            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.
    
        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianPDF
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = self.M[:, :, : self.Dx]
        b = self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)

        Mk = self.M[:, :, self.Dx :]
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde)
        p_x_k = p_x.multiply(self.k_func, update_full=True)
        E_k = jnp.reshape(p_x_k.integrate(), (p_x.R, self.Dk))
        E_Mk = jnp.einsum("abc,ac->ab", self.Lambda, jnp.einsum("abc,ac->ab", Mk, E_k))
        linear_term = linear_integral + E_Mk
        quadratic_integral = p_x.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        E_k_lin = jnp.reshape(
            p_x_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_x.R, self.Dk, self.Dy),
        )
        E_Mk_lin = jnp.einsum("abc,acb->a", Mk, E_k_lin)
        p_x_kk = p_x_k.multiply(self.k_func, update_full=True)
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant_term = -0.5 * (
            quadratic_integral
            + 2 * E_Mk_lin
            + kernel_kernel_integral
            + self.ln_det_Sigma
            + self.Dy * jnp.log(2.0 * jnp.pi)
        )

        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_term)
            + constant_term
        )
        if y == None:
            return log_expectation_y
        else:
            return log_expectation_y(y)

@dataclass(kw_only=True)
class HCCovGaussianConditional(conditional.ConditionalGaussianPDF):
    """A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::
        
        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`. 
    The covariance matrix has the form
    
    .. math::

        \Sigma_y(x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\\top,

    and :math:`D_i(x) = 2 * \\beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    :param M: Matrix in the mean function. Dimensions should be [1, Dy, Dx].
    :type M: jnp.ndarray
    :param b: Vector in the conditional mean function. Dimensions should be [1, Dy].
    :type b: jnp.ndarray
    :param sigma_x: Diagonal noise parameter.
    :type sigma_x: jnp.ndarray
    :param U: Othonormal vectors for low rank noise part. Dimensions should be [Dy, Du].
    :type U: jnp.ndarray
    :param W: Noise weights for low rank components (w_i & b_i). Dimensions should be [Du, Dx + 1].
    :type W: jnp.ndarray
    :param beta: Scaling for low rank noise components. Dimensions should be [Du].
    :type beta: jnp.ndarray
    :raises NotImplementedError: Only works with R==1.
    """
    M: jnp.ndarray
    b: jnp.ndarray
    sigma_x: jnp.ndarray
    U: jnp.ndarray
    W: jnp.ndarray
    beta: jnp.ndarray

    def __post_init__(
        self,

    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        self.sigma2_x = self.sigma_x ** 2
        self._setup_noise_diagonal_functions()
        
    @property
    def R(self):
        return self.M.shape[0]
    
    @property
    def Dy(self):
        return self.M.shape[1]
    
    @property
    def Dx(self):
        return self.M.shape[2]
    
    @property
    def Du(self):
        return self.beta.shape[0]

    def _setup_noise_diagonal_functions(self):
        """Create the functions, that later need to be integrated over, i.e.

        .. math::
            
            \exp(h_i(z)) \\text{ and } \exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta)
        self.exp_h_minus = factor.LinearFactor(nu=-nu, ln_beta=-ln_beta)

    def get_conditional_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::
        
            \Sigma_y(X=x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

        with :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.


        :param x: Instances, the :math:`\mu` should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional covariance. Dimensions are [N, Dy, Dy]
        :rtype: jnp.ndarray
        """
        D_x = self.beta[None, :, None] * (self.exp_h_plus(x) + self.exp_h_minus(x))
        Sigma_0 = self.sigma2_x * jnp.eye(self.Dy)
        Sigma_y_x = Sigma_0[None] + jnp.einsum(
            "ab,cb->ac", jnp.einsum("ab,cb->ca", self.U, D_x), self.U
        )
        return Sigma_y_x

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get Gaussian Density conditioned on :math:`X=x`.

        :param x: Instances, the mu and Sigma should be conditioned on. Dimensions should be [N, Dx]
        :type x: jnp.ndarray
        :return: The density conditioned on :math:`X=x`.
        :rtype: pdf.GaussianPDF
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new = self.get_conditional_cov(x)
        return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: jnp.ndarray, **kwargs):
        """Not valid function for this model class.

        :param y: Data for :math:`Y`, where the rth entry is associated with the rth conditional density. 
        :type y: jnp.ndarray [R, Dy]
        :raises AttributeError: Raised because doesn't :math:`p(Y|X)` is not a ConjugateFactor for :math:`X`. 
        """
        raise AttributeError("HCCovGaussianConditional doesn't have attributee set_y.")

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> jnp.ndarray:
        r"""Integrate covariance with respect to :math:`p(X)`.
        
        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        :param p_x: The density the covatiance is integrated with.
        :type p_x: pdf.GaussianPDF
        :return: Integrated covariance matrix. Dimensions are [Dy, Dy]
        :rtype: jnp.ndarray
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus).integrate()
            + p_x.multiply(self.exp_h_minus).integrate()
        )
        D_int = self.beta[None] * D_int.reshape((p_x.R, self.Du))
        return self.sigma2_x * jnp.eye(self.Dy)[None] + jnp.einsum(
            "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        )

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the expected mean and covariance

        mu_y = E[y] = M E[x] + b

        Sigma_y = E[yy'] - mu_y mu_y' = sigma_x^2 I + \sum_i U_i E[D_i(x)] U_i' + E[mu(x)mu(x)'] - mu_y mu_y'


        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: Returns the expected mean and covariance. Dimensions should be [p_R, Dy] and [p_R, Dy, Dy].
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        # Sigma_y = .5 * (Sigma_y + Sigma_y.T)
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> jnp.ndarray:
        """Compute E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M x + b)x' p(x) dx

        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: Cross expectations. Dimensions are [p_R, Dx, Dy]
        :rtype: jnp.ndarray
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Get an approximation of the joint density

            p(x,y) ~= N(mu_{xy},Sigma_{xy}),

        The mean is given by

            mu_{xy} = (mu_x, mu_y)'

        with mu_y = E[mu_y(x)]. The covariance is given by

            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').

        :param p_x: The density which we average over.
        :type p_x: pdf.GaussianPDF
        :return: Joint distribution of p(x,y).
        :rtype: pdf.GaussianPDF
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        mu_xy = jnp.concatenate([mu_x, mu_y], axis=1)
        Sigma_xy1 = jnp.concatenate(
            [p_x.Sigma, jnp.swapaxes(cov_yx, axis1=1, axis2=2)], axis=2
        )
        Sigma_xy2 = jnp.concatenate([cov_yx, Sigma_y], axis=2)
        Sigma_xy = jnp.concatenate([Sigma_xy1, Sigma_xy2], axis=1)
        p_xy = pdf.GaussianPDF(Sigma=Sigma_xy, mu=mu_xy)
        return p_xy

    def affine_conditional_transformation(
        self, p_x: pdf.GaussianPDF
    ) -> conditional.ConditionalGaussianPDF:
        r"""Get an approximation of the joint density via moment matching
        
        .. math::

            p(X|Y) \approx {\cal N}(\mu_{X|Y},\Sigma_{X|Y}).

        :param p_x: Marginal Gaussian density over :math:`X`.
        :type p_x: pdf.GaussianPDF
        :return: Conditional density of :math:`p(X|Y)`.
        :rtype: conditional.ConditionalGaussianPDF
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = conditional.ConditionalGaussianPDF(
            M=M_new, b=b_new, Sigma=Sigma_new,
        )
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the marginal density

        .. math
        
            p(Y) \approx N(\mu_Y,\Sigma_Y),

        The mean is given by

        .. math::
        
            \mu_Y = \mathbb{E}[\mu_Y(X)]. 

        The covariance is given by
        
        .. math::

            \Sigma_y = \mathbb{E}[YY^\top] - \mu_Y\mu_Y^\top.

        :param p_x: Marginal Gaussian density over :math`X`.
        :type p_x: pdf.GaussianPDF
        :return: The marginal density :math:`p(Y)`.
        :rtype: pdf.GaussianPDF
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y


    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: jnp.ndarray, **kwargs) -> callable:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::
        
            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.
    
        :param p_x: Density over :math:`X`.
        :type p_x: measure.GaussianPDF
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of :math:`Y`.
        :rtype: callable
        """
        vec = y - self.b
        E_epsilon2 = jnp.sum(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=-self.M, a_vec=vec, B_mat=-self.M, b_vec=vec),
            axis=0,
        )
        
        def scan_body_function(carry, args_i):
            W_i, u_i, beta_i = args_i
            omega_star_i = lax.stop_gradient(self._get_omega_star_i(W_i, u_i, beta_i, p_x, y)[0])
            uRu_i, log_lb_sum_i = self._get_lb_i(W_i, u_i, beta_i, omega_star_i, p_x, y)
            result = (uRu_i, log_lb_sum_i)
            return carry, result

        _, result = lax.scan(scan_body_function, None, (self.W, self.U.T, self.beta))
        uRu, log_lb_sum = result
        
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -0.5 * (E_epsilon2 - E_D_inv_epsilon2) / self.sigma_x**2
        # determinant part
        Qm = Qm - 0.5 * E_ln_sigma2_f + 0.5 * p_x.R * (self.Du - self.Dy) * jnp.log(self.sigma_x**2)

    def _get_lb_i(self, W_i: jnp.ndarray, u_i: jnp.ndarray, beta_i: jnp.ndarray, omega_star: jnp.ndarray, p_x: pdf.GaussianPDF, y: jnp.ndarray):
        # phi = pdf.GaussianPDF(**phi_dict)
        # beta = self.beta[iu:iu + 1]
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        R = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (R, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        uC = jnp.dot(u_i.T, -self.M[0])
        uy_d = jnp.dot(u_i.T, (y - self.b[0]).T)
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        )
        g_omega = self.g(omega_star, beta_i)
        nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
        ln_beta = (
            -jnp.log(self.sigma_x**2 + self.f(omega_star, beta_i))
            - 0.5 * g_omega * (b_i**2 - omega_star**2)
            + jnp.log(beta_i)
        )
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        # Create the two measures
        exp_phi_plus = p_x.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = p_x.hadamard(exp_factor_minus, update_full=True)
        # Fourth order integrals E[h^2 (x-Cz-d)^2]
        quart_int_plus = exp_phi_plus.integrate(
            "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
            A_mat=uC,
            a_vec=uy_d.T,
            B_mat=uC,
            b_vec=uy_d.T,
            C_mat=w_i,
            c_vec=b_i,
            D_mat=w_i,
            d_vec=b_i,
        )
        quart_int_minus = exp_phi_minus.integrate(
            "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
            A_mat=uC,
            a_vec=uy_d.T,
            B_mat=uC,
            b_vec=uy_d.T,
            C_mat=w_i,
            c_vec=b_i,
            D_mat=w_i,
            d_vec=b_i,
        )
        quart_int = quart_int_plus + quart_int_minus
        # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
        quad_int_plus = exp_phi_plus.integrate(
            "(Ax+a)'(Bx+b)", A_mat=uC, a_vec=uy_d.T, B_mat=uC, b_vec=uy_d.T
        )
        quad_int_minus = exp_phi_minus.integrate(
            "(Ax+a)'(Bx+b)", A_mat=uC, a_vec=uy_d.T, B_mat=uC, b_vec=uy_d.T
        )
        quad_int = quad_int_plus + quad_int_minus
        omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))
        f_omega_dagger = self.f(omega_dagger, beta_i)
        log_lb = jnp.log(self.sigma_x**2 + f_omega_dagger)
        g_omega = self.g(omega_star, beta_i)
        nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
        ln_beta = (
            -jnp.log(self.sigma_x**2 + self.f(omega_star, beta_i))
            - 0.5 * g_omega * (b_i**2 - omega_star**2)
            + jnp.log(beta_i)
        )
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        # Create the two measures
        exp_phi_plus = p_x.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = p_x.hadamard(exp_factor_minus, update_full=True)
        mat1 = -self.M[0]
        vec1 = y - self.b[0]
        R_plus = exp_phi_plus.integrate(
            "(Ax+a)(Bx+b)'", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R_minus = exp_phi_minus.integrate(
            "(Ax+a)(Bx+b)'", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R = R_plus + R_minus
        R = jnp.sum(R, axis=0)
        # R = .5 * (R + R.T)
        uRu = jnp.sum(u_i * jnp.dot(R, u_i))
        log_lb_sum = jnp.sum(log_lb)
        return uRu, log_lb_sum
    
    def _get_omega_star_i(self, W_i: jnp.ndarray, u_i: jnp.ndarray, beta_i: jnp.ndarray, p_x: pdf.GaussianPDF, y: jnp.ndarray, conv_crit=1e-3):
        R = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (R, 1))
        b_i = W_i[:1]
        u_i = u_i[:].reshape((-1, 1))
        uM = jnp.dot(u_i.T, -self.M[0])
        uy_b = jnp.dot(u_i.T, (y - self.b[0]).T)
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        )
        omega_star = omega_dagger
        # omega_star = omega_star_init
        omega_old = omega_dagger + 1

        def body_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # From the lower bound term
            g_omega = self.g(omega_star, beta_i)
            nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
            nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
            ln_beta = (
                -jnp.log(self.sigma_x**2 + self.f(omega_star, beta_i))
                - 0.5 * g_omega * (b_i**2 - omega_star**2)
                + jnp.log(beta_i)
            )
            ln_beta_plus = ln_beta + b_i
            ln_beta_minus = ln_beta - b_i
            # Create OneRankFactors
            exp_factor_plus = factor.OneRankFactor(
                v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
            )
            exp_factor_minus = factor.OneRankFactor(
                v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
            )
            # Create the two measures
            exp_phi_plus = p_x.hadamard(exp_factor_plus, update_full=True)
            exp_phi_minus = p_x.hadamard(exp_factor_minus, update_full=True)
            # Fourth order integrals E[h^2 (x-Cz-d)^2]
            quart_int_plus = exp_phi_plus.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=uM,
                a_vec=uy_b.T,
                B_mat=uM,
                b_vec=uy_b.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int_minus = exp_phi_minus.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=uM,
                a_vec=uy_b.T,
                B_mat=uM,
                b_vec=uy_b.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int = quart_int_plus + quart_int_minus
            # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate(
                "(Ax+a)'(Bx+b)", A_mat=uM, a_vec=uy_b.T, B_mat=uM, b_vec=uy_b.T
            )
            quad_int_minus = exp_phi_minus.integrate(
                "(Ax+a)'(Bx+b)", A_mat=uM, a_vec=uy_b.T, B_mat=uM, b_vec=uy_b.T
            )
            quad_int = quad_int_plus + quad_int_minus
            omega_old = omega_star
            omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))
            num_iter = num_iter + 1
            return omega_star, omega_old, num_iter

        def cond_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # return lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old)), 'i') > conv_crit
            return jnp.logical_and(
                jnp.amax(jnp.amax(jnp.abs(omega_star - omega_old) / omega_star))
                > conv_crit,
                num_iter < 100,
            )

        num_iter = 0
        init_val = (omega_star, omega_old, num_iter)
        omega_star, omega_old, num_iter = lax.while_loop(cond_fun, body_fun, init_val)
        indices_non_converged = (
            jnp.abs(omega_star - omega_old) / omega_star
        ) > conv_crit

        return omega_star, indices_non_converged
    
    def f(self, h: jnp.ndarray, beta: float) -> jnp.ndarray:
        """Compute the function

        f(h) = 2 * beta * cosh(h)

        :param h: Activation functions.
        :type h: jnp.ndarray
        :param beta: Scaling factor.
        :type beta: float
        :return: Evaluated functions.
        :rtype:jnp.ndarray
        """
        return 2 * beta * jnp.cosh(h)

    def f_prime(self, h: jnp.ndarray, beta: float) -> jnp.ndarray:
        """Computes the derivative of f

            f'(h) = 2 * beta * sinh(h)

        :param h: Activation functions.
        :type h: jnp.ndarray
        :param beta: Scaling factor.
        :type beta: float
        :return: Evaluated derivative functions.
        :rtype:jnp.ndarray
        """
        return 2 * beta * jnp.sinh(h)

    def g(self, omega: jnp.ndarray, beta: float) -> jnp.ndarray:
        r"""Computes the function

            g(omega) = f'(omega) / (sigma_x^2 + f(omega)) / |omega|

            for the variational bound

        :param omega: Free variational parameter.
        :type omega: jnp.ndarray
        :param beta:  Scaling factor.
        :type beta: jnp.ndarray
        :param sigma_x: Noise parameter.
        :type sigma_x: float
        :return: Evaluated function.
        :rtype: jnp.ndarray
        """
        return (
            self.f_prime(omega, beta)
            / (self.sigma_x**2 + self.f(omega, beta))
            / jnp.abs(omega)
        )
