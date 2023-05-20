__author__ = "Christian Donner"

from jax import numpy as jnp
from typing import Tuple, Union
from . import pdf, factor, measure, conditional
from .utils.linalg import invert_matrix

from .utils.dataclass import dataclass
from jaxtyping import Array, Float, Int, Bool
from jax import lax
from jax import jit
from jax import scipy as jsc

from dataclasses import field
from jax import random, vmap
from abc import abstractmethod

@dataclass(kw_only=True)
class LConjugateFactorMGaussianConditional(conditional.ConditionalGaussianPDF):
    """Base class for approximate conditional."""

    def evaluate_phi(self, x: Float[Array, "N Dx"]) -> Float[Array, "N Dphi"]:
        """Evaluate the feature vector

        .. math::

            \phi(X=x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x)))^\\top.

        Args:
            x: Points where phi should be evaluated.

        Returns:
            Feature vector.
        """
        phi_x = jnp.block([x, self.k_func.evaluate(x).T])
        return phi_x

    def get_conditional_mu(
        self, x: Float[Array, "N Dx"], **kwargs
    ) -> Float[Array, "1 N Dy"]:
        """Compute the conditional mu function :math:`\mu(X=x) = M \phi(x) + b`.

        Args:
            x: Points where :math:`\phi` should be evaluated.

        Returns:
            Conditional mu.
        """
        phi_x = self.evaluate_phi(x)
        mu_y = jnp.einsum("ab,cb->ca", self.M[0], phi_x) + self.b[0][None]
        return mu_y

    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        """Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise NotImplementedError("This class doesn't have the function set_y.")

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "R Dy Dy"]]:
        """Compute the expected covariance :math:`\Sigma_Y = \mathbb{E}[YY^\\top] - \mathbb{E}[Y]\mathbb{E}[Y]^\\top`.

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        #### \mathbb{E}[f(x)] ####
        # \mathbb{E}[x] [R, Dx]
        Ex = p_x.integrate("x")
        # \mathbb{E}[k(x)] [R, Dphi - Dx]
        p_k = p_x.multiply(self.k_func, update_full=True)
        Ekx = p_k.integrate().reshape((p_x.R, self.Dphi - self.Dx))
        # \mathbb{E}[f(x)]
        Ef = jnp.concatenate([Ex, Ekx], axis=1)

        #### \mathbb{E}[f(x)f(x)'] ####
        # Linear terms \mathbb{E}[xx']
        Exx = p_x.integrate("xx'")
        # Cross terms \mathbb{E}[x k(x)']
        Ekx = p_k.integrate("x").reshape((p_x.R, self.Dk, self.Dx))
        # kernel terms \mathbb{E}[k(x)k(x)']
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
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        """Compute :math:`\mathbb{E}[YX^\\top] = \int\int YX^\\top p(Y|X)p(X) {\\rm d}Y{\\rm d}x = \int (M f(X) + b)X^\\top p(X) {\\rm d}X`.

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the cross expectations.
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

        Args:
            p_x: The density which we average over.

        Returns:
            The joint distribution p(x,y).
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

        Args:
            p_x: The density which we average over.

        Returns:
            The conditional density ::math:`p(X|Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        Sigma_new = 0.5 * (Sigma_new + jnp.swapaxes(Sigma_new, -1, -2))
        cond_p_xy = conditional.ConditionalGaussianPDF(
            M=M_new,
            b=b_new,
            Sigma=Sigma_new,
        )
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the marginal density

        .. math::

           p(Y)\aprox N(\mu_Y,\Sigma_y),

        The mean is given by

        .. math::

           \mu_Y = \mathbb{E}[\mu_Y(X)].

        The covariance is given by

        .. math::

           \Sigma_Y = \mathbb{E}[YY^\top] - \mu_Y\mu_Y^\top.

        Args:
            p_x: The density which we average over.

        Returns:
            The joint distribution p(y).
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(
            Sigma=Sigma_y,
            mu=mu_y,
        )
        return p_y

    def integrate_log_conditional(
        self, p_yx: measure.GaussianMeasure, **kwargs
    ) -> Float[Array, "R"]:
        raise NotImplementedError("Log integral not implemented!")

    def integrate_log_conditional_y(
        self, p_x: measure.GaussianMeasure, **kwargs
    ) -> callable:
        raise NotImplementedError("Log integral not implemented!")


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

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        mu: Parameters for linear mapping in the nonlinear functions.
        length_scale: Length-scale of the kernels.
        Sigma: The covariance matrix of the conditional.
        Lambda: Information (precision) matrix of the Gaussians.
        ln_det_Sigma: Log determinant of the covariance matrix.

    Raises:
        RuntimeError: If neither Sigma nor Lambda are provided.
    """
    M: Float[Array, "1 Dy Dk+Dx"]
    b: Float[Array, "1 Dy"]
    mu: Float[Array, "Dk Dx"]
    length_scale: Float[Array, "Dk Dx"]
    Sigma: Float[Array, "1 Dy Dy"] = None
    Lambda: Float[Array, "1 Dy Dy"] = None
    ln_det_Sigma: Float[Array, "1"] = None

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
    def Dk(self) -> int:
        r"""Number of kernels."""
        return self.mu.shape[0]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.mu.shape[1]

    @property
    def Dphi(self) -> int:
        r"""Dimensionality of feature vector (:math:`D_x+D_k`)."""
        return self.Dk + self.Dx

    def update_phi(self):
        """Set up the non-linear kernel function in :math:`\phi(x)`."""
        Lambda = jnp.eye(self.Dx)[None] / self.length_scale[:, None] ** 2
        nu = self.mu / self.length_scale**2
        ln_beta = -0.5 * jnp.sum((self.mu / self.length_scale) ** 2, axis=1)
        self.k_func = measure.GaussianDiagMeasure(Lambda=Lambda, nu=nu, ln_beta=ln_beta)

    def integrate_log_conditional(
        self, p_yx: pdf.GaussianPDF, p_x: pdf.GaussianPDF = None, **kwargs
    ) -> Float[Array, "R"]:
        r"""Integrate over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.

        .. math::

            \int \log(p(Y|X))p(Y,X){\rm d}Y{\rm d}X.

        Args:
            p_yx: Probability density function (first dimensions are
                :math:`Y`, last ones are :math:`X`).

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            Returns the integral with respect to density :math:`p(Y,X)`.
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

    def integrate_log_conditional_y(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "R Dy"] = None, **kwargs
    ) -> Union[callable, Float[Array, "R Dy"]]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            callable
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

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        W: Parameters for linear mapping in the nonlinear functions.
        Sigma: The covariance matrix of the conditional.
        Lambda: Information (precision) matrix of the Gaussians.
        ln_det_Sigma: Log determinant of the covariance matrix.

    Raises:
        RuntimeError: If neither Sigma nor Lambda are provided.
    """

    M: Float[Array, "1 Dy Dk+Dx"]
    b: Float[Array, "1 Dy"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = None
    Lambda: Float[Array, "1 Dy Dy"] = None
    ln_det_Sigma: Float[Array, "1"] = None

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
    def Dk(self) -> int:
        r"""Number of kernels."""
        return self.W.shape[0]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.W.shape[1]

    @property
    def Dphi(self) -> int:
        r"""Dimensionality of feature vector (:math:`D_x+D_k`)."""
        return self.Dk + self.Dx

    def update_phi(self):
        """Set up the non-linear kernel function in :math:`\phi(x)`."""
        v = self.W
        nu = self.W * self.w0[:, None]
        ln_beta = -0.5 * self.w0**2
        self.k_func = factor.OneRankFactor(v=v, nu=nu, ln_beta=ln_beta)

    def integrate_log_conditional(
        self, p_yx: pdf.GaussianPDF, p_x: pdf.GaussianPDF = None, **kwargs
    ) -> Float[Array, "R"]:
        r"""Integrate over the log conditional with respect to the pdf :math:`p(Y,X)`. I.e.

        .. math::

            \int \log(p(Y|X))p(Y,X){\rm d}Y{\rm d}X.

        Args:
            p_yx: Probability density function (first dimensions are
                :math:`Y`, last ones are :math:`X`).

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            Returns the integral with respect to density :math:`p(Y,X)`.
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

    def integrate_log_conditional_y(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "R Dy"] = None, **kwargs
    ) -> Union[callable, Float[Array, "R"]]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral as function of :math:`Y`. If provided already
            evaluated for :math:`Y=y`.
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
class HeteroscedasticConditional(conditional.ConditionalGaussianPDF):
    r"""A conditional Gaussian density, with a heteroscedastic covariance,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = AA^\top + AD(x)A^\top.

    and :math:`D_i(x) = \exp(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    
    def __post_init__(
        self,
    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        try:
            assert self.Dy <= self.Da
        except AssertionError:
            raise NotImplementedError("A must have at least as many rows as columns.")
        try:
            assert self.Dk <= self.Da
        except AssertionError:
            raise NotImplementedError("Diagonal matrix can have at most as many entries as A has columns.")
        
        self.Sigma = jnp.einsum("abc,adc->abd", self.A, self.A)
        self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        
    @abstractmethod
    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        pass

    @property
    def R(self) -> int:
        """Number of conditionals (leading dimension)."""
        return self.M.shape[0]

    @property
    def Dy(self) -> int:
        r"""Dimensionality of :math:`Y`."""
        return self.M.shape[1]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.M.shape[2]

    @property
    def Da(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.A.shape[2]
    
    @property
    def Dk(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.W.shape[0]
    
    def linear_layer(self, x: Float[Array, "N Dx"]) -> Float[Array, "N Dk"]:
        """Linear layer of the argument of heteroscedastic link function.

        :return: :math:`w^\top x + w_0`.
        """
        return jnp.einsum("ab,cb->ca", self.W[:,1:], x) + self.W[:,0][None]
    
    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \Sigma + A_{:k} D(x) A_{:k}^\top,

        with :math:`D_i(x) = \cosh(h_i(x))-1` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        h = self.linear_layer(x)
        D_x = self.link_function(h) 
        Sigma = self.Sigma + jnp.einsum(
            "abc,dc->abd", jnp.einsum("ab,cb->cab", self.A[0,:,:self.Dk], D_x), self.A[0,:,:self.Dk]
        )
        if invert:
            G_x = D_x / (1 + D_x) # [N x Dk]
            A_inv = jnp.einsum('ab,bc->ac', self.Lambda[0], self.A[0,:,:self.Dk]) # [Dy x Dk] # [N x Dy x Dk]
            Lambda = self.Lambda - jnp.einsum(
                "abc,dc->abd", jnp.einsum("ab,cb->cab", A_inv, G_x), A_inv
            )
            ln_det_Sigma_y_x = self.ln_det_Sigma + jnp.sum(jnp.log(1 + D_x), axis=1)
            return Sigma, Lambda, ln_det_Sigma_y_x
        else:
            return Sigma
        
    def condition_on_x(self, x: Float[Array, "N Dx"], **kwargs) -> pdf.GaussianPDF:
        r"""Get Gaussian Density conditioned on :math:`X=x`.

        Args:
            x: Instances, the mu and Sigma should be conditioned on.

        Returns:
            The density conditioned on :math:`X=x`.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new, Lambda_new, ln_det_Sigma_new = self.get_conditional_cov(
            x, invert=True
        )
        # Sigma_new = .5 * (Sigma_new + jnp.swapaxes(Sigma_new, -2, -1))
        return pdf.GaussianPDF(
            Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        # Sigma_new = self.get_conditional_cov(x)
        # return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)
        
    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        r"""Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise AttributeError("HeteroscedasticConditional doesn't have function set_y.")
    
    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int f(h(z)) dphi(z)
        D_int = self._integrate_noise_diagonal(p_x)
        # rotation_mat_int = jnp.eye(self.Dy)[None] + jnp.einsum(
        #    "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        # )
        Sigma_int = self.Sigma + jnp.einsum(
            "ab,cb->ac", jnp.einsum("ab,b->ab", self.A[0,:,:self.Dk], D_int), self.A[0,:,:self.Dk]
        )[None]
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int
    
    @abstractmethod
    def _integrate_noise_diagonal(self, p_x: pdf.GaussianPDF) -> Float[Array, "N Dk"]:
        r"""Integrate the noise diagonal with respect to :math:`p(X)`.

        .. math::

            \int D(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        pass
    
    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "1 Dy Dy"]]:
        r"""Compute the expected mean and covariance

        .. math::

            \mu_y = \mathbb{E}[y] = M \mathbb{E}[x] + b

        .. math::

            \Sigma_y = \mathbb{E}[yy'] - \mu_y \mu_y^\top = \sigma_x^2 I + \sum_i U_i \mathbb{E}[D_i(x)] U_i^\top + \mathbb{E}[\mu(x)\mu(x)^\top] - \mu_y \mu_y^\top

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y
    
    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density
        
        .. math::

            p(x,y) ~= {\cal N}(\mu_{xy},\Sigma_{xy}),

        The mean is given by
        
        .. math::

            \mu_{xy} = (\mu_x, \mu_y)^\top

        with :math:`\mu_y = \mathbb{E}[\mu_y(x)]`. The covariance is given by

        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                            \Sigma_x & \mathbb{E}[xy^\top] - \mu_x\mu_y^\top \\
                            \mathbb{E}[yx^\top] - \mu_y\mu_x^\top & \mathbb{E}[yy^\top] - \mu_y\mu_y^\top
                        \end{pmatrix}.

        Args:
            p_x: The density which we average over.

        Returns:
            Joint distribution of :math:`p(x,y)`.
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

        Args:
            p_x: Marginal Gaussian density over :math:`X`.

        Returns:
            Conditional density of :math:`p(X|Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        # Sigma_new = 0.5 * (Sigma_new + jnp.swapaxes(Sigma_new, -1, -2))
        cond_p_xy = conditional.ConditionalGaussianPDF(
            M=M_new,
            b=b_new,
            Sigma=Sigma_new,
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

        Args:
            p_x: Marginal Gaussian density over :math`X`.

        Returns:
            The marginal density :math:`p(Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y
    
    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs) -> Float[Array, "N"]:
        lb_quadratic_term = self.get_lb_quadratic_term(p_x, y)
        lb_log_det = self.get_lb_log_det(p_x)
        lb_log_p_y = -.5 * (lb_quadratic_term + lb_log_det + self.Dy * jnp.log(2. * jnp.pi))[0]
        return lb_log_p_y
    
    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        omega_dagger = jnp.sqrt(p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b))
        return omega_dagger

    @abstractmethod
    def k_func(self, p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        pass
    
    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        omega_dagger = lax.stop_gradient(vmap(lambda W: self._get_omega_dagger(p_x=p_x, W_i=W), in_axes=(0,))(self.W))
        k_omega = vmap(lambda W, omega: self.k_func(p_x=p_x, W_i=W, omega_dagger=omega))(self.W, omega_dagger)
        print(k_omega.shape)
        lower_bound_log_det = self.ln_det_Sigma + jnp.sum(k_omega, axis=0)
        return lower_bound_log_det
    
    def get_lb_quadratic_term(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]):
        projected_M = jnp.einsum('acb,acd->abd', self.Lambda, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.Lambda, y - self.b)
        homoscedastic_term = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=-projected_M, a_vec=projected_yb, B_mat=-self.M, b_vec=y - self.b)
        A_inv = jnp.einsum('abc,acd->abd', self.Lambda, self.A[:,:,:self.Dk])[0]
        get_lb_heteroscedastic_term = jnp.sum(vmap(lambda W, A_inv: self.get_lb_heteroscedastic_term_i(p_x, y, W, A_inv))(self.W, A_inv.T), axis=0)
        return homoscedastic_term - get_lb_heteroscedastic_term
    
    def get_lb_heteroscedastic_term_i(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], 
                                      W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = lax.stop_gradient(self._get_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i))
        # Quadratic integral
        return self._lower_bound_integrals(p_x, y, W_i, a_i, omega_star)
    
    @abstractmethod
    def _lower_bound_integrals(self, p_x: measure.GaussianMeasure, y: Float[Array, "N Dy"], 
                              W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"], compute_fourth_order: bool=False) -> Float[Array, "N"]:
        pass                           
    
    def _get_omega_star(self, p_x: pdf.GaussianPDF, y: jnp.ndarray, W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = self._get_omega_dagger(p_x=p_x, W_i=W_i)
        omega_dagger = omega_star + 1.
        cond_func = lambda val: jnp.max(jnp.abs(val[0] - val[1])) > 1e-5
        body_func = lambda val: (self._update_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=val[0]), val[0])
        omega_star, _ = lax.while_loop(cond_func, body_func, (omega_star, omega_dagger))
        return omega_star
    
    def _update_omega_star(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"]) -> Float[Array, "N"]:      
        quadratic_integral, quartic_integral = self._lower_bound_integrals(p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=omega_star, compute_fourth_order=True)
        omega_star = jnp.sqrt(quartic_integral / quadratic_integral)[0]
        return omega_star
    
@dataclass(kw_only=True)
class HeteroscedasticExpConditional(HeteroscedasticConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    
    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.exp(h)
    
        
    def _integrate_noise_diagonal(self, p_x: pdf.GaussianPDF) -> Float[Array, "N Dk"]:
        r"""Integrate the noise diagonal with respect to :math:`p(X)`.

        .. math::

            \int D(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        exp_h = factor.LinearFactor(nu=nu, ln_beta=ln_beta)
        D_int = p_x.multiply(exp_h, update_full=True).integrate()
        return D_int
    
    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        omega_dagger = jnp.sqrt(p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b))
        return omega_dagger

    def k_func(self, p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        Eh = p_x.integrate("(Ax+a)", A_mat=w, a_vec=b)[:,0]
        fomega = jnp.log(jnp.cosh(omega_dagger / 2.)) + jnp.log(2.)
        fprime_omega = .5 * jnp.tanh(omega_dagger / 2.)
        return .5 * Eh + fomega + .5 * fprime_omega  / omega_dagger * (Eh2 - omega_dagger ** 2)
    
    
    def _lower_bound_integrals(self, p_x: measure.GaussianMeasure, y: Float[Array, "N Dy"], 
                              W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"], compute_fourth_order: bool=False) -> Float[Array, "N"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        fomega = jnp.log(jnp.cosh(.5 * omega_star)) + jnp.log(2.)
        fprime_omega = .5 * jnp.tanh(.5 * omega_star)
        g_1 = fprime_omega / omega_star
        nu_1 = -((fprime_omega / omega_star)[:,None] * b - .5) * w
        ln_beta_1 = - fomega - .5 * fprime_omega / omega_star * (b ** 2 - omega_star ** 2) + .5 * b 
        lb_px_measure = p_x.hadamard(factor.OneRankFactor(v=w, g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        quadratic_integral = lb_px_measure.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        if compute_fourth_order:
            quartic_integral = lb_px_measure.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
            return quadratic_integral, quartic_integral
        else:
            return quadratic_integral
    
@dataclass(kw_only=True)
class HeteroscedasticCoshM1Conditional(HeteroscedasticConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    
    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.cosh(h) - 1.
              
    def _integrate_noise_diagonal(self, p_x: pdf.GaussianPDF) -> Float[Array, "N Dk"]:
        r"""Integrate the noise diagonal with respect to :math:`p(X)`.

        .. math::

            \int D(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta - jnp.log(2))
        exp_h_minus = factor.LinearFactor(nu=-nu, ln_beta=-ln_beta - jnp.log(2))
        D_int = (
        p_x.multiply(exp_h_plus, update_full=True).integrate()
        + p_x.multiply(exp_h_minus, update_full=True).integrate()
        - 1.0
        )
        return D_int
    
    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        omega_dagger = jnp.sqrt(p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b))
        return omega_dagger

    def k_func(self, p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        f_omega = jnp.log(jnp.cosh(omega_dagger))
        fprime_omega = jnp.tanh(omega_dagger)
        return f_omega + .5 * fprime_omega  / omega_dagger * (Eh2 - omega_dagger ** 2)
    
    def _lower_bound_integrals(self, p_x: measure.GaussianMeasure, y: Float[Array, "N Dy"], 
                            W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"], compute_fourth_order: bool=False) -> Float[Array, "N"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        f_omega = jnp.log(jnp.cosh(omega_star))
        fprime_omega = .5 * jnp.tanh(omega_star) / omega_star
        g_1 = 2. * fprime_omega
        nu_1 =  - 2. * fprime_omega[:,None] * b * w
        ln_beta_1 = - f_omega - fprime_omega * (b ** 2 - omega_star ** 2)
        lb_px_measure = p_x.hadamard(factor.OneRankFactor(v=jnp.tile(w, (omega_star.shape[0], 1)), g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)  
        exp_h_plus = factor.LinearFactor(nu=w, ln_beta=b - jnp.log(2))
        phi_plus = lb_px_measure.hadamard(exp_h_plus, update_full=True)
        quadratic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        exp_h_minus = factor.LinearFactor(nu=-w, ln_beta=- b - jnp.log(2))
        phi_minus = lb_px_measure.hadamard(exp_h_minus, update_full=True)
        quadratic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        quadratic_1 = lb_px_measure.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb) 
        quadratic_integral = quadratic_plus + quadratic_minus - quadratic_1
        if compute_fourth_order:
            quartic_integral = phi_plus.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
            quartic_integral += phi_minus.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
            quartic_integral -= lb_px_measure.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
            return quadratic_integral, quartic_integral
        else:
            return quadratic_integral       
    
    
        
@dataclass(kw_only=True)
class FullHSExpGaussianConditional(conditional.ConditionalGaussianPDF):
    r"""A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = A(I +  D(x))A^\top,

    and :math:`D_i(x) = \exp(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    exp_h_plus: factor.LinearFactor = field(init=False)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)

    def __post_init__(
        self,
    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        try:
            assert self.Dy <= self.Da
        except AssertionError:
            raise NotImplementedError("A must have at least as many rows as columns.")
        try:
            assert self.Dk <= self.Da
        except AssertionError:
            raise NotImplementedError("Diagonal matrix can have at most as many entries as A has columns.")
        
        self.Sigma = jnp.einsum("abc,adc->abd", self.A, self.A)
        self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        self._setup_noise_diagonal_functions()
        
        
    @property
    def R(self) -> int:
        """Number of conditionals (leading dimension)."""
        return self.M.shape[0]

    @property
    def Dy(self) -> int:
        r"""Dimensionality of :math:`Y`."""
        return self.M.shape[1]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.M.shape[2]

    @property
    def Da(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.A.shape[2]
    
    @property
    def Dk(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.W.shape[0]

    def _setup_noise_diagonal_functions(self):
        r"""Create the functions, that later need to be integrated over, i.e.

        .. math::

            \exp(h_i(z)) \text{ and } \exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta)

    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \Sigma + A_{:k} D(x) A_{:k}^\top,

        with :math:`D_i(x) = \cosh(h_i(x))-1` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        D_x = self.exp_h_plus(x).T # [N x Dk] # [N x Dy x Dk]
        Sigma = self.Sigma + jnp.einsum(
            "abc,dc->abd", jnp.einsum("ab,cb->cab", self.A[0,:,:self.Dk], D_x), self.A[0,:,:self.Dk]
        )
        if invert:
            G_x = D_x / (1 + D_x) # [N x Dk]
            A_inv = jnp.einsum('ab,bc->ac', self.Lambda[0], self.A[0,:,:self.Dk]) # [Dy x Dk] # [N x Dy x Dk]
            Lambda = self.Lambda - jnp.einsum(
                "abc,dc->abd", jnp.einsum("ab,cb->cab", A_inv, G_x), A_inv
            )
            ln_det_Sigma_y_x = self.ln_det_Sigma + jnp.sum(jnp.log(1 + D_x), axis=1)
            return Sigma, Lambda, ln_det_Sigma_y_x
        else:
            return Sigma

    def condition_on_x(self, x: Float[Array, "N Dx"], **kwargs) -> pdf.GaussianPDF:
        r"""Get Gaussian Density conditioned on :math:`X=x`.

        Args:
            x: Instances, the mu and Sigma should be conditioned on.

        Returns:
            The density conditioned on :math:`X=x`.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new, Lambda_new, ln_det_Sigma_new = self.get_conditional_cov(
            x, invert=True
        )
        # Sigma_new = .5 * (Sigma_new + jnp.swapaxes(Sigma_new, -2, -1))
        return pdf.GaussianPDF(
            Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        # Sigma_new = self.get_conditional_cov(x)
        # return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        r"""Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise AttributeError("HCCovGaussianConditional doesn't have function set_y.")

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = p_x.multiply(self.exp_h_plus, update_full=True).integrate()
        # rotation_mat_int = jnp.eye(self.Dy)[None] + jnp.einsum(
        #    "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        # )
        Sigma_int = self.Sigma + jnp.einsum(
            "ab,cb->ac", jnp.einsum("ab,b->ab", self.A[0,:,:self.Dk], D_int), self.A[0,:,:self.Dk]
        )[None]
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "1 Dy Dy"]]:
        r"""Compute the expected mean and covariance

        .. math::

            \mu_y = \mathbb{E}[y] = M \mathbb{E}[x] + b

        .. math::

            \Sigma_y = \mathbb{E}[yy'] - \mu_y \mu_y^\top = \sigma_x^2 I + \sum_i U_i \mathbb{E}[D_i(x)] U_i^\top + \mathbb{E}[\mu(x)\mu(x)^\top] - \mu_y \mu_y^\top

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density
        
        .. math::

            p(x,y) ~= {\cal N}(\mu_{xy},\Sigma_{xy}),

        The mean is given by
        
        .. math::

            \mu_{xy} = (\mu_x, \mu_y)^\top

        with :math:`\mu_y = \mathbb{E}[\mu_y(x)]`. The covariance is given by

        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                            \Sigma_x & \mathbb{E}[xy^\top] - \mu_x\mu_y^\top \\
                            \mathbb{E}[yx^\top] - \mu_y\mu_x^\top & \mathbb{E}[yy^\top] - \mu_y\mu_y^\top
                        \end{pmatrix}.

        Args:
            p_x: The density which we average over.

        Returns:
            Joint distribution of :math:`p(x,y)`.
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

        Args:
            p_x: Marginal Gaussian density over :math:`X`.

        Returns:
            Conditional density of :math:`p(X|Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        try:
            Lambda_y = invert_matrix(Sigma_y)[0]
        except FloatingPointError:
            print(jnp.linalg.eigvals(Sigma_y))
            self.get_expected_moments2(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        # Sigma_new = 0.5 * (Sigma_new + jnp.swapaxes(Sigma_new, -1, -2))
        try:
            cond_p_xy = conditional.ConditionalGaussianPDF(
                M=M_new,
                b=b_new,
                Sigma=Sigma_new,
            )
        except FloatingPointError:
            print(jnp.linalg.cholesky(Sigma_y[0]))
            print(Sigma_new)
            raise FloatingPointError(jnp.linalg.eigvals(Sigma_new))
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

        Args:
            p_x: Marginal Gaussian density over :math`X`.

        Returns:
            The marginal density :math:`p(Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y
    
    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs) -> Float[Array, "N"]:
        lb_quadratic_term = self.get_lb_quadratic_term(p_x, y)
        lb_log_det = self.get_lb_log_det(p_x)
        lb_log_p_y = -.5 * (lb_quadratic_term + lb_log_det + self.Dy * jnp.log(2. * jnp.pi))[0]
        return lb_log_p_y
    
    def get_lb_quadratic_term(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]):
        projected_M = jnp.einsum('acb,acd->abd', self.Lambda, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.Lambda, y - self.b)
        homoscedastic_term = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=-projected_M, a_vec=projected_yb, B_mat=-self.M, b_vec=y - self.b)
        A_inv = jnp.einsum('abc,acd->abd', self.Lambda, self.A[:,:,:self.Dk])[0]
        get_lb_heteroscedastic_term = jnp.sum(vmap(lambda W, A_inv: self.get_lb_heteroscedastic_term_i(p_x, y, W, A_inv))(self.W, A_inv.T), axis=0)
        return homoscedastic_term - get_lb_heteroscedastic_term
    
    def get_lb_heteroscedastic_term_i(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], 
                                      W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = lax.stop_gradient(self._get_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i))
        b = W_i[None,:1]
        w = W_i[None,1:]
        g_1 = .5 * jnp.tanh(.5 * omega_star) / omega_star
        nu_1 =  -.5 * ((jnp.tanh(.5 * omega_star) / omega_star)[:,None] * b - 1) * w
        ln_beta_1 = - jnp.log(jnp.cosh(.5 * omega_star)) - .25 * jnp.tanh(.5 * omega_star) / omega_star * (b ** 2 - omega_star ** 2) + .5 * b - jnp.log(2.)
        phi = p_x.hadamard(factor.OneRankFactor(v=w, g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        # Quadratic integral
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        G_i = phi.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        return G_i
    
    def _get_omega_star(self, p_x: pdf.GaussianPDF, y: jnp.ndarray, W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = self._get_omega_dagger(p_x=p_x, W_i=W_i)
        omega_dagger = omega_star + 1.
        cond_func = lambda val: jnp.max(jnp.abs(val[0] - val[1])) > 1e-5
        body_func = lambda val: (self._update_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=val[0]), val[0])
        omega_star, _ = lax.while_loop(cond_func, body_func, (omega_star, omega_dagger))
        return omega_star
    
    def _update_omega_star(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"]) -> Float[Array, "N"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        g_1 = .5 * jnp.tanh(.5 * omega_star) / omega_star
        nu_1 = -.5 * ((jnp.tanh(.5 * omega_star) / omega_star)[:,None] * b - 1) * w
        ln_beta_1 = - jnp.log(jnp.cosh(.5 * omega_star)) - .25 * jnp.tanh(.5 * omega_star) / omega_star * (b ** 2 - omega_star ** 2) + .5 * b - jnp.log(2.)
        phi = p_x.hadamard(factor.OneRankFactor(v=w, g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)

        # Quartic integral
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        quartic_integral = phi.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                            C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
        # Quadratic integral
        quadratic_integral = phi.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        omega_star = jnp.sqrt(quartic_integral / quadratic_integral)[0]
        return omega_star

    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        omega_dagger = jnp.sqrt(p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b))
        return omega_dagger

    @staticmethod
    def k_func(p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        Eh = p_x.integrate("(Ax+a)", A_mat=w, a_vec=b)[:,0]
        print(Eh.shape, Eh2.shape, omega_dagger.shape)
        return .5 * Eh + jnp.log(jnp.cosh(omega_dagger / 2.)) + .25 * jnp.tanh(omega_dagger / 2.) / omega_dagger * (Eh2 - omega_dagger ** 2) + jnp.log(2.)

    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        omega_dagger = lax.stop_gradient(vmap(lambda W: self._get_omega_dagger(p_x=p_x, W_i=W), in_axes=(0,))(self.W))
        k_omega = vmap(lambda W, omega: self.k_func(p_x=p_x, W_i=W, omega_dagger=omega))(self.W, omega_dagger)
        print(k_omega.shape)
        lower_bound_log_det = self.ln_det_Sigma + jnp.sum(k_omega, axis=0)
        return lower_bound_log_det


        
@dataclass(kw_only=True)
class FullHCCovGaussianConditional(conditional.ConditionalGaussianPDF):
    r"""A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = A(I +  D(x))A^\top,

    and :math:`D_i(x) = \cosh(h_i(x)) - 1` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    exp_h_plus: factor.LinearFactor = field(init=False)
    exp_h_minus: factor.LinearFactor = field(init=False)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)

    def __post_init__(
        self,
    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        try:
            assert self.Dy <= self.Da
        except AssertionError:
            raise NotImplementedError("A must have at least as many rows as columns.")
        try:
            assert self.Dk <= self.Da
        except AssertionError:
            raise NotImplementedError("Diagonal matrix can have at most as many entries as A has columns.")
        
        self.Sigma = jnp.einsum("abc,adc->abd", self.A, self.A)
        self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        self._setup_noise_diagonal_functions()
        
    @property
    def R(self) -> int:
        """Number of conditionals (leading dimension)."""
        return self.M.shape[0]

    @property
    def Dy(self) -> int:
        r"""Dimensionality of :math:`Y`."""
        return self.M.shape[1]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.M.shape[2]

    @property
    def Da(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.A.shape[2]
    
    @property
    def Dk(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.W.shape[0]

    def _setup_noise_diagonal_functions(self):
        r"""Create the functions, that later need to be integrated over, i.e.

        .. math::

            \exp(h_i(z)) \text{ and } \exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta - jnp.log(2))
        self.exp_h_minus = factor.LinearFactor(nu=-nu, ln_beta=-ln_beta - jnp.log(2))

    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \Sigma + A_{:k} D(x) A_{:k}^\top,

        with :math:`D_i(x) = \cosh(h_i(x))-1` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        D_x = (self.exp_h_plus(x) + self.exp_h_minus(x) - 1).T # [N x Dk] # [N x Dy x Dk]
        Sigma = self.Sigma + jnp.einsum(
            "abc,dc->abd", jnp.einsum("ab,cb->cab", self.A[0,:,:self.Dk], D_x), self.A[0,:,:self.Dk]
        )
        if invert:
            G_x = D_x / (1 + D_x) # [N x Dk]
            A_inv = jnp.einsum('ab,bc->ac', self.Lambda[0], self.A[0,:,:self.Dk]) # [Dy x Dk] # [N x Dy x Dk]
            Lambda = self.Lambda - jnp.einsum(
                "abc,dc->abd", jnp.einsum("ab,cb->cab", A_inv, G_x), A_inv
            )
            ln_det_Sigma_y_x = self.ln_det_Sigma + jnp.sum(jnp.log(1 + D_x), axis=1)
            return Sigma, Lambda, ln_det_Sigma_y_x
        else:
            return Sigma

    def condition_on_x(self, x: Float[Array, "N Dx"], **kwargs) -> pdf.GaussianPDF:
        r"""Get Gaussian Density conditioned on :math:`X=x`.

        Args:
            x: Instances, the mu and Sigma should be conditioned on.

        Returns:
            The density conditioned on :math:`X=x`.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new, Lambda_new, ln_det_Sigma_new = self.get_conditional_cov(
            x, invert=True
        )
        # Sigma_new = .5 * (Sigma_new + jnp.swapaxes(Sigma_new, -2, -1))
        return pdf.GaussianPDF(
            Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        # Sigma_new = self.get_conditional_cov(x)
        # return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        r"""Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise AttributeError("HCCovGaussianConditional doesn't have function set_y.")

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus, update_full=True).integrate()
            + p_x.multiply(self.exp_h_minus, update_full=True).integrate()
            - 1.0
        )
        # rotation_mat_int = jnp.eye(self.Dy)[None] + jnp.einsum(
        #    "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        # )
        Sigma_int = self.Sigma + jnp.einsum(
            "ab,cb->ac", jnp.einsum("ab,b->ab", self.A[0,:,:self.Dk], D_int), self.A[0,:,:self.Dk]
        )[None]
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "1 Dy Dy"]]:
        r"""Compute the expected mean and covariance

        .. math::

            \mu_y = \mathbb{E}[y] = M \mathbb{E}[x] + b

        .. math::

            \Sigma_y = \mathbb{E}[yy'] - \mu_y \mu_y^\top = \sigma_x^2 I + \sum_i U_i \mathbb{E}[D_i(x)] U_i^\top + \mathbb{E}[\mu(x)\mu(x)^\top] - \mu_y \mu_y^\top

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density
        
        .. math::

            p(x,y) ~= {\cal N}(\mu_{xy},\Sigma_{xy}),

        The mean is given by
        
        .. math::

            \mu_{xy} = (\mu_x, \mu_y)^\top

        with :math:`\mu_y = \mathbb{E}[\mu_y(x)]`. The covariance is given by

        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                            \Sigma_x & \mathbb{E}[xy^\top] - \mu_x\mu_y^\top \\
                            \mathbb{E}[yx^\top] - \mu_y\mu_x^\top & \mathbb{E}[yy^\top] - \mu_y\mu_y^\top
                        \end{pmatrix}.

        Args:
            p_x: The density which we average over.

        Returns:
            Joint distribution of :math:`p(x,y)`.
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

        Args:
            p_x: Marginal Gaussian density over :math:`X`.

        Returns:
            Conditional density of :math:`p(X|Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        try:
            Lambda_y = invert_matrix(Sigma_y)[0]
        except FloatingPointError:
            print(jnp.linalg.eigvals(Sigma_y))
            self.get_expected_moments2(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        # Sigma_new = 0.5 * (Sigma_new + jnp.swapaxes(Sigma_new, -1, -2))
        try:
            cond_p_xy = conditional.ConditionalGaussianPDF(
                M=M_new,
                b=b_new,
                Sigma=Sigma_new,
            )
        except FloatingPointError:
            print(jnp.linalg.cholesky(Sigma_y[0]))
            print(Sigma_new)
            raise FloatingPointError(jnp.linalg.eigvals(Sigma_new))
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

        Args:
            p_x: Marginal Gaussian density over :math`X`.

        Returns:
            The marginal density :math:`p(Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y
    
    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs) -> Float[Array, "N"]:
        lb_quadratic_term = self.get_lb_quadratic_term(p_x, y)
        lb_log_det = self.get_lb_log_det(p_x)
        lb_log_p_y = -.5 * (lb_quadratic_term + lb_log_det + self.Dy * jnp.log(2. * jnp.pi))[0]
        return lb_log_p_y
    
    def get_lb_quadratic_term(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]):
        projected_M = jnp.einsum('acb,acd->abd', self.Lambda, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.Lambda, y - self.b)
        homoscedastic_term = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=-projected_M, a_vec=projected_yb, B_mat=-self.M, b_vec=y - self.b)
        A_inv = jnp.einsum('abc,acd->abd', self.Lambda, self.A[:,:,:self.Dk])[0]
        get_lb_heteroscedastic_term = jnp.sum(vmap(lambda W, A_inv: self.get_lb_heteroscedastic_term_i(p_x, y, W, A_inv))(self.W, A_inv.T), axis=0)
        return homoscedastic_term - get_lb_heteroscedastic_term
    
    def get_lb_heteroscedastic_term_i(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], 
                                      W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = lax.stop_gradient(self._get_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i))
        b = W_i[None,:1]
        w = W_i[None,1:]
        g_1 = jnp.tanh(omega_star) / omega_star
        nu_1 = - (jnp.tanh(omega_star) / omega_star)[:,None] * b * w
        ln_beta_1 = - jnp.log(jnp.cosh(omega_star)) - .5 * jnp.tanh(omega_star) / omega_star * (b ** 2 - omega_star ** 2)
        phi_1 = p_x.hadamard(factor.OneRankFactor(v=jnp.tile(w, (omega_star.shape[0], 1)), g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        phi_plus = phi_1.hadamard(factor.LinearFactor(nu=w, ln_beta=b-jnp.log(2.)), update_full=True)
        phi_minus = phi_1.hadamard(factor.LinearFactor(nu=-w, ln_beta=-b-jnp.log(2.)), update_full=True)
        # Quadratic integral
        U_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        U_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        quadratic_1 = phi_1.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)

        G_i = - quadratic_1 + quadratic_plus + quadratic_minus
        return G_i
    
    def _get_omega_star(self, p_x: pdf.GaussianPDF, y: jnp.ndarray, W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = self._get_omega_dagger(p_x=p_x, W_i=W_i)
        omega_dagger = omega_star + 1.
        cond_func = lambda val: jnp.max(jnp.abs(val[0] - val[1])) > 1e-5
        body_func = lambda val: (self._update_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=val[0]), val[0])
        omega_star, _ = lax.while_loop(cond_func, body_func, (omega_star, omega_dagger))
        return omega_star
    
    def _update_omega_star(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"]) -> Float[Array, "N"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        
        g_1 = jnp.tanh(omega_star) / omega_star
        nu_1 = - (jnp.tanh(omega_star) / omega_star)[:,None] * b * w
        ln_beta_1 = - jnp.log(jnp.cosh(omega_star)) - .5 * jnp.tanh(omega_star) / omega_star * (b[0] ** 2 - omega_star ** 2)
        phi_1 = p_x.hadamard(factor.OneRankFactor(v=jnp.tile(w, (omega_star.shape[0], 1)), g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        phi_plus = phi_1.hadamard(factor.LinearFactor(nu=w, ln_beta=b-jnp.log(2.)), update_full=True)
        phi_minus = phi_1.hadamard(factor.LinearFactor(nu=-w, ln_beta=-b-jnp.log(2.)), update_full=True)

        # Quartic integral
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        
        quartic_1 = phi_1.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                    C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
        quartic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                        C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)
        quartic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                            C_mat=-a_projected_M, c_vec=a_projected_yb, D_mat=-a_projected_M, d_vec=a_projected_yb)

        quartic_integral = - quartic_1 + quartic_plus + quartic_minus
        # Quadratic integral
        quadratic_1 = phi_1.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        quadratic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        quadratic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)", A_mat=-a_projected_M, a_vec=a_projected_yb, B_mat=-a_projected_M, b_vec=a_projected_yb)
        quadratic_integral = - quadratic_1 + quadratic_plus + quadratic_minus
        omega_star = jnp.sqrt(quartic_integral / quadratic_integral)[0]
        return omega_star

    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        omega_dagger = jnp.sqrt(p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b))
        return omega_dagger

    @staticmethod
    def k_func(p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        return jnp.log(jnp.cosh(omega_dagger)) + .5 * jnp.tanh(omega_dagger) / omega_dagger * (Eh2 - omega_dagger ** 2)

    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        omega_dagger = lax.stop_gradient(vmap(lambda W: self._get_omega_dagger(p_x=p_x, W_i=W), in_axes=(0,))(self.W))
        k_omega = vmap(lambda W, omega: self.k_func(p_x=p_x, W_i=W, omega_dagger=omega))(self.W, omega_dagger)
        lower_bound_log_det = self.ln_det_Sigma + jnp.sum(k_omega, axis=0)
        return lower_bound_log_det

    
    def f(self, h: Float[Array, "N"]) -> Float[Array, "N"]:
        """Compute the function

        .. math::

        f(h) = \cosh(h) - 1

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated functions
        """
        return jnp.cosh(h) - 1

    def f_prime(self, h: Float[Array, "N"]) -> Float[Array, "N"]:
        """Computes the derivative of f

        .. math::

            f^\prime(h) = 2 * \beta * \sinh(h)

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated derivative functions.
        """
        return jnp.sinh(h)

    def g(self, omega: Float[Array, "N"]) -> Float[Array, "N"]:
        r"""Computes the function

        .. math::

            g(\omega) = f'(\omega) / (sigma_x^2 + f(\omega)) / |\omega|

            for the variational bound

        Args:
            omega: Free variational parameter.
            beta: Scaling factor.
            sigma_x: Noise parameter.

        Returns:
            Evaluated function.
        """
        return jnp.tanh(omega) / jnp.abs(omega)


        
        
@dataclass(kw_only=True)
class FullHCCovGaussianConditional_old_new(conditional.ConditionalGaussianPDF):
    r"""A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

    and :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    Sigma: Float[Array, "1 Dy Dy"]
    U: Float[Array, "Dy Du"]
    W: Float[Array, "Du Dx+1"]
    L: Float[Array, "1 Dy Dy"] = field(default=None)
    exp_h_plus: factor.LinearFactor = field(init=False)
    exp_h_minus: factor.LinearFactor = field(init=False)
    L_inv: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: factor.LinearFactor = field(init=False)

    def __post_init__(
        self,
    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        if self.Dy < self.Du:
            raise NotImplementedError(
                "There must be less vectors U than dimensionality of Y."
            )
        if self.L is None:
            cho_factor = jsc.linalg.cho_factor(self.Sigma[0])
            self.L = jnp.array([cho_factor[0]])

        self.L_inv = jnp.array(
            [
                jsc.linalg.solve_triangular(
                    self.L[0], jnp.eye(self.L.shape[1]), lower=False
                )
            ]
        )
        self.ln_det_Sigma = jnp.array(
            [2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.L[0]))))]
        )
        # self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        self._setup_noise_diagonal_functions()

    def update_chol(self, L: Float[Array, "1 Dy Dy"]):
        self.L = L
        self.Sigma = jnp.einsum("abc,abd->acd", self.L, self.L) # L**2
        self.L_inv = jnp.array(
            [jsc.linalg.solve_triangular(self.L[0], jnp.eye(self.L.shape[1]))]
        )
        #self.L_inv = jnp.array([jnp.diag(1.0 / jnp.diag(self.L[0]))])
        self.ln_det_Sigma = jnp.array(
            [2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.L[0]))))]
        )

    @classmethod
    def init_with_cholesky(
        cls,
        M: Float[Array, "1 Dy Dx"],
        b: Float[Array, "1 Dy"],
        L: Float[Array, "1 Dy Dy"],
        U: Float[Array, "Dy Du"],
        W: Float[Array, "Du Dx+1"],
    ):
        r"""Initialize the conditional with a Cholesky factor of the covariance matrix.

        Args:
            M: Matrix in the mean function.
            b: Vector in the conditional mean function.
            L: Cholesky factor of the covariance matrix.
            U: Othonormal vectors for low rank noise part.
            W: Noise weights for low rank components (w_i & b_i).
            beta: Scaling for low rank noise components.

        Returns:
            FullHCCovGaussianConditional: The initialized conditional.
        """
        Sigma = jnp.einsum("abc,abd->acd", L, L)
        return cls(M=M, b=b, Sigma=Sigma, U=U, W=W, L=L)

    @property
    def R(self) -> int:
        """Number of conditionals (leading dimension)."""
        return self.M.shape[0]

    @property
    def Dy(self) -> int:
        r"""Dimensionality of :math:`Y`."""
        return self.M.shape[1]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.M.shape[2]

    @property
    def Du(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.U.shape[1]

    def _setup_noise_diagonal_functions(self):
        r"""Create the functions, that later need to be integrated over, i.e.

        .. math::

            \exp(h_i(z)) \text{ and } \exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta - jnp.log(2))
        self.exp_h_minus = factor.LinearFactor(nu=-nu, ln_beta=-ln_beta - jnp.log(2))

    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

        with :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        D_x = (self.exp_h_plus(x) + self.exp_h_minus(x) - 1).T
        rotation_mat = jnp.eye(self.Dy)[None] + jnp.einsum(
            "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, D_x), self.U
        )
        Sigma = jnp.einsum(
            "acb,abd->acd", jnp.einsum("abc,abd->acd", self.L, rotation_mat), self.L
        )
        if invert:
            G_x = D_x / (1 + D_x)
            rotation_mat_inv = jnp.eye(self.Dy)[None] - jnp.einsum(
                "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, G_x), self.U
            )
            Lambda = jnp.einsum(
                "acb,adb->acd",
                jnp.einsum("acb,abd->acd", self.L_inv, rotation_mat_inv),
                self.L_inv,
            )
            ln_det_Sigma_y_x = self.ln_det_Sigma + jnp.sum(jnp.log(1 + D_x), axis=1)
            return Sigma, Lambda, ln_det_Sigma_y_x
        else:
            return Sigma

    def condition_on_x(self, x: Float[Array, "N Dx"], **kwargs) -> pdf.GaussianPDF:
        r"""Get Gaussian Density conditioned on :math:`X=x`.

        Args:
            x: Instances, the mu and Sigma should be conditioned on.

        Returns:
            The density conditioned on :math:`X=x`.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new, Lambda_new, ln_det_Sigma_new = self.get_conditional_cov(
            x, invert=True
        )
        # Sigma_new = .5 * (Sigma_new + jnp.swapaxes(Sigma_new, -2, -1))
        return pdf.GaussianPDF(
            Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        # Sigma_new = self.get_conditional_cov(x)
        # return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        r"""Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise AttributeError("HCCovGaussianConditional doesn't have function set_y.")

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus, update_full=True).integrate()
            + p_x.multiply(self.exp_h_minus, update_full=True).integrate()
            - 1.0
        )
        # rotation_mat_int = jnp.eye(self.Dy)[None] + jnp.einsum(
        #    "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        # )
        rotation_mat_int = jnp.eye(self.Dy) + jnp.einsum(
            "a,abc->bc", D_int, jnp.einsum("ba,ca->abc", self.U, self.U)
        )
        Sigma_int = jnp.einsum(
            "acb,abd->acd", jnp.einsum("abc,bd->acd", self.L, rotation_mat_int), self.L
        )
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "1 Dy Dy"]]:
        r"""Compute the expected mean and covariance

        .. math::

            \mu_y = \mathbb{E}[y] = M \mathbb{E}[x] + b

        .. math::

            \Sigma_y = \mathbb{E}[yy'] - \mu_y \mu_y^\top = \sigma_x^2 I + \sum_i U_i \mathbb{E}[D_i(x)] U_i^\top + \mathbb{E}[\mu(x)\mu(x)^\top] - \mu_y \mu_y^\top

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density
        
        .. math::

            p(x,y) ~= {\cal N}(\mu_{xy},\Sigma_{xy}),

        The mean is given by
        
        .. math::

            \mu_{xy} = (\mu_x, \mu_y)^\top

        with :math:`\mu_y = \mathbb{E}[\mu_y(x)]`. The covariance is given by

        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                            \Sigma_x & \mathbb{E}[xy^\top] - \mu_x\mu_y^\top \\
                            \mathbb{E}[yx^\top] - \mu_y\mu_x^\top & \mathbb{E}[yy^\top] - \mu_y\mu_y^\top
                        \end{pmatrix}.

        Args:
            p_x: The density which we average over.

        Returns:
            Joint distribution of :math:`p(x,y)`.
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

        Args:
            p_x: Marginal Gaussian density over :math:`X`.

        Returns:
            Conditional density of :math:`p(X|Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        try:
            Lambda_y = invert_matrix(Sigma_y)[0]
        except FloatingPointError:
            print(jnp.linalg.eigvals(Sigma_y))
            self.get_expected_moments2(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        # Sigma_new = 0.5 * (Sigma_new + jnp.swapaxes(Sigma_new, -1, -2))
        try:
            cond_p_xy = conditional.ConditionalGaussianPDF(
                M=M_new,
                b=b_new,
                Sigma=Sigma_new,
            )
        except FloatingPointError:
            print(jnp.linalg.cholesky(Sigma_y[0]))
            print(Sigma_new)
            raise FloatingPointError(jnp.linalg.eigvals(Sigma_new))
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

        Args:
            p_x: Marginal Gaussian density over :math`X`.

        Returns:
            The marginal density :math:`p(Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y
    
    def integrate_log_conditional_y(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs) -> Float[Array, "N"]:
        lb_quadratic_term = self.get_lb_quadratic_term(p_x, y)
        lb_log_det = self.get_lb_log_det(p_x)
        lb_log_p_y = -.5 * (lb_quadratic_term + lb_log_det + self.Dy * jnp.log(2. * jnp.pi))[0]
        return lb_log_p_y
    
    def get_lb_quadratic_term(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]):
        projected_M = jnp.einsum('acb,acd->abd', self.L_inv, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.L_inv, y - self.b)
        homoscedastic_term = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=-projected_M, a_vec=projected_yb, B_mat=-projected_M, b_vec=projected_yb)
        get_lb_heteroscedastic_term = jnp.sum(vmap(lambda W, U: self.get_lb_heteroscedastic_term_i(p_x, y, W, U))(self.W, self.U.T), axis=0)
        return homoscedastic_term - get_lb_heteroscedastic_term
    
    def get_lb_heteroscedastic_term_i(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], 
                                      W_i: Float[Array, "Dx+1"], U_i: Float[Array, "Du"]):
        omega_star = lax.stop_gradient(self._get_omega_star(p_x=p_x, y=y, W_i=W_i, U_i=U_i))
        b = W_i[None,:1]
        w = W_i[None,1:]
        g_1 = jnp.tanh(omega_star) / omega_star
        nu_1 = - (jnp.tanh(omega_star) / omega_star)[:,None] * b * w
        ln_beta_1 = - jnp.log(jnp.cosh(omega_star)) - .5 * jnp.tanh(omega_star) / omega_star * (b ** 2 - omega_star ** 2)
        phi_1 = p_x.hadamard(factor.OneRankFactor(v=jnp.tile(w, (omega_star.shape[0], 1)), g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        phi_plus = phi_1.hadamard(factor.LinearFactor(nu=w, ln_beta=b-jnp.log(2.)), update_full=True)
        phi_minus = phi_1.hadamard(factor.LinearFactor(nu=-w, ln_beta=-b-jnp.log(2.)), update_full=True)
        # Quadratic integral
        projected_M = jnp.einsum('acb,acd->abd', self.L_inv, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.L_inv, y - self.b)
        U_projected_M = jnp.einsum('ab,cad->cbd', U_i[:,None], projected_M)
        U_projected_yb = jnp.einsum('ab,ca->cb', U_i[:,None], projected_yb)
        quadratic_1 = phi_1.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)

        G_i = - quadratic_1 + quadratic_plus + quadratic_minus
        return G_i
    
    def _get_omega_star(self, p_x: pdf.GaussianPDF, y: jnp.ndarray, W_i: Float[Array, "Dx+1"], U_i: Float[Array, "Du"]):
        omega_star = self._get_omega_dagger(p_x=p_x, W_i=W_i)
        omega_dagger = omega_star + 1.
        cond_func = lambda val: jnp.max(jnp.abs(val[0] - val[1])) > 1e-5
        body_func = lambda val: (self._update_omega_star(p_x=p_x, y=y, W_i=W_i, U_i=U_i, omega_star=val[0]), val[0])
        omega_star, _ = lax.while_loop(cond_func, body_func, (omega_star, omega_dagger))
        return omega_star
    
    def _update_omega_star(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], W_i: Float[Array, "Dx+1"], U_i: Float[Array, "Du"], omega_star: Float[Array, "N"]) -> Float[Array, "N"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        
        g_1 = jnp.tanh(omega_star) / omega_star
        nu_1 = - (jnp.tanh(omega_star) / omega_star)[:,None] * b * w
        ln_beta_1 = - jnp.log(jnp.cosh(omega_star)) - .5 * jnp.tanh(omega_star) / omega_star * (b[0] ** 2 - omega_star ** 2)
        phi_1 = p_x.hadamard(factor.OneRankFactor(v=jnp.tile(w, (omega_star.shape[0], 1)), g=g_1, nu=nu_1, ln_beta=ln_beta_1), update_full=True)
        phi_plus = phi_1.hadamard(factor.LinearFactor(nu=w, ln_beta=b-jnp.log(2.)), update_full=True)
        phi_minus = phi_1.hadamard(factor.LinearFactor(nu=-w, ln_beta=-b-jnp.log(2.)), update_full=True)

        # Quartic integral

        projected_M = jnp.einsum('acb,acd->abd', self.L_inv, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.L_inv, y - self.b)
        U_projected_M = jnp.einsum('ab,cad->cbd', U_i[:,None], projected_M)
        U_projected_yb = jnp.einsum('ab,ca->cb', U_i[:,None], projected_yb)
        
        quartic_1 = phi_1.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                    C_mat=-U_projected_M, c_vec=U_projected_yb, D_mat=-U_projected_M, d_vec=U_projected_yb)
        quartic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                        C_mat=-U_projected_M, c_vec=U_projected_yb, D_mat=-U_projected_M, d_vec=U_projected_yb)
        quartic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)", A_mat=w, a_vec=b, B_mat=w, b_vec=b, 
                                            C_mat=-U_projected_M, c_vec=U_projected_yb, D_mat=-U_projected_M, d_vec=U_projected_yb)

        quartic_integral = - quartic_1 + quartic_plus + quartic_minus
        # Quadratic integral
        quadratic_1 = phi_1.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_plus = phi_plus.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_minus = phi_minus.integrate("(Ax+a)'(Bx+b)", A_mat=-U_projected_M, a_vec=U_projected_yb, B_mat=-U_projected_M, b_vec=U_projected_yb)
        quadratic_integral = - quadratic_1 + quadratic_plus + quadratic_minus
        omega_star = jnp.sqrt(quartic_integral / quadratic_integral)[0]
        return omega_star

    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        omega_dagger = jnp.sqrt(p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b))
        return omega_dagger

    @staticmethod
    def k_func(p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        b = W_i[None,:1]
        w = W_i[None,1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        return jnp.log(jnp.cosh(omega_dagger)) + .5 * jnp.tanh(omega_dagger) / omega_dagger * (Eh2 - omega_dagger ** 2)

    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        omega_dagger = lax.stop_gradient(vmap(lambda W: self._get_omega_dagger(p_x=p_x, W_i=W), in_axes=(0,))(self.W))
        k_omega = vmap(lambda W, omega: self.k_func(p_x=p_x, W_i=W, omega_dagger=omega))(self.W, omega_dagger)
        lower_bound_log_det = self.ln_det_Sigma + jnp.sum(k_omega, axis=0)
        return lower_bound_log_det

    
    def f(self, h: Float[Array, "N"]) -> Float[Array, "N"]:
        """Compute the function

        .. math::

        f(h) = \cosh(h) - 1

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated functions
        """
        return jnp.cosh(h) - 1

    def f_prime(self, h: Float[Array, "N"]) -> Float[Array, "N"]:
        """Computes the derivative of f

        .. math::

            f^\prime(h) = 2 * \beta * \sinh(h)

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated derivative functions.
        """
        return jnp.sinh(h)

    def g(self, omega: Float[Array, "N"]) -> Float[Array, "N"]:
        r"""Computes the function

        .. math::

            g(\omega) = f'(\omega) / (sigma_x^2 + f(\omega)) / |\omega|

            for the variational bound

        Args:
            omega: Free variational parameter.
            beta: Scaling factor.
            sigma_x: Noise parameter.

        Returns:
            Evaluated function.
        """
        return jnp.tanh(omega) / jnp.abs(omega)



@dataclass(kw_only=True)
class FullHCCovGaussianConditional_old(conditional.ConditionalGaussianPDF):
    r"""A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

    and :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    Sigma: Float[Array, "1 Dy Dy"]
    U: Float[Array, "Dy Du"]
    W: Float[Array, "Du Dx+1"]
    L: Float[Array, "1 Dy Dy"] = field(default=None)
    exp_h_plus: factor.LinearFactor = field(init=False)
    exp_h_minus: factor.LinearFactor = field(init=False)
    L_inv: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: factor.LinearFactor = field(init=False)

    def __post_init__(
        self,
    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        if self.Dy < self.Du:
            raise NotImplementedError(
                "There must be less vectors U than dimensionality of Y."
            )
        if self.L is None:
            cho_factor = jsc.linalg.cho_factor(self.Sigma[0])
            self.L = jnp.array([cho_factor[0]])

        self.L_inv = jnp.array(
            [
                jsc.linalg.solve_triangular(
                    self.L[0], jnp.eye(self.L.shape[1]), lower=False
                )
            ]
        )
        self.ln_det_Sigma = jnp.array(
            [2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.L[0]))))]
        )
        # self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        self._setup_noise_diagonal_functions()

    def update_chol(self, L: Float[Array, "1 Dy Dy"]):
        self.L = L
        self.Sigma = jnp.einsum("abc,abd->acd", self.L, self.L) # L**2
        self.L_inv = jnp.array(
            [jsc.linalg.solve_triangular(self.L[0], jnp.eye(self.L.shape[1]))]
        )
        #self.L_inv = jnp.array([jnp.diag(1.0 / jnp.diag(self.L[0]))])
        self.ln_det_Sigma = jnp.array(
            [2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.L[0]))))]
        )

    @classmethod
    def init_with_cholesky(
        cls,
        M: Float[Array, "1 Dy Dx"],
        b: Float[Array, "1 Dy"],
        L: Float[Array, "1 Dy Dy"],
        U: Float[Array, "Dy Du"],
        W: Float[Array, "Du Dx+1"],
    ):
        r"""Initialize the conditional with a Cholesky factor of the covariance matrix.

        Args:
            M: Matrix in the mean function.
            b: Vector in the conditional mean function.
            L: Cholesky factor of the covariance matrix.
            U: Othonormal vectors for low rank noise part.
            W: Noise weights for low rank components (w_i & b_i).
            beta: Scaling for low rank noise components.

        Returns:
            FullHCCovGaussianConditional: The initialized conditional.
        """
        Sigma = jnp.einsum("abc,abd->acd", L, L)
        return cls(M=M, b=b, Sigma=Sigma, U=U, W=W, L=L)

    @property
    def R(self) -> int:
        """Number of conditionals (leading dimension)."""
        return self.M.shape[0]

    @property
    def Dy(self) -> int:
        r"""Dimensionality of :math:`Y`."""
        return self.M.shape[1]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.M.shape[2]

    @property
    def Du(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.U.shape[1]

    def _setup_noise_diagonal_functions(self):
        r"""Create the functions, that later need to be integrated over, i.e.

        .. math::

            \exp(h_i(z)) \text{ and } \exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta - jnp.log(2))
        self.exp_h_minus = factor.LinearFactor(nu=-nu, ln_beta=-ln_beta - jnp.log(2))

    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

        with :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        D_x = (self.exp_h_plus(x) + self.exp_h_minus(x) - 1).T
        rotation_mat = jnp.eye(self.Dy)[None] + jnp.einsum(
            "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, D_x), self.U
        )
        Sigma = jnp.einsum(
            "acb,abd->acd", jnp.einsum("abc,abd->acd", self.L, rotation_mat), self.L
        )
        if invert:
            G_x = D_x / (1 + D_x)
            rotation_mat_inv = jnp.eye(self.Dy)[None] - jnp.einsum(
                "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, G_x), self.U
            )
            Lambda = jnp.einsum(
                "acb,adb->acd",
                jnp.einsum("acb,abd->acd", self.L_inv, rotation_mat_inv),
                self.L_inv,
            )
            ln_det_Sigma_y_x = self.ln_det_Sigma + jnp.sum(jnp.log(1 + D_x), axis=1)
            return Sigma, Lambda, ln_det_Sigma_y_x
        else:
            return Sigma

    def condition_on_x(self, x: Float[Array, "N Dx"], **kwargs) -> pdf.GaussianPDF:
        r"""Get Gaussian Density conditioned on :math:`X=x`.

        Args:
            x: Instances, the mu and Sigma should be conditioned on.

        Returns:
            The density conditioned on :math:`X=x`.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new, Lambda_new, ln_det_Sigma_new = self.get_conditional_cov(
            x, invert=True
        )
        # Sigma_new = .5 * (Sigma_new + jnp.swapaxes(Sigma_new, -2, -1))
        return pdf.GaussianPDF(
            Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        # Sigma_new = self.get_conditional_cov(x)
        # return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        r"""Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise AttributeError("HCCovGaussianConditional doesn't have function set_y.")

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus, update_full=True).integrate()
            + p_x.multiply(self.exp_h_minus, update_full=True).integrate()
            - 1.0
        )
        # rotation_mat_int = jnp.eye(self.Dy)[None] + jnp.einsum(
        #    "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        # )
        rotation_mat_int = jnp.eye(self.Dy) + jnp.einsum(
            "a,abc->bc", D_int, jnp.einsum("ba,ca->abc", self.U, self.U)
        )
        Sigma_int = jnp.einsum(
            "acb,abd->acd", jnp.einsum("abc,bd->acd", self.L, rotation_mat_int), self.L
        )
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "1 Dy Dy"]]:
        r"""Compute the expected mean and covariance

        .. math::

            \mu_y = \mathbb{E}[y] = M \mathbb{E}[x] + b

        .. math::

            \Sigma_y = \mathbb{E}[yy'] - \mu_y \mu_y^\top = \sigma_x^2 I + \sum_i U_i \mathbb{E}[D_i(x)] U_i^\top + \mathbb{E}[\mu(x)\mu(x)^\top] - \mu_y \mu_y^\top

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density
        
        .. math::

            p(x,y) ~= {\cal N}(\mu_{xy},\Sigma_{xy}),

        The mean is given by
        
        .. math::

            \mu_{xy} = (\mu_x, \mu_y)^\top

        with :math:`\mu_y = \mathbb{E}[\mu_y(x)]`. The covariance is given by

        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                            \Sigma_x & \mathbb{E}[xy^\top] - \mu_x\mu_y^\top \\
                            \mathbb{E}[yx^\top] - \mu_y\mu_x^\top & \mathbb{E}[yy^\top] - \mu_y\mu_y^\top
                        \end{pmatrix}.

        Args:
            p_x: The density which we average over.

        Returns:
            Joint distribution of :math:`p(x,y)`.
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

        Args:
            p_x: Marginal Gaussian density over :math:`X`.

        Returns:
            Conditional density of :math:`p(X|Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        try:
            Lambda_y = invert_matrix(Sigma_y)[0]
        except FloatingPointError:
            print(jnp.linalg.eigvals(Sigma_y))
            self.get_expected_moments2(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        # Sigma_new = 0.5 * (Sigma_new + jnp.swapaxes(Sigma_new, -1, -2))
        try:
            cond_p_xy = conditional.ConditionalGaussianPDF(
                M=M_new,
                b=b_new,
                Sigma=Sigma_new,
            )
        except FloatingPointError:
            print(jnp.linalg.cholesky(Sigma_y[0]))
            print(Sigma_new)
            raise FloatingPointError(jnp.linalg.eigvals(Sigma_new))
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

        Args:
            p_x: Marginal Gaussian density over :math`X`.

        Returns:
            The marginal density :math:`p(Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y

    def integrate_log_conditional_y(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs
    ) -> Float[Array, "N"]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral evaluated for of :math:`Y=y`.
        """
        vec = y - self.b
        vec_projected = jnp.einsum("ba,cb->ca", self.L_inv[0], vec)
        M_projected = jnp.einsum("ba,cbd->cad", self.L_inv[0], self.M)
        E_epsilon2 = p_x.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-M_projected,
            a_vec=vec_projected,
            B_mat=-M_projected,
            b_vec=vec_projected,
        )
        """
        W_i, u_i = self.W[0], self.U[:, 0]
        omega_star_i, omega_dagger_i, _ = lax.stop_gradient(
            self._get_omega_star_i(W_i, u_i, p_x, y)
        )
        uRu_i, log_lb_sum_i = self._get_lb_i(
            W_i, u_i, omega_star_i, omega_dagger_i, p_x, y
        )
        uRu_i_delta1, log_lb_sum_i_delta1 = self._get_lb_i(
            W_i, u_i, omega_star_i + 1e-4, omega_dagger_i, p_x, y
        )
        print(uRu_i - uRu_i_delta1)
        uRu_i_delta2, log_lb_sum_i_delta2 = self._get_lb_i(
            W_i, u_i, omega_star_i - 1e-4, omega_dagger_i, p_x, y
        )
        print(uRu_i - uRu_i_delta2)
        """

        def scan_body_function(carry, args_i):
            W_i, u_i = args_i
            omega_star_i, omega_dagger_i, _ = lax.stop_gradient(
                self._get_omega_star_i(W_i, u_i, p_x, y)
            )
            uRu_i, log_lb_sum_i = self._get_lb_i(
                W_i, u_i, omega_star_i, omega_dagger_i, p_x, y
            )
            result = (uRu_i, log_lb_sum_i)
            return carry, result

        _, result = lax.scan(scan_body_function, None, (self.W, self.U.T))
        uRu, log_lb_sum = result
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0) + self.ln_det_Sigma[0]
        log_int_y = -0.5 * (E_epsilon2 - E_D_inv_epsilon2)
        # determinant part
        log_int_y = (
            log_int_y - 0.5 * E_ln_sigma2_f - 0.5 * self.Dy * jnp.log(2.0 * jnp.pi)
        )
        return log_int_y

    def _get_lb_i(
        self,
        W_i: Float[Array, "Dx+1"],
        u_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        omega_dagger,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
        # phi = pdf.GaussianPDF(**phi_dict)
        # beta = self.beta[iu:iu + 1]
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        G = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (G, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]

        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        f_omega_dagger = self.f(omega_dagger)
        g_omega_dagger = self.g(omega_dagger)
        log_lb = jnp.log(1 + f_omega_dagger) + 0.5 * g_omega_dagger * (
            Eh2 - omega_dagger**2
        )
        g_omega = self.g(omega_star)
        nu = -g_omega[:, None] * b_i * w_i
        nu_plus = w_i + nu
        nu_minus = -w_i + nu
        f_omega_star = self.f(omega_star)
        ln_beta = -jnp.where(
            jnp.isclose(f_omega_star, 0), 0, jnp.log(1 + f_omega_star)
        ) - 0.5 * g_omega * (b_i**2 - omega_star**2)
        ln_beta_plus = ln_beta + b_i - jnp.log(2)
        ln_beta_minus = ln_beta - b_i - jnp.log(2)
        # Create OneRankFactors
        g_omega = jnp.clip(g_omega, a_min=1e-4)
        exp_factor_plus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        one_factor = factor.OneRankFactor(v=v, g=g_omega, nu=nu, ln_beta=ln_beta)
        # Create the two measures
        exp_phi_plus = p_x.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = p_x.hadamard(exp_factor_minus, update_full=True)
        phi_one = p_x.hadamard(one_factor, update_full=True)

        mat1 = -self.M[0]
        vec1 = y - self.b[0]
        vec1_projected = jnp.einsum("ba,cb->ca", self.L_inv[0], vec1)
        mat1_projected = jnp.einsum("ba,bc->ac", self.L_inv[0], mat1)
        G_plus = exp_phi_plus.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=mat1_projected,
            a_vec=vec1_projected,
            B_mat=mat1_projected,
            b_vec=vec1_projected,
        )
        G_minus = exp_phi_minus.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=mat1_projected,
            a_vec=vec1_projected,
            B_mat=mat1_projected,
            b_vec=vec1_projected,
        )
        G_one = phi_one.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=mat1_projected,
            a_vec=vec1_projected,
            B_mat=mat1_projected,
            b_vec=vec1_projected,
        )
        G = G_plus + G_minus - G_one
        # R = .5 * (R + R.T)
        uGu = jnp.sum(u_i.T * jnp.einsum("abc, cb -> ab", G, u_i), axis=1)
        log_lb_sum = log_lb
        return uGu, log_lb_sum

    def get_Sigma_stats(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs
    ) -> Float[Array, "N"]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral evaluated for of :math:`Y=y`.
        """
        vec = y - self.b
        E_epsilon2 = p_x.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=-self.M,
            a_vec=vec,
            B_mat=-self.M,
            b_vec=vec,
        )

        def scan_body_function(carry, args_i):
            W_i, u_i = args_i
            omega_star_i, omega_dagger_i, _ = lax.stop_gradient(
                self._get_omega_star_i(W_i, u_i, p_x, y)
            )
            Guu_i = self._get_Guu_i(W_i, u_i, omega_star_i, omega_dagger_i, p_x, y)
            result = Guu_i
            return carry, result

        _, result = lax.scan(scan_body_function, None, (self.W, self.U.T))
        Guu = result
        E_D_inv_epsilon2 = jnp.sum(Guu, axis=0)
        Sigma_stats = E_epsilon2 - E_D_inv_epsilon2
        Sigma_stats = 0.5 * (
            Sigma_stats + jnp.swapaxes(Sigma_stats, axis1=-1, axis2=-2)
        )
        return Sigma_stats

    def _get_Guu_i(
        self,
        W_i: Float[Array, "Dx+1"],
        u_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        omega_dagger,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
        # phi = pdf.GaussianPDF(**phi_dict)
        # beta = self.beta[iu:iu + 1]
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        G = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (G, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        g_omega = self.g(omega_star)
        nu = -g_omega[:, None] * b_i * w_i
        nu_plus = w_i + nu
        nu_minus = -w_i + nu
        ln_beta = -jnp.log(1 + self.f(omega_star)) - 0.5 * g_omega * (
            b_i**2 - omega_star**2
        )
        ln_beta_plus = ln_beta + b_i - jnp.log(2)
        ln_beta_minus = ln_beta - b_i - jnp.log(2)
        # Create OneRankFactors
        exp_factor_plus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factor.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        one_factor = factor.OneRankFactor(v=v, g=g_omega, nu=nu, ln_beta=ln_beta)
        # Create the two measures
        exp_phi_plus = p_x.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = p_x.hadamard(exp_factor_minus, update_full=True)
        phi_one = p_x.hadamard(one_factor, update_full=True)

        mat1 = -self.M[0]
        vec1 = y - self.b[0]
        vec1_projected = jnp.einsum("ba,cb->ca", u_i, vec1)
        mat1_projected = jnp.einsum("ba,bc->ac", u_i, mat1)
        G_plus_u = exp_phi_plus.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=mat1,
            a_vec=vec1,
            B_mat=mat1_projected,
            b_vec=vec1_projected,
        )
        G_minus_u = exp_phi_minus.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=mat1,
            a_vec=vec1,
            B_mat=mat1_projected,
            b_vec=vec1_projected,
        )
        G_one_u = phi_one.integrate(
            "(Ax+a)(Bx+b)'",
            A_mat=mat1,
            a_vec=vec1,
            B_mat=mat1_projected,
            b_vec=vec1_projected,
        )
        G_u = G_plus_u + G_minus_u - G_one_u
        # R = .5 * (R + R.T)
        Guu = jnp.einsum("abc, dc -> abd", G_u, u_i)
        return Guu

    def _get_omega_star_i(
        self,
        W_i: Float[Array, "Dy+1"],
        u_i: Float[Array, "Dy"],
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N"],
        conv_crit: float = 1e-4,
    ) -> Tuple[Float[Array, "N"], Int[Array, "_"]]:
        R = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (R, 1))
        b_i = W_i[:1]
        u_i = u_i[:].reshape((-1, 1))
        u_i_projected = jnp.dot(self.L_inv[0].T, u_i)
        # uM = jnp.dot(u_i.T, -self.M[0])
        # uy_b = jnp.dot(u_i.T, (y - self.b[0]).T)
        uM_projected = jnp.dot(u_i_projected.T, -self.M[0])
        uM_projected_norm = jnp.linalg.norm(uM_projected, axis=1)
        uM_projected /= uM_projected_norm
        uy_b_projected = jnp.dot(u_i_projected.T, (y - self.b[0]).T)
        uy_b_projected /= uM_projected_norm
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        )
        omega_dagger = jnp.clip(omega_dagger, a_min=1e-6)
        omega_star = omega_dagger
        # omega_star = omega_star_init
        omega_old = 10 * jnp.ones(R)

        def body_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # From the lower bound term
            g_omega = self.g(omega_star)
            nu = -g_omega[:, None] * b_i * w_i
            nu_plus = w_i + nu
            nu_minus = -w_i + nu
            ln_beta = -jnp.log(1 + self.f(omega_star)) - 0.5 * g_omega * (
                b_i**2 - omega_star**2
            )
            ln_beta_plus = ln_beta + b_i - jnp.log(2)
            ln_beta_minus = ln_beta - b_i - jnp.log(2)
            # Create OneRankFactors
            exp_factor_plus = factor.OneRankFactor(
                v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
            )
            exp_factor_minus = factor.OneRankFactor(
                v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
            )
            one_factor = factor.OneRankFactor(v=v, g=g_omega, nu=nu, ln_beta=ln_beta)
            # Create the two measures
            exp_phi_plus = p_x.hadamard(exp_factor_plus, update_full=True)
            exp_phi_minus = p_x.hadamard(exp_factor_minus, update_full=True)
            phi_one = p_x.hadamard(one_factor, update_full=True)
            # Fourth order integrals \mathbb{E}[h^2 (x-Cz-d)^2]
            quart_int_plus = exp_phi_plus.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=uM_projected,
                a_vec=uy_b_projected.T,
                B_mat=uM_projected,
                b_vec=uy_b_projected.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int_minus = exp_phi_minus.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=uM_projected,
                a_vec=uy_b_projected.T,
                B_mat=uM_projected,
                b_vec=uy_b_projected.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int_one = phi_one.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=uM_projected,
                a_vec=uy_b_projected.T,
                B_mat=uM_projected,
                b_vec=uy_b_projected.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int = quart_int_plus + quart_int_minus - quart_int_one
            # Second order integrals \mathbb{E}[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate(
                "(Ax+a)'(Bx+b)",
                A_mat=uM_projected,
                a_vec=uy_b_projected.T,
                B_mat=uM_projected,
                b_vec=uy_b_projected.T,
            )
            quad_int_minus = exp_phi_minus.integrate(
                "(Ax+a)'(Bx+b)",
                A_mat=uM_projected,
                a_vec=uy_b_projected.T,
                B_mat=uM_projected,
                b_vec=uy_b_projected.T,
            )
            quad_int_one = phi_one.integrate(
                "(Ax+a)'(Bx+b)",
                A_mat=uM_projected,
                a_vec=uy_b_projected.T,
                B_mat=uM_projected,
                b_vec=uy_b_projected.T,
            )
            quad_int = quad_int_plus + quad_int_minus - quad_int_one
            # quad_int = jnp.clip(quad_int, a_min=1e-10)
            omega_old = omega_star
            omega_star = jnp.where(
                jnp.isclose(quad_int, 0),
                0,
                jnp.sqrt(jnp.abs(quart_int)) / jnp.sqrt(jnp.abs(quad_int)),
            )
            omega_star = jnp.sqrt(jnp.abs(quart_int)) / jnp.sqrt(jnp.abs(quad_int))
            omega_star = jnp.clip(omega_star, a_min=1e-6)
            num_iter = num_iter + 1
            return omega_star, omega_old, num_iter

        def cond_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # return lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old)), 'i') > conv_crit
            return jnp.logical_and(
                jnp.amax(
                    jnp.amax(
                        jnp.abs(omega_star - omega_old)
                        / jnp.clip(omega_star, a_min=1e-5)
                    )
                )
                > conv_crit,
                num_iter < 1000,
            )

        num_iter = 0
        init_val = (omega_star, omega_old, num_iter)
        omega_star, omega_old, num_iter = lax.while_loop(cond_fun, body_fun, init_val)
        indices_non_converged = (
            jnp.abs(omega_star - omega_old) / jnp.clip(omega_star, a_min=1e-5)
        ) > conv_crit

        return omega_star, omega_dagger, indices_non_converged

    def f(self, h: Float[Array, "N"]) -> Float[Array, "N"]:
        """Compute the function

        .. math::

        f(h) = \cosh(h) - 1

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated functions
        """
        return jnp.cosh(h) - 1

    def f_prime(self, h: Float[Array, "N"]) -> Float[Array, "N"]:
        """Computes the derivative of f

        .. math::

            f^\prime(h) = 2 * \beta * \sinh(h)

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated derivative functions.
        """
        return jnp.sinh(h)

    def g(self, omega: Float[Array, "N"]) -> Float[Array, "N"]:
        r"""Computes the function

        .. math::

            g(\omega) = f'(\omega) / (sigma_x^2 + f(\omega)) / |\omega|

            for the variational bound

        Args:
            omega: Free variational parameter.
            beta: Scaling factor.
            sigma_x: Noise parameter.

        Returns:
            Evaluated function.
        """
        return jnp.tanh(omega) / jnp.abs(omega)


@dataclass(kw_only=True)
class HCCovGaussianConditional(conditional.ConditionalGaussianPDF):
    r"""A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

    and :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    sigma_x: float
    U: Float[Array, "Dy Du"]
    W: Float[Array, "Du Dx+1"]
    beta: Float[Array, "Du"]
    exp_h_plus: factor.LinearFactor = field(init=False)
    exp_h_minus: factor.LinearFactor = field(init=False)
    sigma2_x: float = field(init=False)

    def __post_init__(
        self,
    ):
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        if self.Dy < self.Du:
            raise NotImplementedError(
                "There must be less vectors U than dimensionality of Y."
            )
        self.sigma2_x = self.sigma_x**2
        self._setup_noise_diagonal_functions()

    @property
    def R(self) -> int:
        """Number of conditionals (leading dimension)."""
        return self.M.shape[0]

    @property
    def Dy(self) -> int:
        r"""Dimensionality of :math:`Y`."""
        return self.M.shape[1]

    @property
    def Dx(self) -> int:
        r"""Dimensionality of :math:`X`."""
        return self.M.shape[2]

    @property
    def Du(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.beta.shape[0]

    def _setup_noise_diagonal_functions(self):
        r"""Create the functions, that later need to be integrated over, i.e.

        .. math::

            \exp(h_i(z)) \text{ and } \exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factor.LinearFactor(nu=nu, ln_beta=ln_beta)
        self.exp_h_minus = factor.LinearFactor(nu=-nu, ln_beta=-ln_beta)

    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

        with :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        D_x = self.beta[None] * (self.exp_h_plus(x) + self.exp_h_minus(x)).T
        Sigma_0 = self.sigma2_x * jnp.eye(self.Dy)
        Sigma_y_x = Sigma_0[None] + jnp.einsum(
            "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, D_x), self.U
        )
        if invert:
            G_x = D_x / (self.sigma2_x + D_x)
            Sigma_y_x_inv = jnp.eye(self.Dy)[None] - jnp.einsum(
                "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, G_x), self.U
            )
            Sigma_y_x_inv /= self.sigma2_x
            ln_det_Sigma_y_x = (self.Dy - self.Du) * jnp.log(self.sigma2_x) + jnp.sum(
                jnp.log(self.sigma2_x + D_x), axis=1
            )
            return Sigma_y_x, Sigma_y_x_inv, ln_det_Sigma_y_x
        else:
            return Sigma_y_x

    def condition_on_x(self, x: Float[Array, "N Dx"], **kwargs) -> pdf.GaussianPDF:
        r"""Get Gaussian Density conditioned on :math:`X=x`.

        Args:
            x: Instances, the mu and Sigma should be conditioned on.

        Returns:
            The density conditioned on :math:`X=x`.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new, Lambda_new, ln_det_Sigma_new = self.get_conditional_cov(
            x, invert=True
        )
        # Sigma_new = .5 * (Sigma_new + jnp.swapaxes(Sigma_new, -2, -1))
        return pdf.GaussianPDF(
            Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new
        )
        # Sigma_new = self.get_conditional_cov(x)
        # return pdf.GaussianPDF(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: Float[Array, "R Dy"], **kwargs):
        r"""Not valid function for this model class.

        Args:
            y: Data for :math:`Y`, where the rth entry is associated
                with the rth conditional density.

        Raises:
            AttributeError: Raised because doesn't :math:`p(Y|X)` is not
                a ConjugateFactor for :math:`X`.
        """
        raise AttributeError("HCCovGaussianConditional doesn't have function set_y.")

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus).integrate()
            + p_x.multiply(self.exp_h_minus).integrate()
        )
        D_int = self.beta[None] * D_int.reshape((p_x.R, self.Du))
        Sigma_int = self.sigma2_x * jnp.eye(self.Dy)[None] + jnp.einsum(
            "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        )
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF
    ) -> Tuple[Float[Array, "R Dy"], Float[Array, "1 Dy Dy"]]:
        r"""Compute the expected mean and covariance

        .. math::

            \mu_y = \mathbb{E}[y] = M \mathbb{E}[x] + b

        .. math::

            \Sigma_y = \mathbb{E}[yy'] - \mu_y \mu_y^\top = \sigma_x^2 I + \sum_i U_i \mathbb{E}[D_i(x)] U_i^\top + \mathbb{E}[\mu(x)\mu(x)^\top] - \mu_y \mu_y^\top

        Args:
            p_x: The density which we average over.

        Returns:
            Returns the expected mean and covariance.
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Get an approximation of the joint density
        
        .. math::

            p(x,y) ~= {\cal N}(\mu_{xy},\Sigma_{xy}),

        The mean is given by
        
        .. math::

            \mu_{xy} = (\mu_x, \mu_y)^\top

        with :math:`\mu_y = \mathbb{E}[\mu_y(x)]`. The covariance is given by

        .. math::
        
            \Sigma_{xy} = \begin{pmatrix}
                            \Sigma_x & \mathbb{E}[xy^\top] - \mu_x\mu_y^\top \\
                            \mathbb{E}[yx^\top] - \mu_y\mu_x^\top & \mathbb{E}[yy^\top] - \mu_y\mu_y^\top
                        \end{pmatrix}.

        Args:
            p_x: The density which we average over.

        Returns:
            Joint distribution of :math:`p(x,y)`.
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

        Args:
            p_x: Marginal Gaussian density over :math:`X`.

        Returns:
            Conditional density of :math:`p(X|Y)`.
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
            M=M_new,
            b=b_new,
            Sigma=Sigma_new,
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

        Args:
            p_x: Marginal Gaussian density over :math`X`.

        Returns:
            The marginal density :math:`p(Y)`.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = pdf.GaussianPDF(Sigma=Sigma_y, mu=mu_y)
        return p_y

    def get_correction_factors(self, x_samples, y, ln_det_term_lb, inv_term_lb):
        def get_sampled_lb_terms(x_samples, y):
            mu = self.get_conditional_mu(x_samples)[0]
            Sigma, Sigma_inv, log_det_Sigma = self.get_conditional_cov(
                x_samples, invert=True
            )
            correction_factor_log_det = log_det_Sigma + (self.Du - self.Dy) * jnp.log(
                self.sigma2_x
            )
            epsilon = y - mu
            Sigma_inv_correction = Sigma_inv - jnp.eye(self.Dy)[None] / self.sigma2_x
            correction_factor_inv = jnp.einsum(
                "ab,ab->a",
                epsilon,
                jnp.einsum("abc,ac->ab", Sigma_inv_correction, epsilon),
            )
            return correction_factor_log_det, correction_factor_inv

        correction_factor_log_det_samples, correction_factor_inv_samples = vmap(
            get_sampled_lb_terms, in_axes=(0, None)
        )(x_samples, y)
        ln_det_sampled, inv_term_sampled = jnp.mean(
            correction_factor_log_det_samples, axis=0
        ), jnp.mean(correction_factor_inv_samples, axis=0)
        inv_term_sampled = -inv_term_sampled * self.sigma_x**2
        inv_correction_factor = inv_term_sampled / inv_term_lb
        ln_det_correction_factor = ln_det_sampled / ln_det_term_lb
        return ln_det_correction_factor, inv_correction_factor

    def integrate_log_conditional_y(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        projection: Float[Array, "Dy Dy"] = None,
        x_samples: Float[Array, "N R Dy"] = None,
        **kwargs
    ) -> Float[Array, "N"]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral evaluated for of :math:`Y=y`.
        """
        if projection is None:
            projection = jnp.eye(self.Dy)
        vec = y - self.b
        vec_projected = jnp.einsum("ab,cb->ca", projection, vec)
        M_projected = jnp.einsum("ab,cbd->cad", projection, self.M)
        E_epsilon2 = p_x.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-M_projected,
            a_vec=vec_projected,
            B_mat=-M_projected,
            b_vec=vec_projected,
        )

        def scan_body_function(carry, args_i):
            W_i, u_i, beta_i = args_i
            omega_star_i, omega_dagger_i, _ = lax.stop_gradient(
                self._get_omega_star_i(W_i, u_i, beta_i, p_x, y)
            )
            uRu_i, log_lb_sum_i = self._get_lb_i(
                W_i, u_i, beta_i, omega_star_i, omega_dagger_i, p_x, y, projection
            )
            result = (uRu_i, log_lb_sum_i)
            return carry, result

        _, result = lax.scan(scan_body_function, None, (self.W, self.U.T, self.beta))
        uRu, log_lb_sum = result
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)

        if x_samples is not None:
            ln_det_correction_factor, inv_correction_factor = lax.stop_gradient(
                self.get_correction_factors(
                    x_samples, y, E_ln_sigma2_f, E_D_inv_epsilon2
                )
            )
        else:
            ln_det_correction_factor, inv_correction_factor = 1.0, 1.0

        log_int_y = (
            -0.5
            * (E_epsilon2 - inv_correction_factor * E_D_inv_epsilon2)
            / self.sigma_x**2
        )
        # determinant part
        log_int_y = (
            log_int_y
            - 0.5 * ln_det_correction_factor * E_ln_sigma2_f
            + 0.5 * (self.Du - self.Dy) * jnp.log(self.sigma_x**2)
            - 0.5 * self.Dy * jnp.log(2.0 * jnp.pi)
        )
        return log_int_y

    def _get_lb_i(
        self,
        W_i: Float[Array, "Dx+1"],
        u_i: Float[Array, "Dy"],
        beta_i: Float[Array, "1"],
        omega_star: Float[Array, "N"],
        omega_dagger,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        projection: Float[Array, "Dy Dy"],
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
        # phi = pdf.GaussianPDF(**phi_dict)
        # beta = self.beta[iu:iu + 1]
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        R = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (R, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        # uC = jnp.dot(u_i.T, -self.M[0])
        # uy_d = jnp.dot(u_i.T, (y - self.b[0]).T)
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]

        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        f_omega_dagger = self.f(omega_dagger, beta_i)
        g_omega_dagger = self.g(omega_dagger, beta_i)
        log_lb = jnp.log(self.sigma_x**2 + f_omega_dagger) + 0.5 * g_omega_dagger * (
            Eh2 - omega_dagger**2
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
        mat1 = jnp.einsum("cb,bd->cd", projection, -self.M[0])
        vec1 = jnp.einsum("ab,cb->ca", projection, y - self.b[0])
        R_plus = exp_phi_plus.integrate(
            "(Ax+a)(Bx+b)'", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R_minus = exp_phi_minus.integrate(
            "(Ax+a)(Bx+b)'", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R = R_plus + R_minus
        # R = .5 * (R + R.T)

        uRu = jnp.sum(u_i.T * jnp.einsum("abc, cb -> ab", R, u_i), axis=1)
        log_lb_sum = log_lb
        return uRu, log_lb_sum

    def _get_omega_star_i(
        self,
        W_i: Float[Array, "Dy+1"],
        u_i: Float[Array, "Dy"],
        beta_i: Float[Array, "1"],
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N"],
        conv_crit: float = 1e-4,
    ) -> Tuple[Float[Array, "N"], Int[Array, "_"]]:
        R = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (R, 1))
        b_i = W_i[:1]
        u_i = u_i[:].reshape((-1, 1))
        uM = jnp.dot(u_i.T, -self.M[0])
        norm_uM = jnp.linalg.norm(uM, axis=-1)
        uM = uM / norm_uM
        uy_b = jnp.dot(u_i.T, (y - self.b[0]).T)
        uy_b = uy_b / norm_uM
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        )
        omega_dagger = jnp.where(jnp.isclose(omega_dagger, 0), 1e-5, omega_dagger)
        omega_star = omega_dagger
        # omega_star = omega_star_init
        omega_old = 10 * jnp.ones(R)

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
            # Fourth order integrals \mathbb{E}[h^2 (x-Cz-d)^2]
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
            # Second order integrals \mathbb{E}[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate(
                "(Ax+a)'(Bx+b)", A_mat=uM, a_vec=uy_b.T, B_mat=uM, b_vec=uy_b.T
            )
            quad_int_minus = exp_phi_minus.integrate(
                "(Ax+a)'(Bx+b)", A_mat=uM, a_vec=uy_b.T, B_mat=uM, b_vec=uy_b.T
            )
            quad_int = quad_int_plus + quad_int_minus
            omega_star = jnp.sqrt(jnp.abs(quart_int)) / jnp.sqrt(jnp.abs(quad_int))
            num_iter = num_iter + 1
            return omega_star, omega_star, num_iter

        def cond_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # return lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old)), 'i') > conv_crit
            return jnp.logical_and(
                jnp.amax(jnp.amax(jnp.abs(omega_star - omega_old) / omega_star))
                > conv_crit,
                num_iter < 1000,
            )

        num_iter = 0
        init_val = (omega_star, omega_old, num_iter)
        omega_star, omega_old, num_iter = lax.while_loop(cond_fun, body_fun, init_val)
        indices_non_converged = (
            jnp.abs(omega_star - omega_old) / omega_star
        ) > conv_crit

        return omega_star, omega_dagger, indices_non_converged

    def f(self, h: Float[Array, "N"], beta: float) -> Float[Array, "N"]:
        """Compute the function

        .. math::

        f(h) = 2 * \beta * \cosh(h)

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated functions
        """
        return 2 * beta * jnp.cosh(h)

    def f_prime(self, h: Float[Array, "N"], beta: float) -> Float[Array, "N"]:
        """Computes the derivative of f

        .. math::

            f^\prime(h) = 2 * \beta * \sinh(h)

        Args:
            h: Activation functions.
            beta: Scaling factor.

        Returns:
            Evaluated derivative functions.
        """
        return 2 * beta * jnp.sinh(h)

    def g(self, omega: Float[Array, "N"], beta: float) -> Float[Array, "N"]:
        r"""Computes the function

        .. math::

            g(\omega) = f'(\omega) / (sigma_x^2 + f(\omega)) / |\omega|

            for the variational bound

        Args:
            omega: Free variational parameter.
            beta: Scaling factor.
            sigma_x: Noise parameter.

        Returns:
            Evaluated function.
        """
        return (
            self.f_prime(omega, beta)
            / (self.sigma_x**2 + self.f(omega, beta))
            / jnp.abs(omega)
        )


'''
@dataclass(kw_only=True)
class FullHCCovGaussianConditional(HCCovGaussianConditional):
    r"""A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

    .. math::

        p(y|x) = N(\mu(x), \Sigma(x))

    with the conditional mean function :math:`\mu(x) = M x + b`.
    The covariance matrix has the form

    .. math::

        \Sigma_y(x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

    and :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

    Note, that the affine transformations will be approximated via moment matching.

    Args:
        M: Matrix in the mean function.
        b: Vector in the conditional mean function.
        sigma_x: Diagonal noise parameter.
        U: Othonormal vectors for low rank noise part.
        W: Noise weights for low rank components (w_i & b_i).
        beta: Scaling for low rank noise components.

    Raises:
        NotImplementedError: Only works with R==1.
    """
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    Sigma: Float[Array, "1 Dy Dy"] = None
    Lambda: Float[Array, "R Dy Dy"] = None
    ln_det_Sigma: Float[Array, "R"] = None
    sigma_x: float = field(init=False)
    U: Float[Array, "Dy Du"]
    W: Float[Array, "Du Dx+1"]
    beta: Float[Array, "Du"]
    exp_h_plus: factor.LinearFactor = field(init=False)
    exp_h_minus: factor.LinearFactor = field(init=False)
    sigma2_x: float = field(init=False)

    def __post_init__(
        self,
    ):
        self.sigma_x = 1
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        if self.Dy < self.Du:
            raise NotImplementedError(
                "There must be less vectors U than dimensionality of Y."
            )
        if self.Sigma is None and self.Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif self.Sigma is not None:
            if self.Lambda is None or self.ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        else:
            self.Sigma, ln_det_Lambda = invert_matrix(self.Lambda)
            self.ln_det_Sigma = -ln_det_Lambda
        self.sigma2_x = self.sigma_x**2
        self._setup_noise_diagonal_functions()

    def get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

        with :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        if invert:
            Sigma_y_x, Sigma_y_x_inv, ln_det_Sigma_y_x = self._get_conditional_cov(
                x, invert
            )
            Sigma_y_x = jnp.einsum("abc,acd->abd", self.Sigma, Sigma_y_x)
            Sigma_y_x_inv = jnp.einsum("abc,acd->abd", Sigma_y_x_inv, self.Lambda)
            ln_det_Sigma_y_x = ln_det_Sigma_y_x + self.ln_det_Sigma
            return Sigma_y_x, Sigma_y_x_inv, ln_det_Sigma_y_x
        else:
            Sigma_y_x = self._get_conditional_cov(x, invert)
            return jnp.einsum("abc,acd->abd", self.Sigma, Sigma_y_x)

    def _get_conditional_cov(
        self, x: Float[Array, "N Dx"], invert: bool = False
    ) -> Union[
        Float[Array, "N Dy Dy"],
        Tuple[Float[Array, "N Dy Dy"], Float[Array, "N Dy Dy"], Float[Array, "N"]],
    ]:
        r"""Evaluate the covariance at a given :math:`X=x`, i.e.

        .. math::

            \Sigma_y(X=x) = \sigma_x^2 I + \sum_i U_i D_i(x) U_i^\top,

        with :math:`D_i(x) = 2 * \beta_i * \cosh(h_i(x))` and :math:`h_i(x) = w_i^\top x + b_i`.

        Args:
            x: Instances, the :math:`\mu` should be conditioned on.

        Returns:
            Conditional covariance.
        """
        D_x = self.beta[None] * (self.exp_h_plus(x) + self.exp_h_minus(x)).T
        Sigma_0 = self.sigma2_x * jnp.eye(self.Dy)
        Sigma_y_x = Sigma_0[None] + jnp.einsum(
            "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, D_x), self.U
        )
        if invert:
            G_x = D_x / (self.sigma2_x + D_x)
            Sigma_y_x_inv = jnp.eye(self.Dy)[None] - jnp.einsum(
                "adb,cb->acd", jnp.einsum("ab,cb->cab", self.U, G_x), self.U
            )
            Sigma_y_x_inv /= self.sigma2_x
            ln_det_Sigma_y_x = (self.Dy - self.Du) * jnp.log(self.sigma2_x) + jnp.sum(
                jnp.log(self.sigma2_x + D_x), axis=1
            )
            return Sigma_y_x, Sigma_y_x_inv, ln_det_Sigma_y_x
        else:
            return Sigma_y_x

    def integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        Sigma_int = self._integrate_Sigma_x(p_x)
        Sigma_int = jnp.einsum("abc,acd->abd", self.Sigma, Sigma_int)
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def _integrate_Sigma_x(self, p_x: pdf.GaussianPDF) -> Float[Array, "Dy Dy"]:
        r"""Integrate covariance with respect to :math:`p(X)`.

        .. math::

            \int \Sigma_Y(X)p(X) {\rm d}X.

        Args:
            p_x: The density the covatiance is integrated with.

        Returns:
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus).integrate()
            + p_x.multiply(self.exp_h_minus).integrate()
        )
        D_int = self.beta[None] * D_int.reshape((p_x.R, self.Du))
        Sigma_int = self.sigma2_x * jnp.eye(self.Dy)[None] + jnp.einsum(
            "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        )
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    def integrate_log_conditional_y(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs
    ) -> Float[Array, "N"]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral evaluated for of :math:`Y=y`.
        """
        L = jsc.linalg.cho_factor(self.Lambda)[0][0]
        return self._integrate_log_conditional_y(p_x, y, projection=L)

    def _integrate_log_conditional_y(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        projection: Float[Array, "Dy Dy"] = None,
        **kwargs
    ) -> Float[Array, "N"]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral evaluated for of :math:`Y=y`.
        """
        if projection is None:
            projection = jnp.eye(self.Dy)

        vec = y - self.b
        vec_projected = jnp.einsum("ab,cb->ca", projection, vec)
        M_projected = jnp.einsum("ab,cbd->cad", projection, self.M)
        E_epsilon2 = p_x.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-M_projected,
            a_vec=vec_projected,
            B_mat=-M_projected,
            b_vec=vec_projected,
        )

        def scan_body_function(carry, args_i):
            W_i, u_i, beta_i = args_i
            omega_star_i, omega_dagger_i, _ = lax.stop_gradient(
                self._get_omega_star_i(W_i, u_i, beta_i, p_x, y)
            )
            uRu_i, log_lb_sum_i = self._get_lb_i(
                W_i, u_i, beta_i, omega_star_i, omega_dagger_i, p_x, y, projection
            )
            result = (uRu_i, log_lb_sum_i)
            return carry, result

        _, result = lax.scan(scan_body_function, None, (self.W, self.U.T, self.beta))
        uRu, log_lb_sum = result
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0) + self.ln_det_Sigma
        log_int_y = -0.5 * (E_epsilon2 - E_D_inv_epsilon2) / self.sigma_x**2
        # determinant part
        log_int_y = (
            log_int_y
            - 0.5 * E_ln_sigma2_f
            + 0.5 * (self.Du - self.Dy) * jnp.log(self.sigma_x**2)
            - 0.5 * self.Dy * jnp.log(2.0 * jnp.pi)
        )
        return log_int_y

    def get_Sigma_stats(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs
    ) -> Float[Array, "N"]:
        r"""Compute the expectation over the log conditional, but just over :math:`X`. I.e. it returns

        .. math::

            f(Y) = \int \log(p(Y|X))p(X){\rm d}X.

        Args:
            p_x: Density over :math:`X`.

        Raises:
            NotImplementedError: Only implemented for R=1.

        Returns:
            The integral evaluated for of :math:`Y=y`.
        """
        vec = y - self.b

        E_epsilon2 = jnp.sum(
            p_x.integrate(
                "(Ax+a)(Bx+b)'",
                A_mat=-self.M,
                a_vec=vec,
                B_mat=-self.M,
                b_vec=vec,
            ),
            axis=0,
        )

        def scan_body_function(carry, args_i):
            W_i, u_i, beta_i = args_i
            omega_star_i, _, _ = lax.stop_gradient(
                self._get_omega_star_i(W_i, u_i, beta_i, p_x, y)
            )
            uRu_i = self._get_lb_mat_i(W_i, u_i, beta_i, omega_star_i, p_x, y)
            result = uRu_i
            return carry, result

        _, result = lax.scan(scan_body_function, None, (self.W, self.U.T, self.beta))
        uRu = result
        uRu = jnp.sum(uRu, axis=0)
        Sigma_stats = jnp.diag((E_epsilon2 - uRu).diagonal())
        # Sigma_stats = E_epsilon2 - uRu
        return Sigma_stats

    def _get_lb_mat_i(
        self,
        W_i: Float[Array, "Dx+1"],
        u_i: Float[Array, "Dy"],
        beta_i: Float[Array, "1"],
        omega_star: Float[Array, "N"],
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
    ) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
        # phi = pdf.GaussianPDF(**phi_dict)
        # beta = self.beta[iu:iu + 1]
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
        R = p_x.R
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (R, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        # uC = jnp.dot(u_i.T, -self.M[0])
        # uy_d = jnp.dot(u_i.T, (y - self.b[0]).T)
        # Lower bound for \mathbb{E}[ln (sigma_x^2 + f(h))]
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
        # R = .5 * (R + R.T)

        uRu = jnp.einsum("abc,bc->bc", R, jnp.einsum("ab,cb->ac", u_i, u_i))
        return uRu
'''
