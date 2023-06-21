__author__ = "Christian Donner"

from jax import numpy as jnp
from typing import Any, Tuple, Union
from . import pdf, factor, measure, conditional
from .utils.linalg import invert_matrix, invert_woodbury_diag, invert_woodbury

from .utils.dataclass import dataclass
from jaxtyping import Array, Float, Int, Bool
from jax import lax
from jax import jit
from jax import scipy as jsc

from dataclasses import field
from jax import random, vmap
from abc import abstractmethod
from gaussian_toolbox.experimental import truncated_measure


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
