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
        lower_bound_log_det = self.ln_det_Sigma + jnp.sum(k_omega, axis=0)
        return lower_bound_log_det
    
    def get_lb_quadratic_term(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]) -> Float[Array, "N"]:
        projected_M = jnp.einsum('acb,acd->abd', self.Lambda, self.M)
        projected_yb = jnp.einsum('acb,ac->ab', self.Lambda, y - self.b)
        homoscedastic_term = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=-projected_M, a_vec=projected_yb, B_mat=-self.M, b_vec=y - self.b)
        A_inv = jnp.einsum('abc,acd->abd', self.Lambda, self.A[:,:,:self.Dk])[0]
        get_lb_heteroscedastic_term = jnp.sum(vmap(lambda W, A_inv: self.get_lb_heteroscedastic_term_i(p_x, y, W, A_inv))(self.W, A_inv.T), axis=0)
        return homoscedastic_term - get_lb_heteroscedastic_term
    
    def get_lb_heteroscedastic_term_i(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], 
                                      W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]) -> Float[Array, "N"]:
        omega_star = lax.stop_gradient(self._get_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i))
        # Quadratic integral
        return self._lower_bound_integrals(p_x, y, W_i, a_i, omega_star)
    
    @abstractmethod
    def _lower_bound_integrals(self, p_x: measure.GaussianMeasure, y: Float[Array, "N Dy"], 
                              W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"], compute_fourth_order: bool=False) -> Float[Array, "N"]:
        pass                           
    
    def _get_omega_star(self, p_x: pdf.GaussianPDF, y: jnp.ndarray, W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        omega_star = self._get_omega_dagger(p_x=p_x, W_i=W_i)
        omega_dagger = omega_star
        iteration = 0
        cond_func = lambda val: jnp.logical_and(jnp.max(jnp.abs(val[0] - val[1])) > 1e-5, val[2] < 100)
        
        def body_func(val):
            return self._update_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=val[0]), val[0], val[2] + 1
        omega_star, _, _ = lax.while_loop(cond_func, body_func, (omega_star, omega_dagger, iteration))
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
class HeteroscedasticHeavisideConditional(HeteroscedasticConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    
    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.where(jnp.greater_equal(h, 0.), 1., 0.)
    
        
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
        w = self.W[:, 1:]
        w0 = self.W[:, 0]
        
        def integrate_f_i(w_i, w0_i):
            p_h = p_x.get_density_of_linear_sum(w_i[None, None], w0_i[None, None])
            tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0., upper_limit=jnp.inf)
            D_i_int = tp_h.integral()[0]
            return D_i_int
        #D_int = []
        #for i in range(self.Dk):
        #    D_int.append(integrate_f_i(w[i], w0[i]))
        #D_int = jnp.stack(D_int, axis=0)
        D_int = vmap(integrate_f_i, in_axes=(0,0))(w, w0)
        return D_int
    
    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        # int f(h(z)) dphi(z)
        w = self.W[:, 1:]
        w0 = self.W[:, 0]
        
        def integrate_f_i(w_i, w0_i):
            p_h = p_x.get_density_of_linear_sum(w_i[None,None], w0_i[None, None])
            tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0., upper_limit=jnp.inf)
            D_i_int = tp_h.integral()
            return D_i_int

        # int ln(1 + f(h)) dp(h)
        int_ln1pf_h = jnp.log(2.) * vmap(integrate_f_i, out_axes=0)(w, w0)
        log_det = self.ln_det_Sigma + jnp.sum(int_ln1pf_h, axis=0)
        return log_det
    
    def get_lb_heteroscedastic_term_i(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], 
                                      W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"]):
        w0 = W_i[None,0]
        w = W_i[None,1:]
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        if self.Dx == 1:
            factor = a_projected_M[:,0,0] / w[:,0]
            constant = - (a_projected_yb[:,0] +  factor * w0)
            p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
            tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0., upper_limit=jnp.inf)
            
        else:
            sum_weights = jnp.tile(jnp.concatenate([-a_projected_M[:,0], w])[None], (a_projected_yb.shape[0],1,1))
            sum_bias = jnp.hstack([a_projected_yb, jnp.tile(w0[None], (a_projected_yb.shape[0],1))])
            p_hg = p_x.get_density_of_linear_sum(sum_weights, sum_bias)
            p_h = p_hg.get_marginal(jnp.array([1]))
            tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0.)
            p_g_given_h = p_hg.condition_on_explicit(jnp.array([1]), jnp.array([0]))
            factor = p_g_given_h.M[:,0,0]
            constant = p_g_given_h.b[:,0]
            
        Zh = tp_h.integrate()
        Eh = tp_h.integrate("x")[:,0]
        Eh2 = tp_h.integrate("x**2")[:,0]
        #heteroscedastic_term_i = Zh * constant**2 + Eh2 * factor**2 + 2 * Eh * factor * constant
        heteroscedastic_term_i = Zh * constant**2 + Eh2 * factor**2 + 2 * Eh * factor * constant
        if self.Dx > 1:
            heteroscedastic_term_i += Zh * p_g_given_h.Sigma[:,0,0]                  
        heteroscedastic_term_i *= 0.5
        return heteroscedastic_term_i[None]

    def k_func(self, p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        pass

    def _lower_bound_integrals(self, p_x: measure.GaussianMeasure, y: Float[Array, "N Dy"], 
                            W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"], compute_fourth_order: bool=False) -> Float[Array, "N"]:
        pass      
    
    
@dataclass(kw_only=True)
class HeteroscedasticReLUConditional(HeteroscedasticConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"] = field(default=None)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    
    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.maximum(h, 0.)
    
        
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
        w = self.W[:, 1:]
        w0 = self.W[:, 0]
        
        def integrate_f_i(w_i, w0_i):
            p_h = p_x.get_density_of_linear_sum(w_i[None, None], w0_i[None, None])
            tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0., upper_limit=jnp.inf)
            D_i_int = tp_h.integrate('x')[0,0]
            return D_i_int
        #D_int = []
        #for i in range(self.Dk):
        #    D_int.append(integrate_f_i(w[i], w0[i]))
        #D_int = jnp.stack(D_int, axis=0)
        D_int = vmap(integrate_f_i, in_axes=(0,0))(w, w0)
        return D_int

    @staticmethod
    def _get_omega_dagger(p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]) -> Float[Array, "R"]:
        w0 = W_i[None,0]
        w = W_i[None,1:]
        p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
        tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0., upper_limit=jnp.inf)
        omega_dagger = tp_h.integrate('x')[:,0]
        return omega_dagger
    
    def _update_omega_star(self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"]) -> Float[Array, "N"]:      
        cubic_integral, quartic_integral = self._lower_bound_integrals(p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=omega_star, compute_fourth_order=True)
        cubic_integral= jnp.where(cubic_integral != 0., cubic_integral, 1.)
        omega_star = (quartic_integral / cubic_integral)[0]
    
        return omega_star

    def k_func(self, p_x: pdf.GaussianPDF, W_i:Float[Array, "Dx+1"], omega_dagger: Float[Array, "R"]) -> Float[Array, "R"]:
        w0 = W_i[None,0]
        w = W_i[None,1:]
        c0 = jnp.log(1. + omega_dagger)
        c1 = 1 / (1. + omega_dagger)
        p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
        tp_h = truncated_measure.TruncatedGaussianMeasure(measure=p_h, lower_limit=0., upper_limit=jnp.inf)
        Zh = tp_h.integrate()
        Eh = tp_h.integrate('x')[:,0]
        return Zh * c0 + c1 * (Eh - Zh * omega_dagger)

    def _lower_bound_integrals(self, p_x: measure.GaussianMeasure, y: Float[Array, "N Dy"], 
                            W_i: Float[Array, "Dx+1"], a_i: Float[Array, "Dy"], omega_star: Float[Array, "N"], compute_fourth_order: bool=False) -> Float[Array, "N"]:
        w0 = W_i[None,0]
        w = W_i[None,1:]
        a_projected_M = jnp.einsum('ab,cad->cbd', a_i[:,None], self.M)
        a_projected_yb = jnp.einsum('ab,ca->cb', a_i[:,None], y - self.b)
        
        nu_phi = - 1. / (1. + omega_star)
        ln_beta_phi = - jnp.log(1. + omega_star) + omega_star / (1. + omega_star)
        phi_h_factor = factor.LinearFactor(nu=nu_phi[:,None], ln_beta=ln_beta_phi)
        if self.Dx == 1:
            c1 = a_projected_M[:,0,0] / w[:,0]
            c0 = - (a_projected_yb[:,0] +  c1 * w0)
            p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
        else:
            sum_weights = jnp.tile(jnp.concatenate([-a_projected_M[:,0], w])[None], (a_projected_yb.shape[0],1,1))
            sum_bias = jnp.hstack([a_projected_yb, jnp.tile(w0[None], (a_projected_yb.shape[0],1))])
            p_hg = p_x.get_density_of_linear_sum(sum_weights, sum_bias)
            p_h = p_hg.get_marginal(jnp.array([1]))
            p_g_given_h = p_hg.condition_on_explicit(jnp.array([1]), jnp.array([0]))
            c1 = p_g_given_h.M[:,0,0]
            c0 = p_g_given_h.b[:,0]
            
        phi_h = p_h.hadamard(phi_h_factor, update_full=True)
        tp_h = truncated_measure.TruncatedGaussianMeasure(measure=phi_h, lower_limit=0.) 
        Eh = tp_h.integrate("x")[:,0]
        Eh2 = tp_h.integrate("x**2")[:,0]
        Eh3 = tp_h.integrate("x**k", k=3)[:,0]
        cubic_integral = Eh * c0**2 + Eh3 * c1**2 + 2 * Eh2 * c1 * c0
        if self.Dx > 1:
            cubic_integral += Eh * p_g_given_h.Sigma[:,0,0]                   
        
        if compute_fourth_order:
            Eh4 = tp_h.integrate("x**k", k=4)[:,0]
            quartic_integral = Eh2 * c0**2 + Eh4 * c1**2 + 2 * Eh3 * c1 * c0
            if self.Dx > 1:
                quartic_integral += Eh2 * p_g_given_h.Sigma[:,0,0]   
            return cubic_integral[None], quartic_integral[None]
        else:
            return cubic_integral[None]  
      