__author__ = "Christian Donner"

from jax import numpy as jnp
from typing import Any, Tuple, Union
from . import pdf, factor, measure, conditional
from .utils import linalg

from .utils.dataclass import dataclass
from jaxtyping import Array, Float
from jax import lax

from dataclasses import field
from jax import vmap
from abc import abstractmethod
from gaussian_toolbox.experimental import truncated_measure


@dataclass(kw_only=True)
class HeteroscedasticBaseConditional(conditional.ConditionalGaussianPDF):
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
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

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
        return self.A.shape[1]

    @property
    def Dk(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.W.shape[0]

    def linear_layer(self, x: Float[Array, "N Dx"]) -> Float[Array, "N Dk"]:
        """Linear layer of the argument of heteroscedastic link function.

        :return: :math:`w^\top x + w_0`.
        """
        return jnp.einsum("ab,cb->ca", self.W[:, 1:], x) + self.W[:, 0][None]

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
            "abc,dc->abd",
            jnp.einsum("ab,cb->cab", self.A[0, :, -self.Dk :], D_x),
            self.A[0, :, -self.Dk :],
        )
        if invert:
            G_x = D_x / (1 + D_x)  # [N x Dk]
            A_inv = jnp.einsum(
                "ab,bc->ac", self.Lambda[0], self.A[0, :, -self.Dk :]
            )  # [Dy x Dk] # [N x Dy x Dk]
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
        # Sigma_int = (
        #     self.Sigma
        #     + jnp.einsum(
        #         "ab,cb->ac",
        #         jnp.einsum("ab,b->ab", self.A[0, :, : self.Dk], D_int),
        #         self.A[0, :, : self.Dk],
        #     )[None]
        # )
        Sigma_int = self.Sigma + jnp.einsum(
            "b, adb, aeb -> ade",
            D_int,
            self.A[:, :, -self.Dk :],
            self.A[:, :, -self.Dk :],
        )
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
        # Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
        #    "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        # )
        # Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        Sigma_y = self.integrate_Sigma_x(p_x) + jnp.einsum(
            "abc, adb, aec -> ade", p_x.Sigma, self.M, self.M
        )
        Sigma_y = 0.5 * (Sigma_y + jnp.swapaxes(Sigma_y, axis1=-1, axis2=-2))
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: pdf.GaussianPDF) -> Float[Array, "R Dx Dy"]:
        r"""Compute :math:`\mathbb{E}[yx^\top] = \int\int yx^\top p(y|x)p(x) {\rm d}y{\rm d}x = \int (M x + b)x^\top p(x) {\rm d}x`.

        Args:
            p_x: The density which we average over.

        Returns:
            Cross expectations.
        """
        cross_cov = jnp.einsum("abc, adb -> adc", p_x.Sigma, self.M)
        # Eyx = p_x.integrate(
        #    "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        # )
        # return Eyx
        return cross_cov

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
        cov_yx = self.get_expected_cross_terms(p_x)
        # Eyx = self.get_expected_cross_terms(p_x)
        # mu_x = p_x.mu
        # cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        mu_xy = jnp.concatenate([p_x.mu, mu_y], axis=1)
        # Sigma_xy1 = jnp.concatenate(
        #    [p_x.Sigma, jnp.swapaxes(cov_yx, axis1=1, axis2=2)], axis=2
        # )
        # Sigma_xy2 = jnp.concatenate([cov_yx, Sigma_y], axis=2)
        # Sigma_xy = jnp.concatenate([Sigma_xy1, Sigma_xy2], axis=1)
        print(p_x.Sigma.shape, cov_yx.shape, Sigma_y.shape)
        Sigma_xy = jnp.block(
            [[p_x.Sigma, cov_yx.transpose((0, 2, 1))], [cov_yx, Sigma_y]]
        )
        print(Sigma_xy.shape, mu_xy.shape)
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
        Lambda_y = linalg.invert_matrix(Sigma_y)[0]
        cov_yx = self.get_expected_cross_terms(p_x)
        # Eyx = self.get_expected_cross_terms(p_x)
        # mu_x = p_x.mu
        # cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = p_x.mu - jnp.einsum("abc,ac->ab", M_new, mu_y)
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

    def integrate_log_conditional_y(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs
    ) -> Float[Array, "N"]:
        lb_quadratic_term = self.get_lb_quadratic_term(p_x, y)
        lb_log_det = self.get_lb_log_det(p_x)
        lb_log_p_y = (
            -0.5 * (lb_quadratic_term + lb_log_det + self.Dy * jnp.log(2.0 * jnp.pi))[0]
        )
        return lb_log_p_y

    @staticmethod
    def _get_omega_dagger(
        p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]
    ) -> Float[Array, "R"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        )
        return omega_dagger

    @abstractmethod
    def k_func(
        self,
        p_x: pdf.GaussianPDF,
        W_i: Float[Array, "Dx+1"],
        omega_dagger: Float[Array, "R"],
    ) -> Float[Array, "R"]:
        pass

    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        omega_dagger = lax.stop_gradient(
            vmap(lambda W: self._get_omega_dagger(p_x=p_x, W_i=W), in_axes=(0,))(self.W)
        )
        k_omega = vmap(
            lambda W, omega: self.k_func(p_x=p_x, W_i=W, omega_dagger=omega)
        )(self.W, omega_dagger)
        lower_bound_log_det = self.ln_det_Sigma + jnp.sum(k_omega, axis=0)
        return lower_bound_log_det

    def get_lb_quadratic_term(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]
    ) -> Float[Array, "N"]:
        projected_M = jnp.einsum("acb,acd->abd", self.Lambda, self.M)
        projected_yb = jnp.einsum("acb,ac->ab", self.Lambda, y - self.b)
        homoscedastic_term = p_x.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-projected_M,
            a_vec=projected_yb,
            B_mat=-self.M,
            b_vec=y - self.b,
        )
        A_inv = jnp.einsum("abc,acd->abd", self.Lambda, self.A[:, :, -self.Dk :])[0]
        get_lb_heteroscedastic_term = jnp.sum(
            vmap(lambda W, A_inv: self.get_lb_heteroscedastic_term_i(p_x, y, W, A_inv))(
                self.W, A_inv.T
            ),
            axis=0,
        )
        return homoscedastic_term - get_lb_heteroscedastic_term

    def get_lb_heteroscedastic_term_i(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
    ) -> Float[Array, "N"]:
        omega_star = lax.stop_gradient(
            self._get_omega_star(p_x=p_x, y=y, W_i=W_i, a_i=a_i)
        )
        # Quadratic integral
        return self._lower_bound_integrals(p_x, y, W_i, a_i, omega_star)

    @abstractmethod
    def _lower_bound_integrals(
        self,
        p_x: measure.GaussianMeasure,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        compute_fourth_order: bool = False,
    ) -> Float[Array, "N"]:
        pass

    def _get_omega_star(
        self,
        p_x: pdf.GaussianPDF,
        y: jnp.ndarray,
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
    ):
        omega_star = self._get_omega_dagger(p_x=p_x, W_i=W_i)
        omega_dagger = omega_star
        iteration = 0
        cond_func = lambda val: jnp.logical_and(
            jnp.max(jnp.abs(val[0] - val[1])) > 1e-5, val[2] < 100
        )

        def body_func(val):
            return (
                self._update_omega_star(
                    p_x=p_x, y=y, W_i=W_i, a_i=a_i, omega_star=val[0]
                ),
                val[0],
                val[2] + 1,
            )

        omega_star, _, _ = lax.while_loop(
            cond_func, body_func, (omega_star, omega_dagger, iteration)
        )
        return omega_star

    def _update_omega_star(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        quadratic_integral, quartic_integral = self._lower_bound_integrals(
            p_x=p_x,
            y=y,
            W_i=W_i,
            a_i=a_i,
            omega_star=omega_star,
            compute_fourth_order=True,
        )
        omega_star = jnp.sqrt(quartic_integral / quadratic_integral)[0]
        return omega_star


@dataclass(kw_only=True)
class HeteroscedasticExpConditional(HeteroscedasticBaseConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

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
    def _get_omega_dagger(
        p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]
    ) -> Float[Array, "R"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        )
        return omega_dagger

    def k_func(
        self,
        p_x: pdf.GaussianPDF,
        W_i: Float[Array, "Dx+1"],
        omega_dagger: Float[Array, "R"],
    ) -> Float[Array, "R"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        Eh = p_x.integrate("(Ax+a)", A_mat=w, a_vec=b)[:, 0]
        fomega = jnp.log(jnp.cosh(omega_dagger / 2.0)) + jnp.log(2.0)
        fprime_omega = 0.5 * jnp.tanh(omega_dagger / 2.0)
        return (
            0.5 * Eh
            + fomega
            + 0.5 * fprime_omega / omega_dagger * (Eh2 - omega_dagger**2)
        )

    def _lower_bound_integrals(
        self,
        p_x: measure.GaussianMeasure,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        compute_fourth_order: bool = False,
    ) -> Float[Array, "N"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        fomega = jnp.log(jnp.cosh(0.5 * omega_star)) + jnp.log(2.0)
        fprime_omega = 0.5 * jnp.tanh(0.5 * omega_star)
        g_1 = fprime_omega / omega_star
        nu_1 = -((fprime_omega / omega_star)[:, None] * b - 0.5) * w
        ln_beta_1 = (
            -fomega
            - 0.5 * fprime_omega / omega_star * (b**2 - omega_star**2)
            + 0.5 * b
        )
        lb_px_measure = p_x.hadamard(
            factor.OneRankFactor(v=w, g=g_1, nu=nu_1, ln_beta=ln_beta_1),
            update_full=True,
        )
        a_projected_M = jnp.einsum("ab,cad->cbd", a_i[:, None], self.M)
        a_projected_yb = jnp.einsum("ab,ca->cb", a_i[:, None], y - self.b)
        quadratic_integral = lb_px_measure.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        if compute_fourth_order:
            quartic_integral = lb_px_measure.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=w,
                a_vec=b,
                B_mat=w,
                b_vec=b,
                C_mat=-a_projected_M,
                c_vec=a_projected_yb,
                D_mat=-a_projected_M,
                d_vec=a_projected_yb,
            )
            return quadratic_integral, quartic_integral
        else:
            return quadratic_integral


@dataclass(kw_only=True)
class HeteroscedasticCoshM1Conditional(HeteroscedasticBaseConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.cosh(h) - 1.0

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
    def _get_omega_dagger(
        p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]
    ) -> Float[Array, "R"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        omega_dagger = jnp.sqrt(
            p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        )
        return omega_dagger

    def k_func(
        self,
        p_x: pdf.GaussianPDF,
        W_i: Float[Array, "Dx+1"],
        omega_dagger: Float[Array, "R"],
    ) -> Float[Array, "R"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        Eh2 = p_x.integrate("(Ax+a)'(Bx+b)", A_mat=w, a_vec=b, B_mat=w, b_vec=b)
        f_omega = jnp.log(jnp.cosh(omega_dagger))
        fprime_omega = jnp.tanh(omega_dagger)
        return f_omega + 0.5 * fprime_omega / omega_dagger * (Eh2 - omega_dagger**2)

    def _lower_bound_integrals(
        self,
        p_x: measure.GaussianMeasure,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        compute_fourth_order: bool = False,
    ) -> Float[Array, "N"]:
        b = W_i[None, :1]
        w = W_i[None, 1:]
        f_omega = jnp.log(jnp.cosh(omega_star))
        fprime_omega = 0.5 * jnp.tanh(omega_star) / omega_star
        g_1 = 2.0 * fprime_omega
        nu_1 = -2.0 * fprime_omega[:, None] * b * w
        ln_beta_1 = -f_omega - fprime_omega * (b**2 - omega_star**2)
        lb_px_measure = p_x.hadamard(
            factor.OneRankFactor(
                v=jnp.tile(w, (omega_star.shape[0], 1)),
                g=g_1,
                nu=nu_1,
                ln_beta=ln_beta_1,
            ),
            update_full=True,
        )
        a_projected_M = jnp.einsum("ab,cad->cbd", a_i[:, None], self.M)
        a_projected_yb = jnp.einsum("ab,ca->cb", a_i[:, None], y - self.b)
        exp_h_plus = factor.LinearFactor(nu=w, ln_beta=b - jnp.log(2))
        phi_plus = lb_px_measure.hadamard(exp_h_plus, update_full=True)
        quadratic_plus = phi_plus.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        exp_h_minus = factor.LinearFactor(nu=-w, ln_beta=-b - jnp.log(2))
        phi_minus = lb_px_measure.hadamard(exp_h_minus, update_full=True)
        quadratic_minus = phi_minus.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        quadratic_1 = lb_px_measure.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        quadratic_integral = quadratic_plus + quadratic_minus - quadratic_1
        if compute_fourth_order:
            quartic_integral = phi_plus.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=w,
                a_vec=b,
                B_mat=w,
                b_vec=b,
                C_mat=-a_projected_M,
                c_vec=a_projected_yb,
                D_mat=-a_projected_M,
                d_vec=a_projected_yb,
            )
            quartic_integral += phi_minus.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=w,
                a_vec=b,
                B_mat=w,
                b_vec=b,
                C_mat=-a_projected_M,
                c_vec=a_projected_yb,
                D_mat=-a_projected_M,
                d_vec=a_projected_yb,
            )
            quartic_integral -= lb_px_measure.integrate(
                "(Ax+a)'(Bx+b)(Cx+c)'(Dx+d)",
                A_mat=w,
                a_vec=b,
                B_mat=w,
                b_vec=b,
                C_mat=-a_projected_M,
                c_vec=a_projected_yb,
                D_mat=-a_projected_M,
                d_vec=a_projected_yb,
            )
            return quadratic_integral, quartic_integral
        else:
            return quadratic_integral


@dataclass(kw_only=True)
class HeteroscedasticHeavisideConditional(HeteroscedasticBaseConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.where(jnp.greater_equal(h, 0.0), 1.0, 0.0)

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
            tp_h = truncated_measure.TruncatedGaussianMeasure(
                measure=p_h, lower_limit=0.0, upper_limit=jnp.inf
            )
            D_i_int = tp_h.integral()[0]
            return D_i_int

        # D_int = []
        # for i in range(self.Dk):
        #    D_int.append(integrate_f_i(w[i], w0[i]))
        # D_int = jnp.stack(D_int, axis=0)
        D_int = vmap(integrate_f_i, in_axes=(0, 0))(w, w0)
        return D_int

    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        # int f(h(z)) dphi(z)
        w = self.W[:, 1:]
        w0 = self.W[:, 0]

        def integrate_f_i(w_i, w0_i):
            p_h = p_x.get_density_of_linear_sum(w_i[None, None], w0_i[None, None])
            tp_h = truncated_measure.TruncatedGaussianMeasure(
                measure=p_h, lower_limit=0.0, upper_limit=jnp.inf
            )
            D_i_int = tp_h.integral()
            return D_i_int

        # int ln(1 + f(h)) dp(h)
        int_ln1pf_h = jnp.log(2.0) * vmap(integrate_f_i, out_axes=0)(w, w0)
        log_det = self.ln_det_Sigma + jnp.sum(int_ln1pf_h, axis=0)
        return log_det

    def get_lb_heteroscedastic_term_i(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
    ):
        w0 = W_i[None, 0]
        w = jnp.sign(W_i[None, 1:]) * jnp.maximum(jnp.abs(W_i[None, 1:]), 1e-5)
        a_projected_M = jnp.einsum("ab,cad->cbd", a_i[:, None], self.M)
        a_projected_yb = jnp.einsum("ab,ca->cb", a_i[:, None], y - self.b)
        if self.Dx == 1:
            factor = a_projected_M[:, 0, 0] / w[:, 0]
            constant = -(a_projected_yb[:, 0] + factor * w0)
            p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
            tp_h = truncated_measure.TruncatedGaussianMeasure(
                measure=p_h, lower_limit=0.0, upper_limit=jnp.inf
            )

        else:
            sum_weights = jnp.tile(
                jnp.concatenate([-a_projected_M[:, 0], w])[None],
                (a_projected_yb.shape[0], 1, 1),
            )
            sum_bias = jnp.hstack(
                [a_projected_yb, jnp.tile(w0[None], (a_projected_yb.shape[0], 1))]
            )
            p_hg = p_x.get_density_of_linear_sum(sum_weights, sum_bias)
            p_h = p_hg.get_marginal(jnp.array([1]))
            tp_h = truncated_measure.TruncatedGaussianMeasure(
                measure=p_h, lower_limit=0.0
            )
            p_g_given_h = p_hg.condition_on_explicit(jnp.array([1]), jnp.array([0]))
            factor = p_g_given_h.M[:, 0, 0]
            constant = p_g_given_h.b[:, 0]

        Zh = tp_h.integrate()
        Eh = tp_h.integrate("x")[:, 0]
        Eh2 = tp_h.integrate("x**2")[:, 0]
        # heteroscedastic_term_i = Zh * constant**2 + Eh2 * factor**2 + 2 * Eh * factor * constant
        heteroscedastic_term_i = (
            Zh * constant**2 + Eh2 * factor**2 + 2 * Eh * factor * constant
        )
        if self.Dx > 1:
            heteroscedastic_term_i += Zh * p_g_given_h.Sigma[:, 0, 0]
        heteroscedastic_term_i *= 0.5
        return heteroscedastic_term_i[None]

    def k_func(
        self,
        p_x: pdf.GaussianPDF,
        W_i: Float[Array, "Dx+1"],
        omega_dagger: Float[Array, "R"],
    ) -> Float[Array, "R"]:
        pass

    def _lower_bound_integrals(
        self,
        p_x: measure.GaussianMeasure,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        compute_fourth_order: bool = False,
    ) -> Float[Array, "N"]:
        pass


@dataclass(kw_only=True)
class HeteroscedasticReLUConditional(HeteroscedasticBaseConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Da"]
    W: Float[Array, "Dk Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.maximum(h, 0.0)

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
            tp_h = truncated_measure.TruncatedGaussianMeasure(
                measure=p_h, lower_limit=0.0, upper_limit=jnp.inf
            )
            D_i_int = tp_h.integrate("x")[0, 0]
            return D_i_int

        # D_int = []
        # for i in range(self.Dk):
        #    D_int.append(integrate_f_i(w[i], w0[i]))
        # D_int = jnp.stack(D_int, axis=0)
        D_int = vmap(integrate_f_i, in_axes=(0, 0))(w, w0)
        return D_int

    @staticmethod
    def _get_omega_dagger(
        p_x: pdf.GaussianPDF, W_i: Float[Array, "Dx+1"]
    ) -> Float[Array, "R"]:
        w0 = W_i[None, 0]
        w = W_i[None, 1:]
        p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
        tp_h = truncated_measure.TruncatedGaussianMeasure(
            measure=p_h, lower_limit=0.0, upper_limit=jnp.inf
        )
        omega_dagger = tp_h.integrate("x")[:, 0]
        return omega_dagger

    def _update_omega_star(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        cubic_integral, quartic_integral = self._lower_bound_integrals(
            p_x=p_x,
            y=y,
            W_i=W_i,
            a_i=a_i,
            omega_star=omega_star,
            compute_fourth_order=True,
        )
        cubic_integral = jnp.where(cubic_integral != 0.0, cubic_integral, 1.0)
        omega_star = (quartic_integral / cubic_integral)[0]

        return omega_star

    def k_func(
        self,
        p_x: pdf.GaussianPDF,
        W_i: Float[Array, "Dx+1"],
        omega_dagger: Float[Array, "R"],
    ) -> Float[Array, "R"]:
        w0 = W_i[None, 0]
        w = W_i[None, 1:]
        c0 = jnp.log(1.0 + omega_dagger)
        c1 = 1 / (1.0 + omega_dagger)
        p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
        tp_h = truncated_measure.TruncatedGaussianMeasure(
            measure=p_h, lower_limit=0.0, upper_limit=jnp.inf
        )
        Zh = tp_h.integrate()
        Eh = tp_h.integrate("x")[:, 0]
        return Zh * c0 + c1 * (Eh - Zh * omega_dagger)

    def _lower_bound_integrals(
        self,
        p_x: measure.GaussianMeasure,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
        omega_star: Float[Array, "N"],
        compute_fourth_order: bool = False,
    ) -> Float[Array, "N"]:
        w0 = W_i[None, 0]
        w = W_i[None, 1:]
        a_projected_M = jnp.einsum("ab,cad->cbd", a_i[:, None], self.M)
        a_projected_yb = jnp.einsum("ab,ca->cb", a_i[:, None], y - self.b)

        nu_phi = -1.0 / (1.0 + omega_star)
        ln_beta_phi = -jnp.log(1.0 + omega_star) + omega_star / (1.0 + omega_star)
        phi_h_factor = factor.LinearFactor(nu=nu_phi[:, None], ln_beta=ln_beta_phi)
        if self.Dx == 1:
            c1 = a_projected_M[:, 0, 0] / w[:, 0]
            c0 = -(a_projected_yb[:, 0] + c1 * w0)
            p_h = p_x.get_density_of_linear_sum(w[None], w0[None])
        else:
            sum_weights = jnp.tile(
                jnp.concatenate([-a_projected_M[:, 0], w])[None],
                (a_projected_yb.shape[0], 1, 1),
            )
            sum_bias = jnp.hstack(
                [a_projected_yb, jnp.tile(w0[None], (a_projected_yb.shape[0], 1))]
            )
            p_hg = p_x.get_density_of_linear_sum(sum_weights, sum_bias)
            p_h = p_hg.get_marginal(jnp.array([1]))
            p_g_given_h = p_hg.condition_on_explicit(jnp.array([1]), jnp.array([0]))
            c1 = p_g_given_h.M[:, 0, 0]
            c0 = p_g_given_h.b[:, 0]

        phi_h = p_h.hadamard(phi_h_factor, update_full=True)
        tp_h = truncated_measure.TruncatedGaussianMeasure(
            measure=phi_h, lower_limit=0.0
        )
        Eh = tp_h.integrate("x")[:, 0]
        Eh2 = tp_h.integrate("x**2")[:, 0]
        Eh3 = tp_h.integrate("x**k", k=3)[:, 0]
        cubic_integral = Eh * c0**2 + Eh3 * c1**2 + 2 * Eh2 * c1 * c0
        if self.Dx > 1:
            cubic_integral += Eh * p_g_given_h.Sigma[:, 0, 0]

        if compute_fourth_order:
            Eh4 = tp_h.integrate("x**k", k=4)[:, 0]
            quartic_integral = Eh2 * c0**2 + Eh4 * c1**2 + 2 * Eh3 * c1 * c0
            if self.Dx > 1:
                quartic_integral += Eh2 * p_g_given_h.Sigma[:, 0, 0]
            return cubic_integral[None], quartic_integral[None]
        else:
            return cubic_integral[None]


@dataclass(kw_only=True)
class HeteroscedasticConditional:
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A_vec: Float[Array, "Dy*Da-Dy*(Dy-1)/2"]
    W: Float[Array, "Dk Dx+1"]
    link_function: str = field(default="exp")
    Sigma: Float[Array, "1 Dy Dy"] = field(init=False)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    conditional_instance: HeteroscedasticBaseConditional = field(init=False)

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
            raise NotImplementedError(
                "Diagonal matrix can have at most as many entries as A has columns."
            )

        self.Sigma = jnp.einsum("abc,adc->abd", self.A, self.A)
        self.Lambda, self.ln_det_Sigma = linalg.invert_matrix(self.Sigma)
        self.conditional_instance = self.conditional_class_dict[self.link_function](
            M=self.M,
            b=self.b,
            A=self.A,
            W=self.W,
            Sigma=self.Sigma,
            Lambda=self.Lambda,
            ln_det_Sigma=self.ln_det_Sigma,
        )

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
        return (self.A_vec.shape[0] - self.Dy * (self.Dy + 1) // 2) // self.Dy + self.Dy

    @property
    def Dk(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return self.W.shape[0]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.conditional_instance(*args, **kwds)

    @property
    def A(self) -> Float[Array, "1 Dy Da"]:
        num_entries_triangular = self.Dy * (self.Dy + 1) // 2
        indices = jnp.triu_indices(self.Dy)  # Generate upper triangular indices
        A_tria = jnp.zeros((self.Dy, self.Dy))  # Create an initial matrix of zeros
        A_tria = A_tria.at[indices].set(self.A_vec[:num_entries_triangular])
        A_ntria = self.A_vec[num_entries_triangular:].reshape(
            self.Dy, self.Da - self.Dy
        )
        A = jnp.array([jnp.concatenate([A_tria, A_ntria], axis=1)])
        return A

    # called when an attribute is not found:
    # source: https://stackoverflow.com/questions/65754399/conditional-inheritance-based-on-arguments-in-python
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.conditional_instance.__getattribute__(name)

    @property
    def conditional_class_dict(self) -> dict:
        return {
            "exp": HeteroscedasticExpConditional,
            "coshm1": HeteroscedasticCoshM1Conditional,
            "heaviside": HeteroscedasticHeavisideConditional,
            "ReLU": HeteroscedasticReLUConditional,
        }


@dataclass(kw_only=True)
class ScalableHeteroscedasticConditional(HeteroscedasticConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A_vec: Float[Array, "Dy+Dy*(Da-Dy)"]
    W: Float[Array, "Dk Dx+1"]
    link_function: str = field(default="exp")
    Sigma: Float[Array, "1 Dy Dy"] = field(init=False)
    Lambda: Float[Array, "1 Dy Dy"] = field(init=False)
    ln_det_Sigma: Float[Array, "1"] = field(init=False)
    conditional_instance: HeteroscedasticBaseConditional = field(init=False)

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
            raise NotImplementedError(
                "Diagonal matrix can have at most as many entries as A has columns."
            )

        i, j = jnp.diag_indices(self.Dy)
        Sigma = jnp.einsum(
            "abc,adc->abd", self.A[:, :, self.Dy :], self.A[:, :, self.Dy :]
        )
        self.Sigma = Sigma.at[..., i, j].add(self.A[..., i, j] ** 2)
        self.Lambda, self.ln_det_Sigma = linalg.invert_woodbury_diag(
            A_diagonal=self.A[..., i, j] ** 2,
            B_inv=jnp.eye(self.Da - self.Dy)[None],
            M=self.A[:, :, self.Dy :],
            ln_det_B=jnp.zeros(self.R),
        )
        self.conditional_instance = self.conditional_class_dict[self.link_function](
            M=self.M,
            b=self.b,
            A=self.A,
            W=self.W,
            Sigma=self.Sigma,
            Lambda=self.Lambda,
            ln_det_Sigma=self.ln_det_Sigma,
        )

    @property
    def Da(self) -> int:
        r"""Number of orthonormal low rank vectors :math:`U`."""
        return (self.A_vec.shape[0] - self.Dy) // self.Dy + self.Dy

    @property
    def A(self) -> Float[Array, "1 Dy Da"]:
        num_entries_diag = self.Dy
        indices = jnp.diag_indices(self.Dy)  # Generate upper triangular indices
        A_tria = jnp.zeros((self.Dy, self.Dy))  # Create an initial matrix of zeros
        A_tria = A_tria.at[indices].set(self.A_vec[:num_entries_diag])
        A_ntria = self.A_vec[num_entries_diag:].reshape(self.Dy, self.Da - self.Dy)
        A = jnp.array([jnp.concatenate([A_tria, A_ntria], axis=1)])
        return A

    def integrate_Sigma_x(
        self, p_x: pdf.GaussianPDF, invert: bool = False
    ) -> Float[Array, "Dy Dy"]:
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
            "b, adb, aeb -> ade",
            D_int,
            self.A[:, :, -self.Dk :],
            self.A[:, :, -self.Dk :],
        )
        if invert:
            D = jnp.array([jnp.diag(D_int)])
            D_inv, ln_det_D = linalg.invert_diagonal(D)
            Lambda_int, ln_det_Sigma_int = linalg.invert_woodbury(
                self.Lambda,
                D_inv,
                self.A[:, :, -self.Dk :],
                self.ln_det_Sigma,
                ln_det_D,
            )
            return Sigma_int, Lambda_int, ln_det_Sigma_int
        else:
            return Sigma_int

    def get_expected_moments(
        self, p_x: pdf.GaussianPDF, invert: bool = False
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
        # Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
        #    "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        # )
        # Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]

        if invert:
            Sigma_int, Lambda_int, ln_det_Sigma_int = self.integrate_Sigma_x(
                p_x, invert=True
            )
            Sigma_y = Sigma_int + jnp.einsum(
                "abc, adb, aec -> ade", p_x.Sigma, self.M, self.M
            )
            Lambda_y, ln_det_Sigma_y = linalg.invert_woodbury(
                Lambda_int, p_x.Lambda, self.M, ln_det_Sigma_int, p_x.ln_det_Sigma
            )
            return mu_y, Sigma_y, Lambda_y, ln_det_Sigma_y
        else:
            Sigma_y = self.integrate_Sigma_x(p_x) + jnp.einsum(
                "abc, adb, aec -> ade", p_x.Sigma, self.M, self.M
            )
            return mu_y, Sigma_y

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
        mu_y, Sigma_y, Lambda_y, ln_det_Sigma_y = self.get_expected_moments(
            p_x, invert=True
        )
        cov_yx = self.get_expected_cross_terms(p_x)
        mu_xy = jnp.concatenate([p_x.mu, mu_y], axis=1)
        print(p_x.Sigma.shape, cov_yx.shape, Sigma_y.shape)
        Sigma_xy = jnp.block(
            [[p_x.Sigma, cov_yx.transpose((0, 2, 1))], [cov_yx, Sigma_y]]
        )
        Lambda_xy, ln_det_Sigma_xy = linalg.invert_block_matrix(
            Lambda_y,
            p_x.Sigma,
            cov_yx.transpose((0, 2, 1)),
            ln_det_Sigma_y,
            A_is_up=False,
        )
        print(Sigma_xy.shape, mu_xy.shape)
        p_xy = pdf.GaussianPDF(
            Sigma=Sigma_xy, mu=mu_xy, Lambda=Lambda_xy, ln_det_Sigma=ln_det_Sigma_xy
        )
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
        mu_y, Sigma_y, Lambda_y, ln_det_Sigma_y = self.get_expected_moments(
            p_x, invert=True
        )
        cov_yx = self.get_expected_cross_terms(p_x)
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = p_x.mu - jnp.einsum("abc,ac->ab", M_new, mu_y)
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
        mu_y, Sigma_y, Lambda_y, ln_det_Sigma_y = self.get_expected_moments(
            p_x, invert=True
        )
        p_y = pdf.GaussianPDF(
            Sigma=Sigma_y, mu=mu_y, Lambda=Lambda_y, ln_det_Sigma=ln_det_Sigma_y
        )
        return p_y
