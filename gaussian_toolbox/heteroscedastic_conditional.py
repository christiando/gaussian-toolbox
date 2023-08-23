__author__ = "Christian Donner"

from jax import numpy as jnp
import jax
from typing import Any, Tuple, Union
from . import pdf, factor, measure, conditional
from .utils import linalg

from .utils.dataclass import dataclass
from jaxtyping import Array, Float
from jax import lax
from jax import scipy as jsc
from jax.scipy.optimize import minimize as jax_minimize

from dataclasses import field
from jax import vmap
from abc import abstractmethod
from gaussian_toolbox.experimental import truncated_measure
from jax.flatten_util import ravel_pytree


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
    A: Float[Array, "1 Dy Dy"]
    W: Float[Array, "Dy Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

    @abstractmethod
    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        pass

    @abstractmethod
    def log_link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
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

    def linear_layer(self, x: Float[Array, "N Dx"]) -> Float[Array, "N Dy"]:
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
        log_D_x = self.log_link_function(h)
        D_x = jnp.exp(log_D_x)
        Sigma = jnp.einsum(
            "abc,dc->abd",
            jnp.einsum("ab,cb->cab", self.A[0, :, :], D_x),
            self.A[0, :, :],
        )
        if invert:
            if self.Dy == 1:
                Lambda = 1.0 / Sigma
                ln_det_Sigma_y_x = jnp.log(Sigma[:, 0, 0])
            else:
                G_x = jnp.exp(-log_D_x)
                A_inv = jnp.einsum("abc, acd -> abd", self.Lambda, self.A)
                Lambda = jnp.einsum(
                    "abc,dc->abd",
                    jnp.einsum("ab,cb->cab", A_inv[0, :, :], G_x),
                    A_inv[0, :, :],
                )
                ln_det_Sigma_y_x = self.ln_det_Sigma + jnp.sum(log_D_x, axis=1)
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
        Sigma_int = jnp.einsum(
            "b, adb, aeb -> ade",
            D_int,
            self.A[:, :, :],
            self.A[:, :, :],
        )
        Sigma_int = 0.5 * (Sigma_int + jnp.swapaxes(Sigma_int, -2, -1))
        return Sigma_int

    @abstractmethod
    def _integrate_noise_diagonal(self, p_x: pdf.GaussianPDF) -> Float[Array, "N Dy"]:
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
        Sigma_xy = jnp.block(
            [[p_x.Sigma, cov_yx.transpose((0, 2, 1))], [cov_yx, Sigma_y]]
        )
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

    def _construct_variational_posterior(self, q_dict):
        q_mu = q_dict["mu"]
        q_log_Sigma_diag = q_dict["q_log_Sigma_diag"]
        # q_log_Sigma_diag = jnp.clip(
        #    q_log_Sigma_diag, -10.0 * jnp.log(10), 10.0 * jnp.log(10)
        # )
        # triu_indices = jnp.triu_indices(self.Dx)
        diag_indices = jnp.diag_indices(self.Dx)
        q_Sigma = jnp.zeros((1, self.Dx, self.Dx))
        q_Sigma = q_Sigma.at[:, diag_indices[0], diag_indices[1]].set(
            jnp.exp(q_log_Sigma_diag)
        )
        q_Lambda = jnp.zeros((1, self.Dx, self.Dx))
        q_Lambda = q_Lambda.at[:, diag_indices[0], diag_indices[1]].set(
            jnp.exp(-q_log_Sigma_diag)
        )
        # q_L = q_L.at[:, diag_indices[0], diag_indices[1]].set(
        #    jnp.abs(q_L[:, diag_indices[0], diag_indices[1]]) + 1e-4
        # )
        # q_Sigma = jnp.einsum("acb,adc->abd", q_L, q_L.transpose((0, 2, 1)))
        # L = (q_L, False)
        # q_Lambda = jsc.linalg.cho_solve(
        #    L, jnp.tile(jnp.eye(q_Sigma.shape[1])[None], (len(q_Sigma), 1, 1))
        # )
        q_ln_det_Sigma = jnp.sum(q_log_Sigma_diag, axis=1)
        variational_posterior = pdf.GaussianPDF(
            mu=q_mu, Sigma=q_Sigma, Lambda=q_Lambda, ln_det_Sigma=q_ln_det_Sigma
        )
        return variational_posterior

    def _get_neg_elbo(self, q_dict, p_x, y):
        variational_posterior = self._construct_variational_posterior(q_dict)
        expected_likelihood = jnp.sum(
            self.integrate_log_conditional_y(variational_posterior, y)
        )
        dkl = variational_posterior.kl_divergence(p_x)
        elbo = expected_likelihood - jnp.sum(dkl)
        return -elbo

    def affine_variational_conditional_transformation(self, p_x, y):
        init_mu = p_x.mu  # jnp.mean(q_moment_matching.mu, axis=0, keepdims=True)
        diag_indices = jnp.diag_indices(self.Dx)
        # without -1 sometimes numerical optimization got stuck in local minima
        init_log_sigma = (
            0.5 * jnp.log(p_x.Sigma[:, diag_indices[0], diag_indices[1]]) - 1.0
        )
        q_dict = {"mu": init_mu, "q_log_Sigma_diag": init_log_sigma}
        x0_flat, unravel = ravel_pytree(q_dict)

        def fun(x_flat):
            q_dict = unravel(x_flat)
            return self._get_neg_elbo(q_dict, p_x, y) / y.shape[0]

        result = jax_minimize(fun, x0_flat, method="BFGS")
        best_posterior = result.x
        q_dict = unravel(best_posterior)
        variational_posterior = self._construct_variational_posterior(q_dict)
        return variational_posterior

    def integrate_log_conditional_y(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"], **kwargs
    ) -> Float[Array, "N"]:
        quadratic_term = self.get_lb_quadratic_term(p_x, y)
        log_det = self.get_lb_log_det(p_x)
        log_p_y = -0.5 * (quadratic_term + log_det + self.Dy * jnp.log(2.0 * jnp.pi))
        return log_p_y

    def get_lb_log_det(self, p_x: pdf.GaussianPDF) -> Float[Array, "N"]:
        return self.ln_det_Sigma + self.integrate_heteroscedastic_log_det(p_x)

    @abstractmethod
    def integrate_heteroscedastic_log_det(
        self, p_x: pdf.GaussianPDF
    ) -> Float[Array, "N"]:
        pass

    def get_lb_quadratic_term(
        self, p_x: pdf.GaussianPDF, y: Float[Array, "N Dy"]
    ) -> Float[Array, "N"]:

        A_inv = jnp.einsum("abc, acd -> abd", self.Lambda, self.A)
        get_lb_heteroscedastic_term = jnp.sum(
            vmap(lambda W, A_inv: self.get_lb_heteroscedastic_term_i(p_x, y, W, A_inv))(
                self.W, A_inv[0, :, :].T
            ),
            axis=0,
        )
        return get_lb_heteroscedastic_term

    @abstractmethod
    def get_lb_heteroscedastic_term_i(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
    ) -> Float[Array, "N"]:
        pass


@dataclass(kw_only=True)
class HeteroscedasticExpConditional(HeteroscedasticBaseConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Dy"]
    W: Float[Array, "Dy Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return jnp.exp(h)

    def log_link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return h

    def _integrate_noise_diagonal(self, p_x: pdf.GaussianPDF) -> Float[Array, "N Dy"]:
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
        ln_beta = 0 * self.W[:, 0]
        exp_h = factor.LinearFactor(nu=nu, ln_beta=ln_beta)
        D_int = p_x.multiply(exp_h, update_full=True).integrate()
        return D_int

    def integrate_heteroscedastic_log_det(
        self, p_x: pdf.GaussianPDF
    ) -> Float[Array, "N"]:
        r"""Integrate the heteroscedastic noise diagonal with respect to :math:`p(X)`.

        .. math::

            \int \log D(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        log_det_int = jax.vmap(
            lambda W: p_x.integrate("(Ax+a)", A_mat=W[1:][None], a_vec=0 * W[0][None])
        )(self.W)
        return jnp.sum(log_det_int, axis=0)[:, 0]

    def get_lb_heteroscedastic_term_i(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
    ) -> Float[Array, "N"]:
        r"""Compute the lower bound term for the ith heteroscedastic noise component.

        .. math::

            \int \log D_i(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        nu = -W_i[1:]
        ln_beta = -0 * W_i[0]
        exp_mh = factor.LinearFactor(nu=nu[None], ln_beta=ln_beta[None])
        px_measure = p_x.multiply(exp_mh, update_full=True)
        a_projected_M = jnp.einsum("ab,cad->cbd", a_i[:, None], self.M)
        a_projected_yb = jnp.einsum("ab,ca->cb", a_i[:, None], y - self.b)
        quadratic_integral = px_measure.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        return quadratic_integral


@dataclass(kw_only=True)
class HeteroscedasticSigmoidConditional(HeteroscedasticBaseConditional):
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A: Float[Array, "1 Dy Dy"]
    W: Float[Array, "Dy Dx+1"]
    Sigma: Float[Array, "1 Dy Dy"]
    Lambda: Float[Array, "1 Dy Dy"]
    ln_det_Sigma: Float[Array, "1"]

    def link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return 1 / (1.0 + jnp.exp(-h))

    def log_link_function(self, h: Float[Array, "..."]) -> Float[Array, "..."]:
        """Link function for the heteroscedastic noise."""
        return -jnp.log1p(jnp.exp(-h))

    def _integrate_noise_diagonal(self, p_x: pdf.GaussianPDF) -> Float[Array, "N Dy"]:
        r"""Integrate the noise diagonal with respect to :math:`p(X)`.

        .. math::

            \int D(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        p_x_tiled = p_x.tile(self.Dy)
        D_int = p_x_tiled.integrate(
            "sigmoid(Ax+a)", A_mat=self.W[:, 1:], a_vec=0 * self.W[:, 0]
        )
        return D_int

    def integrate_heteroscedastic_log_det(
        self, p_x: pdf.GaussianPDF
    ) -> Float[Array, "N"]:
        r"""Integrate the heteroscedastic noise diagonal with respect to :math:`p(X)`.

        .. math::

            \int \log D(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        get_expected_h = jax.vmap(
            lambda W: p_x.integrate("(Ax+a)", A_mat=W[1:][None], a_vec=0 * W[0][None])
        )
        omega = jax.lax.stop_gradient(get_expected_h(self.W))[:, :, 0]
        Eh = get_expected_h(self.W)[:, :, 0]
        log_det_int = -jnp.log1p(jnp.exp(-omega)) + jnp.exp(
            -jnp.log1p(jnp.exp(omega))
        ) * (Eh - omega)
        return log_det_int

    def get_lb_heteroscedastic_term_i(
        self,
        p_x: pdf.GaussianPDF,
        y: Float[Array, "N Dy"],
        W_i: Float[Array, "Dx+1"],
        a_i: Float[Array, "Dy"],
    ) -> Float[Array, "N"]:
        r"""Compute the lower bound term for the ith heteroscedastic noise component.

        .. math::

            \int \log D_i(x) p(x) {\rm d}x.

        Args:
            p_x: The density the noise diagonal is integrated with.

        Returns:
            Integrated noise diagonal.
        """
        # int f(h(z)) dphi(z)
        nu = -W_i[1:]
        ln_beta = -0 * W_i[0]
        exp_mh = factor.LinearFactor(nu=nu[None], ln_beta=ln_beta[None])
        px_measure = p_x.multiply(exp_mh, update_full=True)
        a_projected_M = jnp.einsum("ab,cad->cbd", a_i[:, None], self.M)
        a_projected_yb = jnp.einsum("ab,ca->cb", a_i[:, None], y - self.b)
        quadratic_integral_a = px_measure.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        quadratic_integral_b = p_x.integrate(
            "(Ax+a)'(Bx+b)",
            A_mat=-a_projected_M,
            a_vec=a_projected_yb,
            B_mat=-a_projected_M,
            b_vec=a_projected_yb,
        )
        quadratic_integral = quadratic_integral_a + quadratic_integral_b
        return quadratic_integral


@dataclass(kw_only=True)
class HeteroscedasticConditional:
    M: Float[Array, "1 Dy Dx"]
    b: Float[Array, "1 Dy"]
    A_vec: Float[Array, "Dy**2-Dy*(Dy-1)/2"]
    W: Float[Array, "Dy Dx+1"]
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
            assert self.Dy <= self.Da
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.conditional_instance(*args, **kwds)

    @property
    def A(self) -> Float[Array, "1 Dy Dy"]:
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
            "sigmoid": HeteroscedasticSigmoidConditional,
        }
