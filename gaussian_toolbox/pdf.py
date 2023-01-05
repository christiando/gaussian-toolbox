##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) probability densities.                    #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

from jax.random import PRNGKey
import jax
from jax import numpy as jnp
import numpy as np

# from .
from . import measure
from .utils.linalg import invert_matrix, invert_diagonal
from jaxtyping import Float, Array, Int
from typing import Dict

from .utils.dataclass import dataclass
from dataclasses import field

@dataclass(kw_only=True)
class GaussianPDF(measure.GaussianMeasure):
    """A normalized Gaussian density, with specified mean and covariance matrix.

    Args:
        Sigma: Covariance matrices of the Gaussian densities.
        mu: Mean of the Gaussians.
        Lambda: Information (precision) matrix of the Gaussians.
        ln_det_Sigma: Log determinant of the covariance matrix.
    """
    Sigma: Float[Array, "R D D"]
    mu: Float[Array, "R D"]
    Lambda: Float[Array, "R D D"] = None
    ln_det_Sigma: Float[Array, "R"] = None
    nu: Float[Array, "R D"] = field(init=False)
    ln_beta: Float[Array, "R"] = field(init=False)
    lnZ: Float[Array, "R"] = field(default=None, init=False)
    
    def __post_init__(self):
        if self.Lambda is None:
            self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        elif self.ln_det_Sigma is None:
            self.ln_det_Sigma = jnp.linalg.slogdet(self.Sigma)[1]
        self.nu = jnp.einsum("abc,ab->ac", self.Lambda, self.mu)
        self._prepare_integration()
        self.normalize()

    def __str__(self) -> str:
        return "Gaussian density p(x)"

    def sample(self, key: PRNGKey, num_samples: int) -> Float[Array, "N R D"]:
        """Sample from the Gaussian density.

        Args:
            key: Jax pseudo random number generator.
            num_samples: Number og samples.

        Returns:
            Samples.
        """
        rand_nums = jax.random.normal(key, (num_samples, self.R, self.D))
        L = jnp.linalg.cholesky(self.Sigma)
        x_samples = self.mu[None] + jnp.einsum("abc,dac->dab", L, rand_nums)
        return x_samples

    def slice(self, indices: Int[Array, "R_new"]) -> "GaussianPDF":
        """Return an object with only the specified entries.

        Args:
            indices: The entries that should be contained in the
                returned object.

        Returns:
            The resulting Gaussian density.
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        mu_new = jnp.take(self.mu, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        # Lambda_new = lax.dynamic_index_in_dim(self.Lambda, indices, axis=0)
        # Sigma_new = lax.dynamic_index_in_dim(self.Sigma, indices, axis=0)
        # mu_new = lax.dynamic_index_in_dim(self.mu, indices, axis=0)
        # ln_det_Sigma_new = lax.dynamic_index_in_dim(self.ln_det_Sigma, indices, axis=0)
        new_measure = GaussianPDF(Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new)
        return new_measure

    def update(self, indices: Int[Array, "R_update"], density: "GaussianPDF"):
        """Update densities at indicated entries.

        Args:
            indices: The entries that should be updated.
            density: New densities.
        """
        self.Lambda = self.Lambda.at[indices].set(density.Lambda)
        self.Sigma = self.Sigma.at[indices].set(density.Sigma)
        self.mu = self.mu.at[indices].set(density.mu)
        self.ln_det_Sigma = self.ln_det_Sigma.at[indices].set(density.ln_det_Sigma)
        #self.ln_det_Lambda = self.ln_det_Lambda.at[indices].set(density.ln_det_Lambda)
        self.lnZ = self.lnZ.at[indices].set(density.lnZ)
        self.nu = self.nu.at[indices].set(density.nu)
        self.ln_beta = self.ln_beta.at[indices].set(density.ln_beta)

    def get_marginal(self, dim_x: Int[Array, "Dx"]) -> "GaussianPDF":
        """Get the marginal of the indicated dimensions.

        Args:
            dim_x: The dimensions of the variables, the marginal is
                required for.

        Returns:
            The resulting marginal Gaussian density.
        """
        idx = jnp.ix_(jnp.arange(self.Sigma.shape[0]), dim_x, dim_x)
        Sigma_new = self.Sigma[idx]
        idx = jnp.ix_(jnp.arange(self.mu.shape[0]), dim_x)
        mu_new = self.mu[idx]
        marginal_density = GaussianPDF(Sigma=Sigma_new, mu=mu_new)
        return marginal_density

    def entropy(self) -> Float[Array, "R"]:
        r"""Computes the entropy of the density.

        .. math::

            H_X = -\int p(X)\log p(X) {\rm d}X

        Returns:
            Entropy of the density
        """
        entropy = 0.5 * (self.D * (1.0 + jnp.log(2 * jnp.pi)) + self.ln_det_Sigma)
        return entropy

    def kl_divergence(self, p1: "GaussianPDF") -> Float[Array, "R"]:
        r"""Compute the Kulback Leibler divergence between two multivariate Gaussians.

                .. math

           D_KL(p|p1) = \int p(X)\log \frac{p(X)}{p_1(X)} {\rm d}X

                :param p1: The other Gaussian Density.
                :return: Kulback Leibler divergence.
        """
        assert self.R == p1.R or p1.R == 1 or self.R == 1
        assert self.D == p1.D
        dmu = p1.mu - self.mu
        dmu_Sigma_dmu = jnp.einsum(
            "ab,ab->a", jnp.einsum("ab,abc->ac", dmu, p1.Lambda), dmu
        )
        tr_Lambda_Sigma = jnp.trace(
            jnp.einsum("abc,acd->abd", p1.Lambda, self.Sigma), axis1=-2, axis2=-1
        )
        kl_div = 0.5 * (
            tr_Lambda_Sigma
            + dmu_Sigma_dmu
            - self.D
            + p1.ln_det_Sigma
            - self.ln_det_Sigma
        )
        return kl_div

    def condition_on(self, dim_y: Float[Array, "Dy"]) -> "ConditionalGaussianPDF":
        """Return density conditioned on indicated dimensions, i.e. :math:`p(X|Y)`.

        Args:
            dim_y: The dimensions of the variables, that should be
                conditioned on.

        Returns:
            The corresponding conditional Gaussian density
            :math:`p(X|Y)`.
        """
        from . import conditional

        dim_xy = jnp.arange(self.D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        # dim_x = dim_xy[jnp.logical_not(jnp.isin(dim_xy, dim_y))]
        Lambda_x = self.Lambda[:, dim_x][:, :, dim_x]
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        M_x = -jnp.einsum("abc,acd->abd", Sigma_x, self.Lambda[:, dim_x][:, :, dim_y])
        b_x = self.mu[:, dim_x] - jnp.einsum("abc,ac->ab", M_x, self.mu[:, dim_y])
        return conditional.ConditionalGaussianPDF(
            M=M_x, b=b_x, Sigma=Sigma_x, Lambda=Lambda_x, ln_det_Sigma=-ln_det_Lambda_x
        )

    def condition_on_explicit(
        self, dim_y: Float[Array, "R Dy"], dim_x: Float[Array, "R Dx"]
    ) -> "ConditionalGaussianPDF":
        """Returns density conditioned on indicated dimensions, i.e. :math:`p(X|Y)`.

        Args:
            dim_y: The dimensions of the variables, that should be
                conditioned on.
            dim_x: The dimensions of the variables, that should be still
                be free.

        Returns:
            The corresponding conditional Gaussian density
            :math:`p(X|Y)`.
        """
        from . import conditional

        Lambda_x = self.Lambda[:, dim_x][:, :, dim_x]
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        M_x = -jnp.einsum("abc,acd->abd", Sigma_x, self.Lambda[:, dim_x][:, :, dim_y])
        b_x = self.mu[:, dim_x] - jnp.einsum("abc,ac->ab", M_x, self.mu[:, dim_y])
        return conditional.ConditionalGaussianPDF(
            M=M_x, b=b_x, Sigma=Sigma_x, Lambda=Lambda_x, ln_det_Sigma=-ln_det_Lambda_x
        )

    def to_dict(self) -> Dict:
        """Write Gaussian into dict.

        Returns:
            Dictionary with relevant parameters.
        """
        density_dict = {
            "Sigma": self.Sigma,
            "mu": self.mu,
            "Lambda": self.Lambda,
            "ln_det_Sigma": self.ln_det_Sigma,
        }
        return density_dict

@dataclass(kw_only=True)
class GaussianDiagPDF(GaussianPDF, measure.GaussianDiagMeasure):
    """A normalized Gaussian density, with specified mean and covariance matrix.

    :math:`\Sigma` should be diagonal (and hence :math:`\Lambda`).

    Args:
        Sigma: Covariance matrices of the Gaussian densities. Must be
            diagonal.
        mu: Mean of the Gaussians.
        Lambda: Information (precision) matrix of the Gaussians.
        ln_det_Sigma: Log determinant of the covariance matrix.
    """
    Sigma: Float[Array, "R D D"]
    mu: Float[Array, "R D"]
    Lambda: Float[Array, "R D D"] = None
    ln_det_Sigma: Float[Array, "R"] = None
    nu: Float[Array, "R D"] = field(init=False)
    ln_beta: Float[Array, "R"] = field(init=False)
    lnZ: Float[Array, "R"] = field(default=None, init=False)
    
    def __post_init__(self):
        self.Lambda, self.ln_det_Sigma = invert_diagonal(self.Sigma)
        if self.nu is None:
            self.nu = jnp.zeros((self.R, self.D))
        if self.ln_beta is None:
            self.ln_beta = jnp.zeros((self.R))
        self.Sigma = self.Sigma
        self.ln_det_Lambda = self.ln_det_Lambda
        self.ln_det_Sigma = self.ln_det_Sigma
        self._prepare_integration()
        self.normalize()
        

    def slice(self, indices: Int[Array, "R_new"]) -> "GaussianDiagPDF":
        """Return an object with only the specified entries.

        Args:
            indices: The entries that should be contained in the
                returned object.

        Returns:
            The resulting Gaussian diagonal density.
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        mu_new = jnp.take(self.mu, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = GaussianDiagPDF(Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new,)
        return new_measure

    def update(self, indices: Int[Array, "R_update"], density: "GaussianDiagPDF"):
        """Update densities at indicated entries.

        Args:
            indices: The entries that should be updated.
            density: New densities.
        """
        self.Lambda = self.Lambda.at[indices].set(density.Lambda)
        self.Sigma = self.Sigma.at[indices].set(density.Sigma)
        self.mu = self.mu.at[indices].set(density.mu)
        self.ln_det_Sigma = self.ln_det_Sigma.at[indices].set(density.ln_det_Sigma)
        #self.ln_det_Lambda = self.ln_det_Lambda.at[indices].set(density.ln_det_Lambda)
        self.lnZ = self.lnZ.at[indices].set(density.lnZ)
        self.nu = self.nu.at[indices].set(density.nu)
        self.ln_beta = self.ln_beta.at[indices].set(density.ln_beta)

    def get_marginal(self, dim_idx: Int[Array, "Dx"]) -> "GaussianDiagPDF":
        """Get the marginal of the indicated dimensions.

        Args:
            dim_idx: The dimensions of the variables, the marginal is
                required for.

        Returns:
            The resulting marginal Gaussian density.
        """
        Sigma_new = self.Sigma[:, dim_idx][:, :, dim_idx]
        mu_new = self.mu[:, dim_idx]
        marginal_density = GaussianDiagPDF(Sigma=Sigma_new, mu=mu_new)
        return marginal_density
