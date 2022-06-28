##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) probability densities.                    #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
__all__ = ["GaussianDensity", "GaussianDiagDensity"]

from jax import numpy as jnp
from jax import lax, random
import numpy as np
from typing import Iterable

# from .
from . import measures
from ..utils.linalg import invert_matrix, invert_diagonal


class GaussianDensity(measures.GaussianMeasure):
    def __init__(
        self,
        Sigma: jnp.ndarray,
        mu: jnp.ndarray,
        Lambda: jnp.ndarray = None,
        ln_det_Sigma: jnp.ndarray = None,
    ):
        """A normalized Gaussian density, with specified mean and covariance matrix.

        :param Sigma: Covariance matrices of the Gaussian densities. Dimensions should be [R, D, D].
        :type Sigma: jnp.ndarray
        :param mu: Mean of the Gaussians. Dimensions should be
        :type mu: jnp.ndarray
        :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [R, D, D], defaults to None
        :type Lambda: jnp.ndarray, optional
        :param ln_det_Sigma: Log determinant of the covariance matrix. Dimensions should be [R], defaults to None
        :type ln_det_Sigma: jnp.ndarray, optional
        """
        if Lambda is None:
            Lambda, ln_det_Sigma = invert_matrix(Sigma)
        elif ln_det_Sigma is None:
            ln_det_Sigma = jnp.linalg.slogdet(Sigma)[1]
        nu = jnp.einsum("abc,ab->ac", Lambda, mu)
        super().__init__(Lambda=Lambda, nu=nu)
        self.Sigma = Sigma
        self.mu = mu
        self.ln_det_Sigma = ln_det_Sigma
        self.ln_det_Lambda = -ln_det_Sigma
        self._prepare_integration()
        self.normalize()

    def __str__(self) -> str:
        return "Gaussian density p(x)"

    def sample(self, num_samples: int) -> jnp.ndarray:
        """Sample from the Gaussian density.

        :param num_samples: Number of samples that are generated.
        :type num_samples: int
        :return: Samples. Dimensions are [num_samples, R, D].
        :rtype: jnp.ndarray
        """
        rand_nums = np.random.randn(num_samples, self.R, self.D)
        L = jnp.linalg.cholesky(self.Sigma)
        x_samples = self.mu[None] + jnp.einsum("abc,dac->dab", L, rand_nums)
        return x_samples

    def slice(self, indices: jnp.ndarray) -> "GaussianDensity":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting Gaussian density.
        :rtype: GaussianDensity
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        mu_new = jnp.take(self.mu, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        # Lambda_new = lax.dynamic_index_in_dim(self.Lambda, indices, axis=0)
        # Sigma_new = lax.dynamic_index_in_dim(self.Sigma, indices, axis=0)
        # mu_new = lax.dynamic_index_in_dim(self.mu, indices, axis=0)
        # ln_det_Sigma_new = lax.dynamic_index_in_dim(self.ln_det_Sigma, indices, axis=0)
        new_measure = GaussianDensity(Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new)
        return new_measure

    def update(self, indices: jnp.ndarray, density: "GaussianDensity"):
        """Update densities at indicated entries.

        :param indices: The entries that should be updated.
        :type indices: jnp.ndarray
        :param density: New densities.
        :type density: GaussianDensity
        """
        self.Lambda = self.Lambda.at[indices].set(density.Lambda)
        self.Sigma = self.Sigma.at[indices].set(density.Sigma)
        self.mu = self.mu.at[indices].set(density.mu)
        self.ln_det_Sigma = self.ln_det_Sigma.at[indices].set(density.ln_det_Sigma)
        self.ln_det_Lambda = self.ln_det_Lambda.at[indices].set(density.ln_det_Lambda)
        self.lnZ = self.lnZ.at[indices].set(density.lnZ)
        self.nu = self.nu.at[indices].set(density.nu)
        self.ln_beta = self.ln_beta.at[indices].set(density.ln_beta)

    def get_marginal(self, dim_x: jnp.ndarray) -> "GaussianDensity":
        """Get the marginal of the indicated dimensions.

        :param dim_x: The dimensions of the variables, the marginal is required for.
        :type dim_x: jnp.ndarray
        :return: The resulting marginal Gaussian density.
        :rtype: GaussianDensity
        """
        idx = jnp.ix_(jnp.arange(self.Sigma.shape[0]), dim_x, dim_x)
        Sigma_new = self.Sigma[idx]
        idx = jnp.ix_(jnp.arange(self.mu.shape[0]), dim_x)
        mu_new = self.mu[idx]
        marginal_density = GaussianDensity(Sigma_new, mu_new)
        return marginal_density

    def entropy(self) -> jnp.ndarray:
        """Computes the entropy of the density.
        
        H[p] = -\int p(x)ln p(x) dx

        :return: Entropy of the density 
        :rtype: jnp.ndarray [R]
        """
        entropy = 0.5 * (self.D * (1.0 + jnp.log(2 * jnp.pi)) + self.ln_det_Sigma)
        return entropy

    def kl_divergence(self, p1: "GaussianDensity") -> jnp.ndarray:
        """ Compute the Kulback Leibler divergence between two multivariate Gaussians.
        
        D_KL(p|p1) = \int p(x)\log p(x)/p1(x) dx

        :param p1: The other Gaussian Density.
        :type p1: GaussianDensity
        :return: Kulback Leibler divergence. Dimensions should be [R].
        :rtype: jnp.ndarray
        """
        assert self.R == p1.R
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

    def condition_on(self, dim_y: jnp.ndarray) -> "ConditionalGaussianDensity":
        """Return density conditioned on indicated dimensions, i.e. p(x|y).

        :param dim_y: The dimensions of the variables, that should be conditioned on.
        :type dim_y: jnp.ndarray
        :return: The corresponding conditional Gaussian density p(x|y).
        :rtype: ConditionalGaussianDensity
        """
        from . import conditionals

        dim_xy = jnp.arange(self.D, dtype=jnp.int32)
        dim_x = jnp.setxor1d(dim_xy, dim_y)
        # dim_x = dim_xy[jnp.logical_not(jnp.isin(dim_xy, dim_y))]
        Lambda_x = self.Lambda[:, dim_x][:, :, dim_x]
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        M_x = -jnp.einsum("abc,acd->abd", Sigma_x, self.Lambda[:, dim_x][:, :, dim_y])
        b_x = self.mu[:, dim_x] - jnp.einsum("abc,ac->ab", M_x, self.mu[:, dim_y])
        return conditionals.ConditionalGaussianDensity(
            M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x
        )

    def condition_on_explicit(
        self, dim_y: jnp.ndarray, dim_x: jnp.ndarray
    ) -> "ConditionalGaussianDensity":
        """Returns density conditioned on indicated dimensions, i.e. p(x|y).

        :param dim_y: The dimensions of the variables, that should be conditioned on.
        :type dim_y: jnp.ndarray
        :param dim_x: The dimensions of the variables, that should be still be free.
        :type dim_x: jnp.ndarray
        :return: The corresponding conditional Gaussian density p(x|y).
        :rtype: ConditionalGaussianDensity
        """
        from . import conditionals

        Lambda_x = self.Lambda[:, dim_x][:, :, dim_x]
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        M_x = -jnp.einsum("abc,acd->abd", Sigma_x, self.Lambda[:, dim_x][:, :, dim_y])
        b_x = self.mu[:, dim_x] - jnp.einsum("abc,ac->ab", M_x, self.mu[:, dim_y])
        return conditionals.ConditionalGaussianDensity(
            M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x
        )

    def to_dict(self):
        density_dict = {
            "Sigma": self.Sigma,
            "mu": self.mu,
            "Lambda": self.Lambda,
            "ln_det_Sigma": self.ln_det_Sigma,
        }
        return density_dict


class GaussianDiagDensity(GaussianDensity, measures.GaussianDiagMeasure):
    def __init__(
        self,
        Sigma: jnp.ndarray,
        mu: jnp.ndarray,
        Lambda: jnp.ndarray = None,
        ln_det_Sigma: jnp.ndarray = None,
    ):
        """A normalized Gaussian density, with specified mean and covariance matrix. 
        
        Sigma should be diagonal (and hence Lambda).

        :param Sigma: Covariance matrices of the Gaussian densities. Dimensions should be [R, D, D].
        :type Sigma: jnp.ndarray
        :param mu: Mean of the Gaussians. Dimensions should be
        :type mu: jnp.ndarray
        :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [R, D, D], defaults to None
        :type Lambda: jnp.ndarray, optional
        :param ln_det_Sigma: Log determinant of the covariance matrix. Dimensions should be [R], defaults to None
        :type ln_det_Sigma: jnp.ndarray, optional
        """
        Lambda, ln_det_Sigma = invert_diagonal(Sigma)
        super().__init__(
            Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma,
        )

    def slice(self, indices: jnp.ndarray) -> "GaussianDiagDensity":
        """Return an object with only the specified entries.

        :param indices: The entries that should be contained in the returned object.
        :type indices: jnp.ndarray
        :return: The resulting Gaussian diagonal density.
        :rtype: GaussianDiagDensity
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        mu_new = jnp.take(self.mu, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = GaussianDiagDensity(
            Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new,
        )
        return new_measure

    def update(self, indices: jnp.ndarray, density: "GaussianDiagDensity"):
        """Update densities at indicated entries.

        :param indices: The entries that should be updated.
        :type indices: jnp.ndarray
        :param density: New densities.
        :type density: GaussianDiagDensity
        """
        self.Lambda = self.Lambda.at[indices].set(density.Lambda)
        self.Sigma = self.Sigma.at[indices].set(density.Sigma)
        self.mu = self.mu.at[indices].set(density.mu)
        self.ln_det_Sigma = self.ln_det_Sigma.at[indices].set(density.ln_det_Sigma)
        self.ln_det_Lambda = self.ln_det_Lambda.at[indices].set(density.ln_det_Lambda)
        self.lnZ = self.lnZ.at[indices].set(density.lnZ)
        self.nu = self.nu.at[indices].set(density.nu)
        self.ln_beta = self.ln_beta.at[indices].set(density.ln_beta)

    def get_marginal(self, dim_idx: jnp.ndarray) -> "GaussianDiagDensity":
        """Get the marginal of the indicated dimensions.

        :param dim_idx: The dimensions of the variables, ther marginal is required for.
        :type dim_idx: jnp.ndarray
        :return: The resulting marginal Gaussian density.
        :rtype: GaussianDiagDensity
        """
        Sigma_new = self.Sigma[:, dim_idx][:, :, dim_idx]
        mu_new = self.mu[:, dim_idx]
        marginal_density = GaussianDiagDensity(Sigma_new, mu_new)
        return marginal_density
