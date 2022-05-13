##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) probability densities.                    #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

from jax import numpy as jnp
from jax import lax, random
import numpy as np
from typing import Iterable

# from .
from src_jax import measures
from utils.linalg import invert_matrix, invert_diagonal


class GaussianMixtureDensity(measures.GaussianMixtureMeasure):
    def __init__(
        self, components: Iterable["GaussianDensity"], weights: jnp.ndarray = None
    ):
        """ Class of mixture of Gaussian measures
        
            u(x) = sum_i w_i * u_i(x)
            
            where w_i are weights and u_i the component measures.
            
        :param components: list
            List of Gaussian densities.
        :param weights: jnp.ndarray [num_components] or None
            Weights of the components, that must be positive. If None they are assumed to be 1/num_components. 
            (Default=None)
        """
        super().__init__(components, weights)
        self.normalize()

    def normalize(self):
        """ Normalizes the mixture (assuming, that its components are already normalized).
        """
        self.weights /= jnp.sum(self.weights)

    def sample(self, num_samples: int) -> jnp.ndarray:
        """ Generates samples from the Gaussian mixture density.
        
        :param num_samples: int
            Number of samples that are generated.
        
        :return: jnp.ndarray [num_samples, R, D]
            The samples.
        """
        cum_weights = jnp.cumsum(self.weights)
        rand_nums = jnp.random.rand(num_samples)
        comp_samples = jnp.searchsorted(cum_weights, rand_nums)
        samples = jnp.empty((num_samples, self.R, self.D))
        for icomp in range(self.num_components):
            comp_idx = jnp.where(comp_samples == icomp)[0]
            samples[comp_idx] = self.components[icomp].sample(len(comp_idx))
        return samples

    def slice(self, indices: list) -> "GaussianMixtureDensity":
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianMixtureMeasure
            The resulting Gaussian mixture measure.
        """
        components_new = []
        for icomp in range(self.num_components):
            comp_sliced = self.components[icomp].slice(indices)
            components_new.append(comp_sliced)

        return GaussianMixtureDensity(components_new, self.weights)


class GaussianDensity(measures.GaussianMeasure):
    def __init__(
        self,
        Sigma: jnp.ndarray,
        mu: jnp.ndarray,
        Lambda: jnp.ndarray = None,
        ln_det_Sigma: jnp.ndarray = None,
    ):
        """ A normalized Gaussian density, with specified mean and covariance matrix.
        
        :param Sigma: jnp.ndarray [R, D, D]
            Covariance matrices of the Gaussian densities.
        :param mu: jnp.ndarray [R, D]
            Mean of the Gaussians.
        :param Lambda: jnp.ndarray [R, D, D] or None
            Information (precision) matrix of the Gaussians. (Default=None)
        :param ln_det_Sigma: jnp.ndarray [R] or None
            Log determinant of the covariance matrix. (Default=None)
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
        """ Generates samples from the Gaussian density.
        
        :param num_samples: int
            Number of samples that are generated.
        
        :return: jnp.ndarray [num_samples, R, D]
            The samples.
        """
        rand_nums = np.random.randn(num_samples, self.R, self.D)
        L = jnp.linalg.cholesky(self.Sigma)
        x_samples = self.mu[None] + jnp.einsum("abc,dac->dab", L, rand_nums)
        return x_samples

    def slice(self, indices: jnp.array) -> "GaussianDensity":
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianDensity
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
        new_measure = GaussianDensity(Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new)
        return new_measure

    def update(self, indices: jnp.array, density: "GaussianDensity"):
        """ Updates densities at indicated entries.
        
        :param indices: list
            The entries that should be updated.
        :param density: GaussianDensity
            New densities.
        """
        self.Lambda = self.Lambda.at[indices].set(density.Lambda)
        self.Sigma = self.Sigma.at[indices].set(density.Sigma)
        self.mu = self.mu.at[indices].set(density.mu)
        self.ln_det_Sigma = self.ln_det_Sigma.at[indices].set(density.ln_det_Sigma)
        self.lnZ = self.lnZ.at[indices].set(density.lnZ)
        self.nu = self.nu.at[indices].set(density.nu)
        self.ln_beta = self.ln_beta.at[indices].set(density.ln_beta)
        # self.Lambda = lax.dynamic_update_index_in_dim(self.Lambda, density.Lambda, indices, 0)
        # self.Sigma = lax.dynamic_update_index_in_dim(self.Sigma, density.Sigma, indices, 0)
        # self.mu = lax.dynamic_update_index_in_dim(self.mu, density.mu, indices, 0)
        # self.ln_det_Sigma = lax.dynamic_update_index_in_dim(self.ln_det_Sigma, density.ln_det_Sigma, indices, 0)
        # self.lnZ = lax.dynamic_update_index_in_dim(self.lnZ, density.lnZ, indices, 0)
        # self.nu = lax.dynamic_update_index_in_dim(self.nu, density.nu, indices, 0)
        # self.ln_beta = lax.dynamic_update_index_in_dim(self.ln_beta, density.ln_beta, indices, 0)

    def get_marginal(self, dim_x: list) -> "GaussianDensity":
        """ Gets the marginal of the indicated dimensions.
        
        :param dim_x: list
            The dimensions of the variables, ther marginal is required for.
            
        :return: GaussianDensity
            The resulting marginal Gaussian density.
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
        """ Computes the Kulback Leibler divergence between two multivariate Gaussians.
        
        D_KL(p|p1) = \int p(x)\log p(x)/p1(x) dx

        :param p1: _description_
        :type p1: GaussianDensity
        :return: _description_
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

    def condition_on(self, dim_y: list) -> "ConditionalGaussianDensity":
        """ Returns density conditioned on indicated dimensions, i.e. p(x|y).
        
        :param dim_y: list
            The dimensions of the variables, that should be conditioned on.
        
        :return: ConditionalGaussianDensity
            The corresponding conditional Gaussian density p(x|y).
        """
        from src_jax import conditionals

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
        self, dim_y: list, dim_x: list
    ) -> "ConditionalGaussianDensity":
        """ Returns density conditioned on indicated dimensions, i.e. p(x|y).
        
        :param dim_y: list
            The dimensions of the variables, that should be conditioned on.
        
        :return: ConditionalGaussianDensity
            The corresponding conditional Gaussian density p(x|y).
        """
        from src_jax import conditionals

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
        """ A normalized Gaussian density, with specified mean and covariance matrix.
        
        :param Sigma: jnp.ndarray [R, D, D]
            Covariance matrices of the Gaussian densities.
        :param mu: jnp.ndarray [R, D]
            Mean of the Gaussians.
        :param Lambda: jnp.ndarray [R, D, D] or None
            Information (precision) matrix of the Gaussians. (Default=None)
        :param ln_det_Sigma: jnp.ndarray [R] or None
            Log determinant of the covariance matrix. (Default=None)
        """
        Lambda, ln_det_Sigma = invert_diagonal(Sigma)
        super().__init__(
            Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma,
        )

    def slice(self, indices: list) -> "GaussianDiagDensity":
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianDiagDensity
            The resulting Gaussian diagonal density.
        """
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        mu_new = jnp.take(self.mu, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = GaussianDiagDensity(
            Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new,
        )
        return new_measure

    def update(self, indices: list, density: "GaussianDiagDensity"):
        """ Updates densities at indicated entries.
        
        :param indices: list
            The entries that should be updated.
        :param density: GaussianDiagDensity
            New densities.
        """
        self.Lambda = self.Lambda.at[indices].add(density.Lambda)
        self.Sigma = self.Sigma.at[indices].add(density.Sigma)
        self.mu = self.mu.at[indices].add(density.mu)
        self.ln_det_Sigma = self.ln_det_Sigma.at[indices].add(density.ln_det_Sigma)
        self.lnZ = self.lnZ.at[indices].add(density.lnZ)
        self.nu = self.nu.at[indices].add(density.nu)
        self.ln_beta = self.ln_beta.at[indices].add(density.ln_beta)

    def get_marginal(self, dim_idx: list) -> "GaussianDiagDensity":
        """ Gets the marginal of the indicated dimensions.
        
        :param dim_idx: list
            The dimensions of the variables, ther marginal is required for.
            
        :return: GaussianDiagDensity
            The resulting marginal Gaussian density.
        """
        Sigma_new = self.Sigma[:, dim_idx][:, :, dim_idx]
        mu_new = self.mu[:, dim_idx]
        marginal_density = GaussianDiagDensity(Sigma_new, mu_new)
        return marginal_density
