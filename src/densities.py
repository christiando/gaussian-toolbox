##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) probability densities.                    #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
import sys
sys.path.append('../')

from autograd import numpy
from . import measures

class GaussianMixtureDensity(measures.GaussianMixtureMeasure):
    
    def __init__(self, components: ['GaussianDensities'], weights: numpy.ndarray=None):
        """ Class of mixture of Gaussian measures
        
            u(x) = sum_i w_i * u_i(x)
            
            where w_i are weights and u_i the component measures.
            
        :param components: list
            List of Gaussian densities.
        :param weights: numpy.ndarray [num_components] or None
            Weights of the components, that must be positive. If None they are assumed to be 1/num_components. (Default=None)
        """
        super().__init__(components, weights)
        self.normalize()
        
    def normalize(self):
        """ Normalizes the mixture (assuming, that its components are already normalized).
        """
        self.weights /= numpy.sum(self.weights)
        
    def sample(self, num_samples: int) -> numpy.ndarray:
        """ Generates samples from the Gaussian mixture density.
        
        :param num_samples: int
            Number of samples that are generated.
        
        :return: numpy.ndarray [num_samples, R, D]
            The samples.
        """
        cum_weights = numpy.cumsum(self.weights)
        rand_nums = numpy.random.rand(num_samples)
        comp_samples = numpy.searchsorted(cum_weights, rand_nums)
        samples = numpy.empty((num_samples, self.R, self.D))
        for icomp in range(self.num_components):
            comp_idx = numpy.where(comp_samples==icomp)[0]
            samples[comp_idx] = self.components[icomp].sample(len(comp_idx))
        return samples
    
    def slice(self, indices: list) -> 'GaussianMixtureDensity':
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
    
    def __init__(self, Sigma: numpy.ndarray, mu: numpy.ndarray, Lambda: numpy.ndarray=None, ln_det_Sigma: numpy.ndarray=None):
        """ A normalized Gaussian density, with specified mean and covariance matrix.
        
        :param Sigma: numpy.ndarray [R, D, D]
            Covariance matrices of the Gaussian densities.
        :param mu: numpy.ndarray [R, D]
            Mean of the Gaussians.
        :param Lambda: numpy.ndarray [R, D, D] or None
            Information (precision) matrix of the Gaussians. (Default=None)
        :param ln_det_Sigma: numpy.ndarray [R] or None
            Log determinant of the covariance matrix. (Default=None)
        """
        if Lambda is None:
            Lambda, ln_det_Sigma = self.invert_matrix(Sigma)
        elif ln_det_Sigma is None:
            ln_det_Sigma = numpy.linalg.slogdet(Sigma)[1]
        nu = numpy.einsum('abc,ab->ac', Lambda, mu)
        super().__init__(Lambda=Lambda, nu=nu, Sigma=Sigma, ln_det_Lambda=-ln_det_Sigma, ln_det_Sigma=ln_det_Sigma)
        self.mu = mu
        self._prepare_integration()
        self.normalize()
        
    def sample(self, num_samples: int) -> numpy.ndarray:
        """ Generates samples from the Gaussian density.
        
        :param num_samples: int
            Number of samples that are generated.
        
        :return: numpy.ndarray [num_samples, R, D]
            The samples.
        """
        L = numpy.linalg.cholesky(self.Sigma)
        rand_nums = numpy.random.randn(num_samples, self.R, self.D)
        x_samples = self.mu[None] + numpy.einsum('abc,dac->dab', L, rand_nums)
        return x_samples
    
    def slice(self, indices: list) -> 'GaussianDensity':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianDensity
            The resulting Gaussian density.
        """
        Lambda_new = self.Lambda[indices]
        Sigma_new = self.Sigma[indices]
        mu_new = self.mu[indices]
        ln_det_Sigma_new = self.ln_det_Sigma[indices]
        new_measure = GaussianDensity(Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new)
        return new_measure
    
    def update(self, indices: list, density: 'GaussianDensity'):
        """ Updates densities at indicated entries.
        
        :param indices: list
            The entries that should be updated.
        :param density: GaussianDensity
            New densities.
        """
        self.Lambda[indices] = density.Lambda
        self.Sigma[indices] = density.Sigma
        self.mu[indices] = density.mu
        self.ln_det_Sigma[indices] = density.ln_det_Sigma
        self.lnZ[indices] = density.lnZ
        self.nu[indices] = density.nu
        self.ln_beta[indices] = density.ln_beta
    
    def get_marginal(self, dim_x: list) -> 'GaussianDensity':
        """ Gets the marginal of the indicated dimensions.
        
        :param dim_x: list
            The dimensions of the variables, ther marginal is required for.
            
        :return: GaussianDensity
            The resulting marginal Gaussian density.
        """
        Sigma_new = self.Sigma[:,dim_x][:,:,dim_x]
        mu_new = self.mu[:,dim_x]
        marginal_density = GaussianDensity(Sigma_new, mu_new)
        return marginal_density
    
    def condition_on(self, dim_y: list) -> 'ConditionalGaussianDensity':
        """ Returns density conditioned on indicated dimensions, i.e. p(x|y).
        
        :param dim_y: list
            The dimensions of the variables, that should be conditioned on.
        
        :return: ConditionalGaussianDensity
            The corresponding conditional Gaussian density p(x|y).
        """
        from . import conditionals
        dim_xy = numpy.arange(self.D)
        dim_x = dim_xy[numpy.logical_not(numpy.isin(dim_xy, dim_y))]
        Lambda_x = self.Lambda[:, dim_x][:, :, dim_x]
        Sigma_x, ln_det_Lambda_x = self.invert_matrix(Lambda_x)
        M_x = -numpy.einsum('abc,acd->abd', Sigma_x, self.Lambda[:,dim_x][:,:,dim_y])
        b_x = self.mu[:, dim_x] - numpy.einsum('abc,ac->ab', M_x, self.mu[:, dim_y])
        return conditionals.ConditionalGaussianDensity(M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x)

class GaussianDiagDensity(GaussianDensity, measures.GaussianDiagMeasure):
    
    def __init__(self, Sigma: numpy.ndarray, mu: numpy.ndarray, Lambda: numpy.ndarray=None, ln_det_Sigma: numpy.ndarray=None):
        """ A normalized Gaussian density, with specified mean and covariance matrix.
        
        :param Sigma: numpy.ndarray [R, D, D]
            Covariance matrices of the Gaussian densities.
        :param mu: numpy.ndarray [R, D]
            Mean of the Gaussians.
        :param Lambda: numpy.ndarray [R, D, D] or None
            Information (precision) matrix of the Gaussians. (Default=None)
        :param ln_det_Sigma: numpy.ndarray [R] or None
            Log determinant of the covariance matrix. (Default=None)
        """
        Lambda, ln_det_Sigma = self.invert_diagonal(Sigma)
        super().__init__(Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma)
        
    def slice(self, indices: list) -> 'GaussianDiagDensity':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianDiagDensity
            The resulting Gaussian diagonal density.
        """
        Lambda_new = self.Lambda[indices]
        Sigma_new = self.Sigma[indices]
        mu_new = self.mu[indices]
        ln_det_Sigma_new = self.ln_det_Sigma[indices]
        new_measure = GaussianDiagDensity(Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new)
        return new_measure
    
    def update(self, indices: list, density: 'GaussianDiagDensity'):
        """ Updates densities at indicated entries.
        
        :param indices: list
            The entries that should be updated.
        :param density: GaussianDiagDensity
            New densities.
        """
        self.Lambda[indices] = density.Lambda
        self.Sigma[indices] = density.Sigma
        self.mu[indices] = density.mu
        self.ln_det_Sigma[indices] = density.ln_det_Sigma
        self.lnZ[indices] = density.lnZ
        self.nu[indices] = density.nu
        self.ln_beta[indices] = density.ln_beta
    
    def get_marginal(self, dim_idx: list) -> 'GaussianDiagDensity':
        """ Gets the marginal of the indicated dimensions.
        
        :param dim_idx: list
            The dimensions of the variables, ther marginal is required for.
            
        :return: GaussianDiagDensity
            The resulting marginal Gaussian density.
        """
        Sigma_new = self.Sigma[:,dim_idx][:,:,dim_idx]
        mu_new = self.mu[:,dim_idx]
        marginal_density = GaussianDiagDensity(Sigma_new, mu_new)
        return marginal_density