##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) probability densities.                    #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

import numpy
import measures
import conditionals

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
        super().__init__(Lambda=Lambda, nu=nu)
        self.Sigma = Sigma
        self.mu = mu
        self.ln_det_Sigma = ln_det_Sigma
        self.ln_det_Lambda = -ln_det_Sigma
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
        dim_xy = numpy.arange(self.D)
        dim_x = dim_xy[numpy.logical_not(numpy.isin(dim_xy, dim_y))]
        Lambda_x = self.Lambda[:, dim_x][:, :, dim_x]
        Sigma_x, ln_det_Lambda_x = self.invert_matrix(Lambda_x)
        M_x = -numpy.einsum('abc,acd->abd', Sigma_x, self.Lambda[:,dim_x][:,:,dim_y])
        b_x = self.mu[:, dim_x] - numpy.einsum('abc,ac->ab', M_x, self.mu[:, dim_y])
        return conditionals.ConditionalGaussianDensity(M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x)
    
    def affine_joint_transformation(self, cond_density: 'ConditionalGaussianDensity') -> 'GaussianDensity':
        """ Returns the joint density 
        
            p(x,y) = p(y|x)p(x),
            
            where p(x) is the object itself.
            
        :param cond_density: ConditionalGaussianDensity
            The conditional density.
        
        :return: GaussianDensity
            The joint density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of multiple marginals
        # and multiple cond
        try:
            assert self.R == 1 or cond_density.R == 1
        except AssertionError:
            raise RuntimeError('The combination of combining multiple marginals with multiple conditionals is not implemented.')
        R = self.R * cond_density.R
        D_xy = self.D + cond_density.Dy
        # Mean
        mu_x = numpy.tile(self.mu[None], (cond_density.R, 1, 1,)).reshape((R, self.D))
        mu_y = cond_density.get_conditional_mu(self.mu).reshape((R, cond_density.Dy))
        mu_xy = numpy.hstack([mu_x, mu_y])
        # Sigma
        Sigma_x = numpy.tile(self.Sigma[None], (cond_density.R, 1, 1, 1)).reshape(R, self.D, self.D)
        MSigma_x = numpy.einsum('abc,dce->adbe', cond_density.M, self.Sigma) # [R1,R,Dy,D]
        MSigmaM = numpy.einsum('abcd,aed->abce', MSigma_x, cond_density.M)
        Sigma_y = (cond_density.Sigma[:,None] + MSigmaM).reshape((R, cond_density.Dy, cond_density.Dy))
        C_xy = MSigma_x.reshape((R, cond_density.Dy, self.D))
        Sigma_xy = numpy.empty((R, D_xy, D_xy))
        Sigma_xy[:,:self.D,:self.D] = Sigma_x
        Sigma_xy[:,self.D:,self.D:] = Sigma_y
        Sigma_xy[:,self.D:,:self.D] = C_xy
        Sigma_xy[:,:self.D,self.D:] = numpy.swapaxes(C_xy, 1, 2)
        # Lambda
        Lambda_y = numpy.tile(cond_density.Lambda[:,None], (1, self.R, 1, 1)).reshape((R, cond_density.Dy, cond_density.Dy))
        Lambda_yM = numpy.einsum('abc,abd->acd', cond_density.Lambda, cond_density.M) # [R1,Dy,D]
        MLambdaM = numpy.einsum('abc,abd->acd', cond_density.M, Lambda_yM)
        Lambda_x = (self.Lambda[None] + MLambdaM[:,None]).reshape((R, self.D, self.D))
        L_xy = numpy.tile(-Lambda_yM[:,None], (1, self.R, 1, 1)).reshape((R, cond_density.Dy, self.D))
        Lambda_xy = numpy.empty((R, D_xy, D_xy))
        Lambda_xy[:,:self.D,:self.D] = Lambda_x
        Lambda_xy[:,self.D:,self.D:] = Lambda_y
        Lambda_xy[:,self.D:,:self.D] = L_xy
        Lambda_xy[:,:self.D,self.D:] = numpy.swapaxes(L_xy, 1, 2)
        # Log determinant
        if self.D > cond_density.Dy:
            CLambda_x = numpy.einsum('abcd,bde->abce', MSigma_x, self.Lambda) # [R1,R,Dy,D]
            CLambdaC = numpy.einsum('abcd,abed->abce', CLambda_x, MSigma_x) # [R1,R,Dy,Dy]
            delta_ln_det = numpy.linalg.slogdet(Sigma_y[:,None] - CLambdaC)[1].reshape((R,))
            ln_det_Sigma_xy = self.ln_det_Sigma + delta_ln_det
        else:
            Sigma_yL = numpy.einsum('abc,acd->abd', cond_density.Sigma, -Lambda_yM) # [R1,Dy,Dy] x [R1, Dy, D] = [R1, Dy, D]
            LSigmaL = numpy.einsum('abc,abd->acd', -Lambda_yM, Sigma_yL) # [R1, Dy, D] x [R1, Dy, D] = [R1, D, D]
            LSigmaL = numpy.tile(LSigmaL[:,None], (1, self.R)).reshape((R, self.D, self.D))
            delta_ln_det = numpy.linalg.slogdet(Lambda_x - LSigmaL)[1]
            ln_det_Sigma_xy = -(numpy.tile(cond_density.ln_det_Lambda[:,None], (1, self.R)).reshape((R,)) + delta_ln_det)
        return GaussianDensity(Sigma_xy, mu_xy, Lambda_xy, ln_det_Sigma_xy)
    
    def affine_marginal_transformation(self, cond_density: 'ConditionalGaussianDensity') -> 'GaussianDensity':
        """ Returns the marginal density p(y) given  p(y|x) and p(x), 
            where p(x) is the object itself.
            
        :param cond_density: ConditionalGaussianDensity
            The conditional density.
        
        :return: GaussianDensity
            The marginal density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of multiple marginals
        # and multiple cond
        try:
            assert self.R == 1 or cond_density.R == 1
        except AssertionError:
            raise RuntimeError('The combination of combining multiple marginals with multiple conditionals is not implemented.')
        R = self.R * cond_density.R
        # Mean
        mu_y = cond_density.get_conditional_mu(self.mu).reshape((R, cond_density.Dy))
        # Sigma
        MSigma_x = numpy.einsum('abc,dce->adbe', cond_density.M, self.Sigma) # [R1,R,Dy,D]
        MSigmaM = numpy.einsum('abcd,aed->abce', MSigma_x, cond_density.M)
        Sigma_y = (cond_density.Sigma[:,None] + MSigmaM).reshape((R, cond_density.Dy, cond_density.Dy))
        return GaussianDensity(Sigma_y, mu_y)
    
    def affine_conditional_transformation(self, cond_density: 'ConditionalGaussianDensity') -> 'ConditionalGaussianDensity':
        """ Returns the conditional density p(x|y), given p(y|x) and p(x),           
            where p(x) is the object itself.
            
        :param cond_density: ConditionalGaussianDensity
            The conditional density.
        
        :return: GaussianDensity
            The marginal density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of multiple marginals
        # and multiple cond
        try:
            assert self.R == 1 or cond_density.R == 1
        except AssertionError:
            raise RuntimeError('The combination of combining multiple marginals with multiple conditionals is not implemented.')
        R = self.R * cond_density.R
        # TODO: Could be flexibly made more effiecient here.
        # Marginal Sigma y
        # MSigma_x = numpy.einsum('abc,dce->adbe', cond_density.M, self.Sigma) # [R1,R,Dy,D]
        # MSigmaM = numpy.einsum('abcd,aed->abce', MSigma_x, cond_density.M)
        # Sigma_y = (cond_density.Sigma[:,None] + MSigmaM).reshape((R, cond_density.Dy, cond_density.Dy))
        # Lambda_y, ln_det_Sigma_y = self.invert_matrix(Sigma_y)
        # Lambda
        Lambda_yM = numpy.einsum('abc,abd->acd', cond_density.Lambda, cond_density.M) # [R1,Dy,D]
        MLambdaM = numpy.einsum('abc,abd->acd', cond_density.M, Lambda_yM)
        Lambda_x = (self.Lambda[None] + MLambdaM[:,None]).reshape((R, self.D, self.D))
        # Sigma
        Sigma_x, ln_det_Lambda_x = self.invert_matrix(Lambda_x)
        # M_x
        M_Lambda_y = numpy.einsum('abc,abd->acd', cond_density.M, cond_density.Lambda) # [R1, D, Dy]
        M_x = numpy.einsum('abcd,ade->abce', Sigma_x.reshape((cond_density.R, self.R, self.D, self.D)), M_Lambda_y) #[R1, R, D, Dy]
        b_x = - numpy.einsum('abcd,ad->abc', M_x, cond_density.b) # [R1, R, D, Dy] x [R1, Dy] = [R1, R, D]
        b_x += numpy.einsum('abcd,bd->abc', Sigma_x.reshape((cond_density.R, self.R, self.D, self.D)), self.nu).reshape((R, self.D))
        M_x = M_x.reshape((R, self.D, cond_density.Dy))
        
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
        nu = numpy.einsum('abc,ac->ab', Lambda, mu)
        super().__init__(Lambda=Lambda, nu=nu)
        self.Sigma = Sigma
        self.ln_det_Sigma = ln_det_Sigma
        self.ln_det_Lambda = -ln_det_Sigma
        self.mu = mu
        self.normalize()
        
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