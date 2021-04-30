import numpy
import measures


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
        
    def sample(self, num_samples: int):
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
        
    def sample(self, num_samples: int):
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
    
    def slice(self, indices: list):
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
        
    def slice(self, indices: list):
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