import numpy
import measures

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
        L = numpy.linalg.cholesky(self.Sigma)
        rand_nums = numpy.random.randn(num_samples, self.R, self.D)
        x_samples = self.mu[None] + numpy.einsum('abc,dac->dab', L, rand_nums)
        return x_samples
    
    def slice(self, indices: list):
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
        Lambda_new = self.Lambda[indices]
        Sigma_new = self.Sigma[indices]
        mu_new = self.mu[indices]
        ln_det_Sigma_new = self.ln_det_Sigma[indices]
        new_measure = GaussianDiagDensity(Sigma_new, mu_new, Lambda_new, ln_det_Sigma_new)
        return new_measure