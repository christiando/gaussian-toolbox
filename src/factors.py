##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for the most general form of functions, that are conjugate to    # 
# Gaussian densities.                                                                            #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################
__author__ = "Christian Donner"

from autograd import numpy

class ConjugateFactor:
    
    def __init__(self, Lambda, nu: numpy.ndarray=None, ln_beta: numpy.ndarray=None):
        """ A general term, which can be multiplied with a Gaussian and the result is still a Gaussian, 
            i.e. has the functional form
        
            f(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),

            D is the dimension, and R the number of Gaussians.

            Note: At least Lambda or nu should be specified!
            
        :param Lambda: numpy.ndarray [R, D, D]
            Information (precision) matrix of the Gaussian distributions. Must be postive semidefinite.
        :param nu: numpy.ndarray [R, D]
            Information vector of a Gaussian distribution. If None all zeros. (Default=None)
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        """
        
        self.R, self.D = Lambda.shape[0], Lambda.shape[1]
        self.Lambda = Lambda
        
        if nu is None:
            self.nu = numpy.zeros((self.R, self.D))
        else:
            self.nu = nu
        if ln_beta is None:
            self.ln_beta = numpy.zeros((self.R))
        else:
            self.ln_beta = ln_beta
            
        
    def evaluate_ln(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Evaluates the log-exponential term at x.
        
        :param x: numpy.ndarray [N, D]
            Points where the factor should be evaluated.
        :param r: list
            Indices of densities that need to be evaluated. If empty, all densities are evaluated. (Default=[])
            
        :return: numpy.ndarray [N, R]
            Log exponential term.
        """
        x_Lambda_x = numpy.einsum('adc,dc->ad', numpy.einsum('abc,dc->adb', self.Lambda, x), x)
        x_nu = numpy.dot(x, self.nu.T).T
        return - .5 * x_Lambda_x + x_nu + self.ln_beta[:,None]
    
    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Evaluates the exponential term at x.
        
        :param x: numpy.ndarray [N, D]
            Points where the factor should be evaluated.
            
        :return: numpy.ndarray [N, R]
            Exponential term.
        """
        return numpy.exp(self.evaluate_ln(x))
    
    def slice(self, indices: list) -> 'ConjugateFactor':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: ConjugateFactor
            The resulting Conjugate factor.
        """
        Lambda_new = self.Lambda[indices]
        nu_new = self.nu[indices]
        ln_beta_new = self.ln_beta[indices]
        return ConjugateFactor(Lambda_new, nu_new, ln_beta_new)
    
    def multiply_with_measure(self, measure: 'GaussianMeasure', update_full: bool=False) -> 'GaussianMeasure':
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=False)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        Lambda_new = (measure.Lambda[:,None] + self.Lambda[None]).reshape((measure.R * self.R, self.D, self.D))
        nu_new = (measure.nu[:,None] + self.nu[None]).reshape((measure.R * self.R, self.D))
        ln_beta_new = (measure.ln_beta[:,None] + self.ln_beta[None]).reshape((measure.R * self.R))
        product = measures.GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            product.Sigma, product.ln_det_Lambda = self.invert_matrix(Lambda_new)
            product.ln_det_Sigma = -product.ln_det_Lambda
        return product  
    
    def hadamard_with_measure(self, measure: 'GaussianMeasure', update_full: bool=False) -> 'GaussianMeasure':
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=False)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        Lambda_new = measure.Lambda + self.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        product = measures.GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            product.Sigma, product.ln_det_Lambda = self.invert_matrix(Lambda_new)
            product.ln_det_Sigma = -product.ln_det_Lambda
        return product  
        
    @staticmethod
    def invert_matrix(A: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        L = numpy.linalg.cholesky(A)
        # TODO: Check whether we can make it mor efficienty with solve_triangular.
        #L_inv = solve_triangular(L, numpy.eye(L.shape[0]), lower=True,
        #                         check_finite=False)
        L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[1])[None])
        A_inv = numpy.einsum('acb,acd->abd', L_inv, L_inv)
        ln_det_A = 2. * numpy.sum(numpy.log(L.diagonal(axis1=-1, axis2=-2)), axis=1)
        return A_inv, ln_det_A
    
    @staticmethod
    def get_trace(A: numpy.ndarray) -> numpy.ndarray:
        return numpy.sum(A.diagonal(axis1=-1,axis2=-2), axis=1)
    
    
class LowRankFactor(ConjugateFactor):
    # TODO implement low rank updates with Woodbury inversion.
    def __init__(self, Lambda: numpy.ndarray=None, nu: numpy.ndarray=None, ln_beta: numpy.ndarray=None):
        super().__init__(Lambda, nu, ln_beta)
        
class OneRankFactor(LowRankFactor):
    
    def __init__(self, v: numpy.ndarray=None, g: numpy.ndarray=None, nu: numpy.ndarray=None, ln_beta: numpy.ndarray=None):
        """ A term, which can be multiplied with a Gaussian and the result is still a Gaussian, 
            i.e. has the functional form
        
            f(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),
            
            but Lambda is of rank 1 and has the form Lambda=g * vv'.

            D is the dimension, and R the number of Gaussians.
            
        :param v: numpy.ndarray [R, D]
            Rank one vector for the constructing the Lambda matrix.
        :param g: numpy.narray [R]
            Factor for the Lambda matrix. If None, it is assumed to be 1. (Default=None)
        :param nu: numpy.ndarray [R, D]
            Information vector of a Gaussian distribution. If None all zeros. (Default=None)
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        """
        self.R, self.D = v.shape
        self.v = v
        if g is None:
            self.g = numpy.ones(self.R)
        else:
            self.g = g
            
        Lambda = self._get_Lambda()
        super().__init__(Lambda, nu, ln_beta)
        
    def slice(self, indices: list) -> 'OneRankFactor':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: OneRankFactor
            The resulting OneRankFactor.
        """
        v_new = self.v[indices]
        g_new = self.g[indices]
        nu_new = self.nu[indices]
        ln_beta_new = self.ln_beta[indices]
        return OneRankFactor(v_new, g_new, nu_new, ln_beta_new)
        
    def _get_Lambda(self) -> numpy.ndarray:
        """ Computes the rank one matrix
        
            Lambda=g* vv'
            
        :return: numpy.ndarray [R, D, D]
            The low rank matrix.
        """
        return numpy.einsum('ab,ac->abc', self.v, self.g[:,None] * self.v)
    
    def multiply_with_measure(self, measure: 'GaussianMeasure', update_full=True) -> 'GaussianMeasure':
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. In contrast to full rank updates, the updated covariances and log determinants can be computed efficiently.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        Lambda_new = (measure.Lambda[:,None] + self.Lambda[None]).reshape((measure.R * self.R, self.D, self.D))
        nu_new = (measure.nu[:,None] + self.nu[None]).reshape((measure.R * self.R, self.D))
        ln_beta_new = (measure.ln_beta[:,None] + self.ln_beta[None]).reshape((measure.R * self.R))
        product = measures.GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            if measure.Sigma is None:
                product.Sigma, product.ln_det_Lambda = self.invert_matrix(Lambda_new)
                product.ln_det_Sigma = -product.ln_det_Lambda
            else:
                # Sherman morrison inversion
                Sigma_v = numpy.einsum('abc,dc->adb', measure.Sigma, self.v)
                v_Sigma_v = numpy.einsum('abc,bc->ab', Sigma_v, self.v)
                denominator = 1. + self.g[None] * v_Sigma_v
                nominator = self.g[None,:,None,None] * numpy.einsum('abc,abd->abcd', Sigma_v, Sigma_v)
                Sigma_new = measure.Sigma[:, None] - nominator / denominator[:,:,None,None]
                product.Sigma = Sigma_new.reshape((measure.R*self.R, self.D, self.D))
                # Matrix determinant lemma
                ln_det_Sigma_new = measure.ln_det_Sigma[:,None] - numpy.log(denominator)
                product.ln_det_Sigma = ln_det_Sigma_new.reshape((measure.R * self.R))
                product.ln_det_Lambda = -product.ln_det_Sigma
        return product
    
    def hadamard_with_measure(self, measure: 'GaussianMeasure', update_full=True) -> 'GaussianMeasure':
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. In contrast to full rank updates, the updated covariances and log determinants can be computed efficiently.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        Lambda_new = measure.Lambda + self.Lambda
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        product = measures.GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            if measure.Sigma is None:
                product.Sigma, product.ln_det_Lambda = self.invert_matrix(Lambda_new)
                product.ln_det_Sigma = -product.ln_det_Lambda
            else:
                # Sherman morrison inversion
                Sigma_v = numpy.einsum('abc,ac->ab', measure.Sigma, self.v)
                v_Sigma_v = numpy.einsum('ab,ab->a', Sigma_v, self.v)
                denominator = 1. + self.g * v_Sigma_v
                nominator = self.g[:,None,None] * numpy.einsum('ab,ac->abc', Sigma_v, Sigma_v)
                Sigma_new = measure.Sigma - nominator / denominator[:,None,None]
                product.Sigma = Sigma_new
                # Matrix determinant lemma
                product.ln_det_Sigma = measure.ln_det_Sigma - numpy.log(denominator)
                product.ln_det_Lambda = -product.ln_det_Sigma
        return product
    
class LinearFactor(ConjugateFactor):
    
    def __init__(self, nu: numpy.ndarray, ln_beta: numpy.ndarray=None):
        """ A term, which can be multiplied with a Gaussian and the result is still a Gaussian and it has the form
            i.e. has the functional form
        
            f(x) = beta * exp(x'nu),

            D is the dimension, and R the number of Gaussians.

            Note: At least Lambda or nu should be specified!
            
        :param nu: numpy.ndarray [R, D]
            Information vector of a Gaussian distribution.
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        """

        self.R, self.D = nu.shape[0], nu.shape[1]
        self.nu = nu
        self.Lambda = numpy.zeros((self.R, self.D))
        if ln_beta is None:
            self.ln_beta = numpy.zeros((self.R))
        else:
            self.ln_beta = ln_beta
            
    def slice(self, indices: list) -> 'LinearFactor':
        nu_new = self.nu[indices]
        ln_beta_new = self.ln_beta[indices]
        return LinearFactor(nu_new, ln_beta_new)
            
    def multiply_with_measure(self, measure: 'GaussianMeasure', update_full=True) -> 'GaussianMeasure':
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        Lambda_new = numpy.tile(measure.Lambda[:,None], (1, self.R, 1, 1)).reshape(measure.R * self.R, self.D, self.D)
        nu_new = (measure.nu[:,None] + self.nu[None]).reshape((measure.R * self.R, self.D))
        ln_beta_new = (measure.ln_beta[:,None] + self.ln_beta[None]).reshape((measure.R * self.R))
        product = measures.GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            if measure.Sigma is None:
                product.Sigma, product.ln_det_Lambda = self.invert_matrix(Lambda_new)
                product.ln_det_Sigma = -product.ln_det_Lambda
            else:
                product.Sigma = numpy.tile(measure.Sigma[:,None], (1, self.R, 1, 1)).reshape(measure.R * self.R, self.D, self.D)
                product.ln_det_Sigma = numpy.tile(measure.ln_det_Sigma[:,None], (1, self.R)).reshape(measure.R * self.R)
                product.ln_det_Lambda = -product.ln_det_Sigma
        return product
    
    def hadamard_with_measure(self, measure: 'GaussianMeasure', update_full=True) -> 'GaussianMeasure':
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        nu_new = measure.nu + self.nu
        ln_beta_new = measure.ln_beta + self.ln_beta
        product = measures.GaussianMeasure(Lambda=measure.Lambda, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            if measure.Sigma is None:
                product.Sigma, product.ln_det_Lambda = self.invert_matrix(measure.Lambda)
                product.ln_det_Sigma = -product.ln_det_Lambda
            else:
                product.Sigma = measure.Sigma
                product.ln_det_Sigma = measure.ln_det_Sigma
                product.ln_det_Lambda = -product.ln_det_Sigma
        return product
    
    
class ConstantFactor(ConjugateFactor):
    
    def __init__(self, ln_beta: numpy.ndarray, D: int):
        """ A term, which can be multiplied with a Gaussian and the result is still a Gaussian and it has the form
            i.e. has the functional form
        
            f(x) = beta,

            D is the dimension, and R the number of Gaussians.
            
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor.
        :param D: int
            The dimension of the Gaussian.
        """

        self.R, self.D = ln_beta.shape[0], D
        Lambda = numpy.zeros((self.R, self.D, self.D))
        nu = numpy.zeros((self.R, self.D))
        ln_beta = ln_beta
        super().__init__(Lambda, nu, ln_beta)
            
    def slice(self, indices: list) -> 'ConstantFactor':
        ln_beta_new = self.ln_beta[indices]
        return ConstantFactor(ln_beta_new, self.D)
    
    def multiply_with_measure(self, measure: 'GaussianMeasure', update_full=True) -> 'GaussianMeasure':
        """ Coumputes the product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        Lambda_new = numpy.tile(measure.Lambda[:,None], (1, self.R, 1, 1)).reshape(measure.R * self.R, self.D, self.D)
        nu_new = numpy.tile(measure.nu[:,None], (1, self.R, 1)).reshape((measure.R * self.R, self.D))
        ln_beta_new = (measure.ln_beta[:,None] + self.ln_beta[None]).reshape((measure.R * self.R))
        product = measures.GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if update_full:
            if measure.Sigma is None:
                product.Sigma, product.ln_det_Lambda = self.invert_matrix(Lambda_new)
                product.ln_det_Sigma = -product.ln_det_Lambda
            else:
                product.Sigma = numpy.tile(measure.Sigma[:,None], (1, self.R, 1, 1)).reshape(measure.R * self.R, self.D, self.D)
                product.ln_det_Sigma = numpy.tile(measure.ln_det_Sigma[:,None], (1, self.R)).reshape(measure.R * self.R)
                product.ln_det_Lambda = -product.ln_det_Sigma
        return product
    
    def hadamard_with_measure(self, measure: 'GaussianMeasure', update_full=True) -> 'GaussianMeasure':
        """ Coumputes the hadamard (componentwise) product between the current factor and a Gaussian measure u
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure. For the linear term, we do not need to update the covariances.
            
        :param u: GaussianMeasure
            The gaussian measure the factor is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        from src import measures
        ln_beta_new = measure.ln_beta + self.ln_beta
        product = measures.GaussianMeasure(Lambda=measure.Lambda, nu=measure.nu, ln_beta=ln_beta_new)
        if update_full:
            if measure.Sigma is None:
                product.Sigma, product.ln_det_Lambda = self.invert_matrix(measure.Sigma)
                product.ln_det_Sigma = -product.ln_det_Lambda
            else:
                product.Sigma = measure.Sigma
                product.ln_det_Sigma = measure.ln_det_Sigma
                product.ln_det_Lambda = -product.ln_det_Sigma
        return product