##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) measures.                                 #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

from autograd import numpy
from src import factors
from autograd.scipy.special import logsumexp

class GaussianMixtureMeasure:
    
    def __init__(self, components: ['GaussianMeasure'], weights: numpy.ndarray=None):
        """ Class of mixture of Gaussian measures
        
            u(x) = sum_i w_i * u_i(x)
            
            where w_i are weights and u_i the component measures.
            
        :param components: list
            List of Gaussian measures.
        :param weights: numpy.ndarray [num_components] or None
            Weights of the components. If None they are assumed to be 1. (Default=None)
        """
        self.num_components = len(components)
        if weights is None:
            self.weights = numpy.ones(self.num_components)
        else:
            self.weights = weights
        self.components = components
        self.R, self.D = self.components[0].R, self.components[0].D
        
    def slice(self, indices: list) -> 'GaussianMixtureMeasure':
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
        
        return GaussianMixtureMeasure(components_new, self.weights)
        
    def evaluate_ln(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Evaluates the log-exponential term at x.
        
        :param x: numpy.ndarray [N, D]
            Points where the factor should be evaluated.
            
        :return: numpy.ndarray [N, R]
            Log exponential term.
        """
        ln_comps = numpy.empty((self.num_components, self.R, x.shape[0]))
        
        for icomp in range(self.num_components):
            ln_comps[icomp] = self.components[icomp].evaluate_ln(x)
        ln_u, signs = logsumexp(ln_comps, b=self.weights[:,None,None], axis=0, 
                                return_sign=True)
        return ln_u, signs
    
    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Evaluates the exponential term at x.

        :param x: numpy.ndarray [N, D]
            Points where the factor should be evaluated.

        :return: numpy.ndarray [N, R]
            Exponential term.
        """
        ln_u, signs = self.evaluate_ln(x)
        return signs * numpy.exp(ln_u)
    
    def multiply(self, factor: factors.ConjugateFactor, update_full: bool=False) -> 'GaussianMeasure':
        """ Computes the product between the measure u and a conjugate factor f
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param factor: ConjugateFactor
            The conjugate factor the measure is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMixtureMeasure
            Returns the resulting GaussianMixtureMeasure.
        """
        components_new = []
        for icomp in range(self.num_components):
            comp_new = factor.multiply_with_measure(self.components[icomp], update_full=update_full)
            components_new.append(comp_new)
        return GaussianMixtureMeasure(components_new, weights=self.weights)
    
    def hadamard(self, factor: factors.ConjugateFactor, update_full: bool=False) -> 'GaussianMeasure':
        """ Computes the hadamard (componentwise) product between the measure u and a conjugate factor f
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param factor: ConjugateFactor
            The conjugate factor the measure is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMixtureMeasure
            Returns the resulting GaussianMixtureMeasure.
        """
        components_new = []
        for icomp in range(self.num_components):
            comp_new = factor.hadamard_with_measure(self.components[icomp], update_full=update_full)
            components_new.append(comp_new)
        return GaussianMixtureMeasure(components_new, weights=self.weights)
    
    def integrate(self, expr:str='1', **kwargs) -> numpy.ndarray:
        """ Integrates the indicated expression with respect to the Gaussian mixture measure.
        
        :param expr: str
            Indicates the expression that should be integrated. Check measure's integration dict. Default='1'.
        :kwargs:
            All parameters, that are required to evaluate the expression.
        """
        integration_res = self.weights[0] * self.components[0].integration_dict[expr](**kwargs)
        for icomp in range(1, self.num_components):
            integration_res += self.weights[icomp] * self.components[icomp].integration_dict[expr](**kwargs)
        return integration_res
    
    
class GaussianMeasure(factors.ConjugateFactor):
    
    def __init__(self, Lambda: numpy.ndarray, nu: numpy.ndarray=None, ln_beta: numpy.ndarray=None):
        """ A measure with a Gaussian form.
        
        u(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),
    
        D is the dimension, and R the number of Gaussians. 

        :param Lambda: numpy.ndarray [R, D, D]
            Information (precision) matrix of the Gaussian distributions. Needs to be postive definite.
        :param nu: numpy.ndarray [R, D]
            Information vector of a Gaussian distribution. If None all zeros. (Default=None)
        :param ln_beta: numpy.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        """
        
        super().__init__(Lambda, nu, ln_beta)
        self.Sigma = None
        self.ln_det_Lambda = None
        self.ln_det_Sigma = None
        self.lnZ = None
        self.mu = None
        self.integration_dict = {'1': self.integral,
                                 'x': self.integrate_x,
                                 'Ax_a': self.integrate_general_linear,
                                 'xx': self.integrate_xxT,
                                 'Ax_aBx_b_inner': self.integrate_general_quadratic_inner,
                                 'Ax_aBx_b_outer': self.integrate_general_quadratic_outer,
                                 'Ax_aBx_bCx_c_inner': self.integrate_general_cubic_inner,
                                 'Ax_aBx_bCx_c_outer': self.integrate_general_cubic_outer,
                                 'xAx_ax': self.integrate_cubic_outer, # Rename
                                 'Ax_aBx_bCx_cDx_d_inner': self.integrate_general_quartic_inner,
                                 'Ax_aBx_bCx_cDx_d_outer': self.integrate_general_quartic_outer}
        
    def slice(self, indices: list) -> 'GaussianMeasure':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianMeasure
            The resulting Gaussian measure.
        """
        Lambda_new = self.Lambda[indices]
        nu_new = self.nu[indices]
        ln_beta_new = self.ln_beta[indices]
        new_measure = GaussianMeasure(Lambda_new, nu_new, ln_beta_new)
        if self.Sigma is not None:
            new_measure.Sigma = self.Sigma[indices]
            new_measure.ln_det_Sigma = self.ln_det_Sigma[indices]
            new_measure.ln_det_Lambda = self.ln_det_Lambda[indices]
        return new_measure
        
    def _prepare_integration(self):
        if self.lnZ is None:
            self.compute_lnZ()
        if self.mu is None:
            self.compute_mu()
        
    def compute_lnZ(self):
        """ Computes the log partition function.
        """
        if self.Sigma is None:
            self.invert_lambda()
        nu_Lambda_nu = numpy.einsum('ab,ab->a', self.nu, numpy.einsum('abc,ac->ab', self.Sigma, self.nu))
        self.lnZ = .5 * (nu_Lambda_nu + self.D * numpy.log(2. * numpy.pi) + self.ln_det_Sigma)
    
    def invert_lambda(self):
        self.Sigma, self.ln_det_Lambda = self.invert_matrix(self.Lambda)
        self.ln_det_Sigma = -self.ln_det_Lambda
        
    def multiply(self, factor: factors.ConjugateFactor, update_full: bool=False) -> 'GaussianMeasure':
        """ Computes the product between the measure u and a conjugate factor f
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param factor: ConjugateFactor
            The conjugate factor the measure is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        return factor.multiply_with_measure(self, update_full=update_full)
    
    def hadamard(self, factor: factors.ConjugateFactor, update_full: bool=False) -> 'GaussianMeasure':
        """ Computes the hadamard (componentwise) product between the measure u and a conjugate factor f
        
            f(x) * u(x)
            
            and returns the resulting Gaussian measure.
            
        :param factor: ConjugateFactor
            The conjugate factor the measure is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. (Default=True)
            
        :return: GaussianMeasure
            Returns the resulting GaussianMeasure.
        """
        return factor.hadamard_with_measure(self, update_full=update_full)
    
    def integrate(self, expr:str='1', **kwargs) -> numpy.ndarray:
        """ Integrates the indicated expression with respect to the Gaussian measure.
        
        :param expr: str
            Indicates the expression that should be integrated. Check measure's integration dict. Default='1'.
        :kwargs:
            All parameters, that are required to evaluate the expression.
        """
        return self.integration_dict[expr](**kwargs)
    
    def log_integral(self) -> numpy.ndarray:
        """ Computes the log integral of the exponential term.
        
        log \int u(x) dx.
        
        :return: numpy.ndarray [R]
            Log integral
        """
        self._prepare_integration()
        return self.lnZ + self.ln_beta
    
    def integral(self) -> numpy.ndarray:
        """ Computes the log integral of the exponential term.
        
        \int u(x) dx.
        
        :return: numpy.ndarray [R]
            Integral
        """
        return numpy.exp(self.log_integral())
    
    def normalize(self):
        """ Normalizes the term such that
        
        int u(x) dx = 1.
        """
        self.compute_lnZ()
        self.ln_beta = -self.lnZ
        
    def is_normalized(self) -> numpy.ndarray:
        return numpy.equal(self.lnZ, -self.ln_beta)
    
    def compute_mu(self):
        """ Converts from information to mean vector.
        
        :return: numpy.ndarray [R, D]
            Mean vector.
        """
        if self.Sigma is None:
            self.invert_lambda()
        self.mu = numpy.einsum('abc,ac->ab', self.Sigma, self.nu)
    
    def get_density(self) -> 'GaussianDensity':
        """ Returns the corresponing normalised density object.
        
        :return: GaussianDensity
            Corresponding density object.
        """
        from src import densities
        self._prepare_integration()
        return densities.GaussianDensity(Sigma=self.Sigma, mu=self.mu, Lambda=self.Lambda, ln_det_Sigma=self.ln_det_Sigma)
            
        
    def _get_default(self, mat, vec) -> (numpy.ndarray, numpy.ndarray):
        """ Small method to get default matrix and vector.
        """
        if mat is None:
            mat = numpy.eye(self.D)
        if vec is None:
            vec = numpy.zeros(mat.shape[0])
        if mat.ndim == 2:
            mat = numpy.tile(mat[None], [1, 1, 1])
        if vec.ndim == 1:
            vec = numpy.tile(vec[None], [1, 1])
        return mat, vec
            
    ##### Linear integals
            
    def _expectation_x(self) -> numpy.ndarray:
        """ Computes the expectation.
        
            int x du(x) / int du(x)
        
        :return: numpy.ndarray [R, D]
            The solved intergal.
        """
        return self.mu 
            
    def integrate_x(self) -> numpy.ndarray:
        """ Computes the integral.
        
            int x du(x)
        
        :return: numpy.ndarray [R, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None] * self._expectation_x()
    
    def _expectation_general_linear(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the linear expectation.
        
            int (Ax+a) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
            
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        return numpy.einsum('abc,ac->ab', A_mat, self.mu) + a_vec
    
    def integrate_general_linear(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the linear expectation.
        
            int (Ax+a) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_linear(A_mat, a_vec)
        
    
    ##### Quadratic integrals
    
    def _expectation_xxT(self) -> numpy.ndarray:
        """ Computes the expectation.
        
            int xx' du(x) / int du(x)
        
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        return self.Sigma + numpy.einsum('ab,ac->acb', self.mu, self.mu)
    
    def integrate_xxT(self) -> numpy.ndarray:
        """ Computes the integral.
        
            int xx' du(x)
        
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxT()
    
    def _expectation_general_quadratic_inner(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, B_mat: numpy.ndarray, b_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)'(Bx+b) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        AB = numpy.einsum('abc,abd->acd', A_mat, B_mat)
        ABSigma_trace = self.get_trace(numpy.einsum('cab,cbd->cad', AB, self.Sigma))
        mu_AB_mu = numpy.einsum('ab,ab->a', numpy.einsum('ab, abc-> ac', self.mu, AB), self.mu)
        muAb = numpy.einsum('ab,ab->a', numpy.einsum('ab,acb->ac', self.mu, A_mat), b_vec)
        aBm_b = numpy.einsum('ab, ab->a', a_vec, self._expectation_general_linear(B_mat, b_vec))
        return ABSigma_trace + mu_AB_mu + muAb + aBm_b
    
    def integrate_general_quadratic_inner(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the quadratic expectation.
        
            int (Ax+a)'(Bx+b) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return constant * self._expectation_general_quadratic_inner(A_mat, a_vec, B_mat, b_vec)
    
    def _expectation_general_quadratic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, B_mat: numpy.ndarray, b_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: numpy.ndarray [R, K, L]
            The solved intergal.
        """
        Exx = self._expectation_xxT()
        AxxB = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', Exx, B_mat))
        Axb = numpy.einsum('ab,ac->abc', numpy.einsum('cab,cb->ca', A_mat, self.mu), b_vec)
        aBx_b = numpy.einsum('ba, bc->bac', a_vec, self._expectation_general_linear(B_mat, b_vec))
        return AxxB + Axb + aBx_b
    
    def integrate_general_quadratic_outer(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                          B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: numpy.ndarray [R,K,L]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return constant[:,None,None] * self._expectation_general_quadratic_outer(A_mat, a_vec, B_mat, b_vec)
    
    ##### Cubic integrals
    
    def _expectation_xbxx(self, b_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the cubic expectation.
        
            int xb'xx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param b_vec: numpy.ndarray [1, D] or [R, D]
            Vector of 
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        Exx = self._expectation_xxT()
        mub_outer = numpy.einsum('ab,ac->abc', self.mu, b_vec)
        mbExx = numpy.einsum('abc,acd->abd', mub_outer, Exx)
        bmu_inner = numpy.einsum('ab,ab->a', self.mu, b_vec)
        bmSigma = numpy.einsum('a,abc->abc', bmu_inner, self.Sigma)
        bmu_outer = numpy.einsum('ab,ac->abc', b_vec, self.mu)
        Sigmabm = numpy.einsum('abd,ade->abe', self.Sigma, bmu_outer)
        return mbExx + bmSigma + Sigmabm
    
    
    def _expectation_cubic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the cubic expectation.
        
            int x(A'x + a)x' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,1,D] or [R,1,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [1,1] or [R,1]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        """
        # xAxx
        xAxx = self._expectation_xbxx(b_vec=A_mat)
        axx = a_vec[:,None,None] * self._expectation_xxT()
        return xAxx + axx
    
    def integrate_cubic_outer(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the cubic integration.
        
            int x(A'x + a)x' du(x).
            
        :param A_mat: numpy.ndarray [1,D] or [R,1,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [1] or [R,1]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        if A_mat is None:
            A_mat = numpy.ones((1,self.D))
        if a_vec is None:
            a_vec = numpy.zeros(1)
        if A_mat.ndim == 2:
            A_mat = numpy.tile(A_mat[None], [1, 1, 1])
        if a_vec.ndim == 1:
            a_vec = numpy.tile(a_vec[None], [1, 1])
        constant = self.integral()
        return constant[:,None,None] * self._expectation_cubic_outer(A_mat=A_mat[:,0], a_vec=a_vec[:,0])

    def intergate_xbxx(self, b_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the cubic integral.
        
            int xb'xx' du(x)
        :param b_vec: numpy.ndarray [D,]
            Vector of 
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xbxx(b_vec)
    
    def _expectation_xAxx(self, A_mat: numpy.ndarray) -> numpy.ndarray:
        """ Computes the cubic integral.
        
            int xAx'x dphi(x)
            
        :param A_mat: numpy.ndarray [D, D]
            Vector of 
        :return: numpy.ndarray [R, D]
            The solved intergal.
        """
        xAxm = numpy.einsum('ab,bc->ac', numpy.einsum('ab,abc->ac', self.mu, self.Sigma), A_mat)
        Am = numpy.einsum('ab,cb->ca', A_mat, self.mu)
        xAmx = numpy.einsum('abc,ab->ac', self.Sigma, Am)
        Exx = self._expectation_general_quadratic_inner(numpy.eye(self.D), numpy.zeros(self.D), numpy.eye(self.D), numpy.zeros(self.D))
        mA = numpy.einsum('ab,bc->ac', self.mu, A_mat)
        mAxx = numpy.einsum('ab,a->ab', mA, Exx)
        return xAxm + xAmx + mAxx
    
    
    def _expectation_general_cubic_inner(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, 
                                         B_mat: numpy.ndarray, b_vec: numpy.ndarray,
                                         C_mat: numpy.ndarray, c_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
        :param C_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        Amu_a = numpy.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = numpy.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = numpy.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        BSigmaC = numpy.einsum('cab,cbd->cad', B_mat, numpy.einsum('abc,adc->abd', self.Sigma, C_mat))
        BmubCmuc = numpy.einsum('ab,ab->a', Bmu_b, Cmu_c)
        
        BCm_c = numpy.einsum('cab,ca->cb', B_mat, Cmu_c)
        CBm_b = numpy.einsum('cab,ca->cb', C_mat, Bmu_b)
        first_term = numpy.einsum('abc,ac->ab', numpy.einsum('cab,cbd->cad', A_mat, self.Sigma), BCm_c + CBm_b)
        second_term = Amu_a * (self.get_trace(BSigmaC) + BmubCmuc)[:,None]
        return first_term + second_term
    
    
    def integrate_general_cubic_inner(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                      B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None, 
                                      C_mat: numpy.ndarray=None, c_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the quadratic integration.
        
            int (Ax+a)(Bx+b)'(Cx+c)  du(x).
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)           
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_cubic_inner(A_mat, a_vec, B_mat, b_vec, C_mat, c_vec)
    
    def _expectation_general_cubic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray,  
                                         B_mat: numpy.ndarray, b_vec: numpy.ndarray,
                                         C_mat: numpy.ndarray, c_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the cubic expectation.
        
            int (Ax+a)'(Bx+b)(Cx+c)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param C_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: numpy.ndarray [R, L]
            The solved intergal.
            
        # REMARK: Does the same thing as inner transposed.
        """
        Amu_a = numpy.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = numpy.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = numpy.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        BSigmaC = numpy.einsum('cab,cbd->cad', B_mat, numpy.einsum('abc,adc->abd', self.Sigma, C_mat))
        ASigmaC = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', self.Sigma, C_mat))
        ASigmaB = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', self.Sigma, B_mat))
        BmubCmuc = numpy.einsum('ab,ac->abc', Bmu_b, Cmu_c)
        AmuaCmuc = numpy.einsum('ab,ac->abc', Amu_a, Cmu_c)
        AmuaBmub = numpy.einsum('ab,ab->a', Amu_a, Bmu_b)
        first_term = numpy.einsum('ab,abc->ac', Amu_a, BSigmaC + BmubCmuc)
        second_term = numpy.einsum('ab,abc->ac', Bmu_b, ASigmaC + AmuaCmuc)
        third_term = - AmuaBmub[:,None] * Cmu_c
        fourth_term = self.get_trace(ASigmaB)[:,None] * Cmu_c
        return first_term + second_term + third_term + fourth_term
    
    def integrate_general_cubic_outer(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                      B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None, 
                                      C_mat: numpy.ndarray=None, c_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the quadratic integration
        
           int (Bx+b)'(Cx+c)(Dx+d)' du(x),
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)      
        :return: numpy.ndarray [R, L]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_cubic_outer(A_mat, a_vec, B_mat, b_vec, C_mat, c_vec)
    
    ##### Quartic integrals
    
    def _expectation_xxTxxT(self) -> numpy.ndarray:
        """ Computes the cubic integral.
        
            int xx'xx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        mumu_outer = numpy.einsum('ab,ac->acb', self.mu, self.mu)
        Sigma_mumu_p = self.Sigma + mumu_outer
        Sigma_mumu_m = self.Sigma - mumu_outer
        mumu_inner = numpy.einsum('ab,ab->a', self.mu, self.mu)
        Sigma_trace = self.get_trace(self.Sigma)
        Sigma_mumu_p2 = numpy.einsum('abc,acd->abd', Sigma_mumu_p, Sigma_mumu_p)
        return 2. * Sigma_mumu_p2 + mumu_inner[:,None,None] * Sigma_mumu_m + Sigma_trace[:,None,None] * Sigma_mumu_p
        
    def integrate_xxTxxT(self) -> numpy.ndarray:
        """ Computes the quartic integral.
        
            int xx'xx' du(x).
            
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxTxxT()
    
    def _expectation_xxTAxxT(self, A_mat: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quartic expectation.
        
            int xx'Axx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A: numpy.ndarray [D,D]
            Square matrix.
            
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        mumu_outer = numpy.einsum('ab,ac->acb', self.mu, self.mu)
        Sigma_mumu_p = self.Sigma + mumu_outer
        Sigma_mumu_m = self.Sigma - mumu_outer
        AAT = A_mat + A_mat.T
        Sigma_AA_Sigma = numpy.einsum('abc, acd->abd', numpy.einsum('abc,cd->abd', Sigma_mumu_p, AAT), Sigma_mumu_p)
        muAmu = numpy.einsum('ab,ab->a', self.mu, numpy.einsum('ab,ca-> cb', A_mat, self.mu))
        ASigma_trace = self.get_trace(numpy.einsum('ab,cbd->cad', A_mat, self.Sigma))
        return Sigma_AA_Sigma + muAmu[:,None,None] * Sigma_mumu_m + ASigma_trace[:,None,None] * Sigma_mumu_p
    
    def integrate_xxTAxxT(self, A_mat: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quartic integral.
        
            int xx'xx' du(x)
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxTAxxT(A_mat)
    
    def _expectation_general_quartic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, B_mat: numpy.ndarray, b_vec: numpy.ndarray, 
                                           C_mat: numpy.ndarray, c_vec: numpy.ndarray, D_mat: numpy.ndarray, d_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
        :param C_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
        :param D_mat: numpy.ndarray [1,M,D] or [R,M,D]
            Real valued matrix.
        :param d_vec: numpy.ndarray [1,M,D] or [R,M,D]
            Real valued vector.
            
        :return: numpy.ndarray [R, K, M]
            The solved intergal.
        """
        Amu_a = numpy.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = numpy.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = numpy.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        Dmu_d = numpy.einsum('cab,cb-> ca', D_mat, self.mu) + d_vec
        ASigmaB = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', self.Sigma, B_mat))
        CSigmaD = numpy.einsum('cab,cbd->cad', C_mat, numpy.einsum('abc,adc->abd', self.Sigma, D_mat))
        ASigmaC = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', self.Sigma, C_mat))
        BSigmaD = numpy.einsum('cab,cbd->cad', B_mat, numpy.einsum('abc,adc->abd', self.Sigma, D_mat))
        ASigmaD = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', self.Sigma, D_mat))
        BSigmaC = numpy.einsum('cab,cbd->cad', B_mat, numpy.einsum('abc,adc->abd', self.Sigma, C_mat))
        AmuaBmub = numpy.einsum('ab,ac->abc', Amu_a, Bmu_b)
        CmucDmud = numpy.einsum('ab,ac->abc', Cmu_c, Dmu_d)
        AmuaCmuc = numpy.einsum('ab,ac->abc', Amu_a, Cmu_c)
        BmubDmud = numpy.einsum('ab,ac->abc', Bmu_b, Dmu_d)
        BmubCmuc = numpy.einsum('ab,ab->a', Bmu_b, Cmu_c)
        AmuaDmud = numpy.einsum('ab,ac->abc', Amu_a, Dmu_d)
        first_term = numpy.einsum('abc,acd->abd', ASigmaB + AmuaBmub, CSigmaD + CmucDmud)
        second_term = numpy.einsum('abc,acd->abd', ASigmaC + AmuaCmuc, BSigmaD + BmubDmud)
        third_term = BmubCmuc[:,None,None] * (ASigmaD - AmuaDmud)
        fourth_term = self.get_trace(BSigmaC)[:,None,None] * (ASigmaD + AmuaDmud)
        return first_term + second_term + third_term + fourth_term
        
    def integrate_general_quartic_outer(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                       B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None, 
                                       C_mat: numpy.ndarray=None, c_vec: numpy.ndarray=None, 
                                       D_mat: numpy.ndarray=None, d_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the quartic integral.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' du(x).
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param D_mat: numpy.ndarray [M,D] or [R,M,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param d_vec: numpy.ndarray [M,D] or [R,M,D]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: numpy.ndarray [R, K, M]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        D_mat, d_vec = self._get_default(D_mat, d_vec)
        constant = self.integral()
        return constant[:,None,None] * self._expectation_general_quartic_outer(A_mat,a_vec,B_mat,b_vec,C_mat,c_vec,D_mat,d_vec)
    
    def _expectation_general_quartic_inner(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, 
                                           B_mat: numpy.ndarray, b_vec: numpy.ndarray, 
                                           C_mat: numpy.ndarray, c_vec: numpy.ndarray, 
                                           D_mat: numpy.ndarray, d_vec: numpy.ndarray) -> numpy.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)'(Bx+b)(Cx+c)'(Dx+d) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: numpy.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [1,K] or [R,K]
            Real valued vector.
        :param C_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
        :param D_mat: numpy.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param d_vec: numpy.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        Amu_a = numpy.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = numpy.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = numpy.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        Dmu_d = numpy.einsum('cab,cb-> ca', D_mat, self.mu) + d_vec
        ASigmaB = numpy.einsum('cab,cbd->cad', A_mat, numpy.einsum('abc,adc->abd', self.Sigma, B_mat))
        CSigmaD = numpy.einsum('cab,cbd->cad', C_mat, numpy.einsum('abc,adc->abd', self.Sigma, D_mat))
        
        AmuaBmub = numpy.einsum('ab,ab->a', Amu_a, Bmu_b)
        CmucDmud = numpy.einsum('ab,ab->a', Cmu_c, Dmu_d)
        CD = numpy.einsum('abc,abd->acd', C_mat, D_mat)
        CD_DC = CD + numpy.swapaxes(CD, axis1=1, axis2=2)
        SCD_DCS = numpy.einsum('abc,acd->abd', numpy.einsum('abc,acd->abd', self.Sigma, CD_DC), self.Sigma)
        ASCD_DCSB = numpy.einsum('abc,adc->abd', numpy.einsum('cab,cbd->cad', A_mat, SCD_DCS), B_mat)
        Am_aB = numpy.einsum('ab,abc->ac', Amu_a, B_mat)
        Bm_bA = numpy.einsum('ab,abc->ac', Bmu_b, A_mat)
        CDm_d = numpy.einsum('cab,ca->cb', C_mat, Dmu_d)
        DCm_c = numpy.einsum('cab,ca->cb', D_mat, Cmu_c)
        first_term = self.get_trace(ASCD_DCSB)
        second_term = numpy.einsum('ab,ab->a', numpy.einsum('ab,abc->ac', Am_aB + Bm_bA, self.Sigma), CDm_d + DCm_c)
        third_term = (self.get_trace(ASigmaB) + AmuaBmub) * (self.get_trace(CSigmaD) + CmucDmud)
        return first_term + second_term + third_term
        
    def integrate_general_quartic_inner(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                       B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None, 
                                       C_mat: numpy.ndarray=None, c_vec: numpy.ndarray=None, 
                                       D_mat: numpy.ndarray=None, d_vec: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the quartic integral.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' du(x).
            
        :param A_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: numpy.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param D_mat: numpy.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param d_vec: numpy.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        D_mat, d_vec = self._get_default(D_mat, d_vec)
        constant = self.integral()
        return constant * self._expectation_general_quartic_inner(A_mat,a_vec,B_mat,b_vec,C_mat,c_vec,D_mat,d_vec)

    
class GaussianDiagMeasure(GaussianMeasure):
    
    def invert_lambda(self):
        self.Sigma = numpy.diag(1. / self.Lambda.diagonal(axis1=1,axis2=2))
        self.ln_det_Lambda = numpy.sum(numpy.log(self.Lambda.diagonal()))
        self.ln_det_Sigma = -self.ln_det_Lambda   
        
    @staticmethod
    def invert_diagonal(A: numpy.ndarray) -> (numpy.ndarray,numpy.ndarray):
        A_inv = numpy.concatenate([numpy.diag(mat)[None] for mat in  1./A.diagonal(axis1=1, axis2=2)], axis=0)
        ln_det_A = numpy.sum(numpy.log(A.diagonal(axis1=1, axis2=2)), axis=1)
        return A_inv, ln_det_A
    
    def slice(self, indices: list) -> 'GaussianDiagMeasure':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianDiagMeasure
            The resulting Gaussian diagonal measure.
        """
        Lambda_new = self.Lambda[indices]
        nu_new = self.nu[indices]
        ln_beta_new = self.ln_beta[indices]
        new_measure = GaussianDiagMeasure(Lambda_new, nu_new, ln_beta_new)
        if self.Sigma is not None:
            new_measure.Sigma = self.Sigma[indices]
            new_measure.ln_det_Sigma = self.ln_det_Sigma[indices]
            new_measure.ln_det_Lambda = self.ln_det_Lambda[indices]
        return new_measure