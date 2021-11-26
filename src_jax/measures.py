##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for Gaussian (mixture) measures.                                 #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

from jax import numpy as jnp
from . import factors
from jax.scipy.special import logsumexp

class GaussianMixtureMeasure:
    
    def __init__(self, components: ['GaussianMeasure'], weights: jnp.ndarray=None):
        """ Class of mixture of Gaussian measures
        
            u(x) = sum_i w_i * u_i(x)
            
            where w_i are weights and u_i the component measures.
            
        :param components: list
            List of Gaussian measures.
        :param weights: jnp.ndarray [num_components] or None
            Weights of the components. If None they are assumed to be 1. (Default=None)
        """
        self.num_components = len(components)
        if weights is None:
            self.weights = jnp.ones(self.num_components)
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
        
    def evaluate_ln(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the log-exponential term at x.
        
        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.
            
        :return: jnp.ndarray [N, R]
            Log exponential term.
        """
        ln_comps = jnp.empty((self.num_components, self.R, x.shape[0]))
        
        for icomp in range(self.num_components):
            ln_comps[icomp] = self.components[icomp].evaluate_ln(x)
        ln_u, signs = logsumexp(ln_comps, b=self.weights[:,None,None], axis=0, 
                                return_sign=True)
        return ln_u, signs
    
    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the exponential term at x.

        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.

        :return: jnp.ndarray [N, R]
            Exponential term.
        """
        ln_u, signs = self.evaluate_ln(x)
        return signs * jnp.exp(ln_u)
    
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
    
    def integrate(self, expr:str='1', **kwargs) -> jnp.ndarray:
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
    
    def __init__(self, Lambda: jnp.ndarray, nu: jnp.ndarray=None, ln_beta: jnp.ndarray=None,
                 Sigma: jnp.ndarray=None, ln_det_Lambda: jnp.ndarray=None, ln_det_Sigma: jnp.ndarray=None):
        """ A measure with a Gaussian form.
        
        u(x) = beta * exp(- 0.5 * x'Lambda x + x'nu),
    
        D is the dimension, and R the number of Gaussians. 

        :param Lambda: jnp.ndarray [R, D, D]
            Information (precision) matrix of the Gaussian distributions. Needs to be postive definite.
        :param nu: jnp.ndarray [R, D]
            Information vector of a Gaussian distribution. If None all zeros. (Default=None)
        :param ln_beta: jnp.ndarray [R]
            The log constant factor of the factor. If None all zeros. (Default=None)
        param Sigma: jnp.ndarray [R, D, D]
            Covariance matrix of the Gaussian distributions. Needs to be positive definite. (Default=None)
        :param ln_det_Lambda: jnp.ndarray [R]
            Log determinant of Lambda. (Default=None)
        :param ln_det_Sigma: jnp.ndarray [R]
            Log determinant of Sigma. (Default=None)
        """
        
        super().__init__(Lambda, nu, ln_beta)
        self.Sigma = Sigma
        self.ln_det_Lambda = ln_det_Lambda
        self.ln_det_Sigma = ln_det_Sigma
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
        nu_Lambda_nu = jnp.einsum('ab,ab->a', self.nu, jnp.einsum('abc,ac->ab', self.Sigma, self.nu))
        self.lnZ = .5 * (nu_Lambda_nu + self.D * jnp.log(2. * jnp.pi) + self.ln_det_Sigma)
    
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
        new_measure_dict = factor._multiply_with_measure(self, update_full=update_full)
        return GaussianMeasure(**new_measure_dict)
    
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
        new_measure_dict = factor._hadamard_with_measure(self, update_full=update_full)
        return GaussianMeasure(**new_measure_dict)

    def integrate(self, expr:str='1', **kwargs) -> jnp.ndarray:
        """ Integrates the indicated expression with respect to the Gaussian measure.
        
        :param expr: str
            Indicates the expression that should be integrated. Check measure's integration dict. Default='1'.
        :kwargs:
            All parameters, that are required to evaluate the expression.
        """
        return self.integration_dict[expr](**kwargs)

    def log_integral_light(self) -> jnp.ndarray:
        """ Computes the log integral of the exponential term.

        log \int u(x) dx.

        :return: jnp.ndarray [R]
            Log integral
        """
        if self.lnZ is None:
            self.compute_lnZ()
        return self.lnZ + self.ln_beta
    
    def log_integral(self) -> jnp.ndarray:
        """ Computes the log integral of the exponential term.
        
        log \int u(x) dx.
        
        :return: jnp.ndarray [R]
            Log integral
        """
        self._prepare_integration()
        return self.lnZ + self.ln_beta

    def integral_light(self) -> jnp.ndarray:
        """ Computes the log integral of the exponential term.

        \int u(x) dx.

        :return: jnp.ndarray [R]
            Integral
        """
        return jnp.exp(self.log_integral_light())
    
    def integral(self) -> jnp.ndarray:
        """ Computes the log integral of the exponential term.
        
        \int u(x) dx.
        
        :return: jnp.ndarray [R]
            Integral
        """
        return jnp.exp(self.log_integral())
    
    def normalize(self):
        """ Normalizes the term such that
        
        int u(x) dx = 1.
        """
        self.compute_lnZ()
        self.ln_beta = -self.lnZ
        
    def is_normalized(self) -> jnp.ndarray:
        return jnp.equal(self.lnZ, -self.ln_beta)
    
    def compute_mu(self):
        """ Converts from information to mean vector.
        
        :return: jnp.ndarray [R, D]
            Mean vector.
        """
        if self.Sigma is None:
            self.invert_lambda()
        self.mu = jnp.einsum('abc,ac->ab', self.Sigma, self.nu)
    
    def get_density(self) -> 'GaussianDensity':
        """ Returns the corresponing normalised density object.
        
        :return: GaussianDensity
            Corresponding density object.
        """
        from src_jax import densities
        self._prepare_integration()
        return densities.GaussianDensity(Sigma=self.Sigma, mu=self.mu, Lambda=self.Lambda, ln_det_Sigma=self.ln_det_Sigma)
            
        
    def _get_default(self, mat, vec) -> (jnp.ndarray, jnp.ndarray):
        """ Small method to get default matrix and vector.
        """
        if mat is None:
            mat = jnp.eye(self.D)
        if vec is None:
            vec = jnp.zeros(mat.shape[0])
        if mat.ndim == 2:
            mat = jnp.tile(mat[None], [1, 1, 1])
        if vec.ndim == 1:
            vec = jnp.tile(vec[None], [1, 1])
        return mat, vec
            
    ##### Linear integals
            
    def _expectation_x(self) -> jnp.ndarray:
        """ Computes the expectation.
        
            int x du(x) / int du(x)
        
        :return: jnp.ndarray [R, D]
            The solved intergal.
        """
        return self.mu 
            
    def integrate_x(self) -> jnp.ndarray:
        """ Computes the integral.
        
            int x du(x)
        
        :return: jnp.ndarray [R, D]
            The solved intergal.
        """
        constant = self.integral()
        return jnp.einsum('a,ab->ab', constant, self._expectation_x())
    
    def _expectation_general_linear(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the linear expectation.
        
            int (Ax+a) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
            
        :return: jnp.ndarray [R, K]
            The solved intergal.
        """
        return jnp.einsum('abc,ac->ab', A_mat, self.mu) + a_vec
    
    def integrate_general_linear(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the linear expectation.
        
            int (Ax+a) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: jnp.ndarray [R, K]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        constant = self.integral()
        return jnp.einsum('a,ab->ab',constant, self._expectation_general_linear(A_mat, a_vec))
        
    
    ##### Quadratic integrals
    
    def _expectation_xxT(self) -> jnp.ndarray:
        """ Computes the expectation.
        
            int xx' du(x) / int du(x)
        
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        return self.Sigma + jnp.einsum('ab,ac->acb', self.mu, self.mu)
    
    def integrate_xxT(self) -> jnp.ndarray:
        """ Computes the integral.
        
            int xx' du(x)
        
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return jnp.einsum('a,abc->abc', constant, self._expectation_xxT())
    
    def _expectation_general_quadratic_inner(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray, B_mat: jnp.ndarray, b_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)'(Bx+b) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param b_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
            
        :return: jnp.ndarray [R]
            The solved intergal.
        """
        AB = jnp.einsum('abc,abd->acd', A_mat, B_mat)
        ABSigma_trace = self.get_trace(jnp.einsum('cab,cbd->cad', AB, self.Sigma))
        mu_AB_mu = jnp.einsum('ab,ab->a', jnp.einsum('ab, abc-> ac', self.mu, AB), self.mu)
        muAb = jnp.einsum('ab,ab->a', jnp.einsum('ab,acb->ac', self.mu, A_mat), b_vec)
        aBm_b = jnp.einsum('ab, ab->a', a_vec, self._expectation_general_linear(B_mat, b_vec))
        return ABSigma_trace + mu_AB_mu + muAb + aBm_b
    
    def integrate_general_quadratic_inner(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None, B_mat: jnp.ndarray=None, b_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the quadratic expectation.
        
            int (Ax+a)'(Bx+b) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: jnp.ndarray [R]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return constant * self._expectation_general_quadratic_inner(A_mat, a_vec, B_mat, b_vec)
    
    def _expectation_general_quadratic_outer(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray, B_mat: jnp.ndarray, b_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param b_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: jnp.ndarray [R, K, L]
            The solved intergal.
        """
        Exx = self._expectation_xxT()
        AxxB = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', Exx, B_mat))
        Axb = jnp.einsum('ab,ac->abc', jnp.einsum('cab,cb->ca', A_mat, self.mu), b_vec)
        aBx_b = jnp.einsum('ba, bc->bac', a_vec, self._expectation_general_linear(B_mat, b_vec))
        return AxxB + Axb + aBx_b
    
    def integrate_general_quadratic_outer(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None, 
                                          B_mat: jnp.ndarray=None, b_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: jnp.ndarray [R,K,L]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return jnp.einsum('a,abc->abc', constant, self._expectation_general_quadratic_outer(A_mat, a_vec, B_mat, b_vec))
    
    ##### Cubic integrals
    
    def _expectation_xbxx(self, b_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the cubic expectation.
        
            int xb'xx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param b_vec: jnp.ndarray [1, D] or [R, D]
            Vector of 
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        Exx = self._expectation_xxT()
        mub_outer = jnp.einsum('ab,ac->abc', self.mu, b_vec)
        mbExx = jnp.einsum('abc,acd->abd', mub_outer, Exx)
        bmu_inner = jnp.einsum('ab,ab->a', self.mu, b_vec)
        bmSigma = jnp.einsum('a,abc->abc', bmu_inner, self.Sigma)
        bmu_outer = jnp.einsum('ab,ac->abc', b_vec, self.mu)
        Sigmabm = jnp.einsum('abd,ade->abe', self.Sigma, bmu_outer)
        return mbExx + bmSigma + Sigmabm
    
    
    def _expectation_cubic_outer(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the cubic expectation.
        
            int x(A'x + a)x' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,1,D] or [R,1,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [1,1] or [R,1]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        """
        # xAxx
        xAxx = self._expectation_xbxx(b_vec=A_mat)
        axx = a_vec[:,None,None] * self._expectation_xxT()
        return xAxx + axx
    
    def integrate_cubic_outer(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the cubic integration.
        
            int x(A'x + a)x' du(x).
            
        :param A_mat: jnp.ndarray [1,D] or [R,1,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [1] or [R,1]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        if A_mat is None:
            A_mat = jnp.ones((1,self.D))
        if a_vec is None:
            a_vec = jnp.zeros(1)
        if A_mat.ndim == 2:
            A_mat = jnp.tile(A_mat[None], [1, 1, 1])
        if a_vec.ndim == 1:
            a_vec = jnp.tile(a_vec[None], [1, 1])
        constant = self.integral()
        return constant[:,None,None] * self._expectation_cubic_outer(A_mat=A_mat[:,0], a_vec=a_vec[:,0])

    def intergate_xbxx(self, b_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the cubic integral.
        
            int xb'xx' du(x)
        :param b_vec: jnp.ndarray [D,]
            Vector of 
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xbxx(b_vec)
    
    def _expectation_xAxx(self, A_mat: jnp.ndarray) -> jnp.ndarray:
        """ Computes the cubic integral.
        
            int xAx'x dphi(x)
            
        :param A_mat: jnp.ndarray [D, D]
            Vector of 
        :return: jnp.ndarray [R, D]
            The solved intergal.
        """
        xAxm = jnp.einsum('ab,bc->ac', jnp.einsum('ab,abc->ac', self.mu, self.Sigma), A_mat)
        Am = jnp.einsum('ab,cb->ca', A_mat, self.mu)
        xAmx = jnp.einsum('abc,ab->ac', self.Sigma, Am)
        Exx = self._expectation_general_quadratic_inner(jnp.eye(self.D), jnp.zeros(self.D), jnp.eye(self.D), jnp.zeros(self.D))
        mA = jnp.einsum('ab,bc->ac', self.mu, A_mat)
        mAxx = jnp.einsum('ab,a->ab', mA, Exx)
        return xAxm + xAmx + mAxx
    
    
    def _expectation_general_cubic_inner(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray, 
                                         B_mat: jnp.ndarray, b_vec: jnp.ndarray,
                                         C_mat: jnp.ndarray, c_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param b_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
        :param C_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: jnp.ndarray [R, K]
            The solved intergal.
        """
        Amu_a = jnp.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        BSigmaC = jnp.einsum('cab,cbd->cad', B_mat, jnp.einsum('abc,adc->abd', self.Sigma, C_mat))
        BmubCmuc = jnp.einsum('ab,ab->a', Bmu_b, Cmu_c)
        
        BCm_c = jnp.einsum('cab,ca->cb', B_mat, Cmu_c)
        CBm_b = jnp.einsum('cab,ca->cb', C_mat, Bmu_b)
        first_term = jnp.einsum('abc,ac->ab', jnp.einsum('cab,cbd->cad', A_mat, self.Sigma), BCm_c + CBm_b)
        second_term = Amu_a * (self.get_trace(BSigmaC) + BmubCmuc)[:,None]
        return first_term + second_term
    
    
    def integrate_general_cubic_inner(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None, 
                                      B_mat: jnp.ndarray=None, b_vec: jnp.ndarray=None, 
                                      C_mat: jnp.ndarray=None, c_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the quadratic integration.
        
            int (Ax+a)(Bx+b)'(Cx+c)  du(x).
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)           
        :return: jnp.ndarray [R, K]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_cubic_inner(A_mat, a_vec, B_mat, b_vec, C_mat, c_vec)
    
    def _expectation_general_cubic_outer(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray,  
                                         B_mat: jnp.ndarray, b_vec: jnp.ndarray,
                                         C_mat: jnp.ndarray, c_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the cubic expectation.
        
            int (Ax+a)'(Bx+b)(Cx+c)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param b_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param C_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: jnp.ndarray [R, L]
            The solved intergal.
            
        # REMARK: Does the same thing as inner transposed.
        """
        Amu_a = jnp.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        BSigmaC = jnp.einsum('cab,cbd->cad', B_mat, jnp.einsum('abc,adc->abd', self.Sigma, C_mat))
        ASigmaC = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', self.Sigma, C_mat))
        ASigmaB = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', self.Sigma, B_mat))
        BmubCmuc = jnp.einsum('ab,ac->abc', Bmu_b, Cmu_c)
        AmuaCmuc = jnp.einsum('ab,ac->abc', Amu_a, Cmu_c)
        AmuaBmub = jnp.einsum('ab,ab->a', Amu_a, Bmu_b)
        first_term = jnp.einsum('ab,abc->ac', Amu_a, BSigmaC + BmubCmuc)
        second_term = jnp.einsum('ab,abc->ac', Bmu_b, ASigmaC + AmuaCmuc)
        third_term = - AmuaBmub[:,None] * Cmu_c
        fourth_term = self.get_trace(ASigmaB)[:,None] * Cmu_c
        return first_term + second_term + third_term + fourth_term
    
    def integrate_general_cubic_outer(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None, 
                                      B_mat: jnp.ndarray=None, b_vec: jnp.ndarray=None, 
                                      C_mat: jnp.ndarray=None, c_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the quadratic integration
        
           int (Bx+b)'(Cx+c)(Dx+d)' du(x),
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)      
        :return: jnp.ndarray [R, L]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_cubic_outer(A_mat, a_vec, B_mat, b_vec, C_mat, c_vec)
    
    ##### Quartic integrals
    
    def _expectation_xxTxxT(self) -> jnp.ndarray:
        """ Computes the cubic integral.
        
            int xx'xx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        mumu_outer = jnp.einsum('ab,ac->acb', self.mu, self.mu)
        Sigma_mumu_p = self.Sigma + mumu_outer
        Sigma_mumu_m = self.Sigma - mumu_outer
        mumu_inner = jnp.einsum('ab,ab->a', self.mu, self.mu)
        Sigma_trace = self.get_trace(self.Sigma)
        Sigma_mumu_p2 = jnp.einsum('abc,acd->abd', Sigma_mumu_p, Sigma_mumu_p)
        return 2. * Sigma_mumu_p2 + mumu_inner[:,None,None] * Sigma_mumu_m + Sigma_trace[:,None,None] * Sigma_mumu_p
        
    def integrate_xxTxxT(self) -> jnp.ndarray:
        """ Computes the quartic integral.
        
            int xx'xx' du(x).
            
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxTxxT()
    
    def _expectation_xxTAxxT(self, A_mat: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quartic expectation.
        
            int xx'Axx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A: jnp.ndarray [D,D]
            Square matrix.
            
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        mumu_outer = jnp.einsum('ab,ac->acb', self.mu, self.mu)
        Sigma_mumu_p = self.Sigma + mumu_outer
        Sigma_mumu_m = self.Sigma - mumu_outer
        AAT = A_mat + A_mat.T
        Sigma_AA_Sigma = jnp.einsum('abc, acd->abd', jnp.einsum('abc,cd->abd', Sigma_mumu_p, AAT), Sigma_mumu_p)
        muAmu = jnp.einsum('ab,ab->a', self.mu, jnp.einsum('ab,ca-> cb', A_mat, self.mu))
        ASigma_trace = self.get_trace(jnp.einsum('ab,cbd->cad', A_mat, self.Sigma))
        return Sigma_AA_Sigma + muAmu[:,None,None] * Sigma_mumu_m + ASigma_trace[:,None,None] * Sigma_mumu_p
    
    def integrate_xxTAxxT(self, A_mat: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quartic integral.
        
            int xx'xx' du(x)
        :return: jnp.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxTAxxT(A_mat)
    
    def _expectation_general_quartic_outer(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray, B_mat: jnp.ndarray, b_vec: jnp.ndarray, 
                                           C_mat: jnp.ndarray, c_vec: jnp.ndarray, D_mat: jnp.ndarray, d_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param b_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
        :param C_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
        :param D_mat: jnp.ndarray [1,M,D] or [R,M,D]
            Real valued matrix.
        :param d_vec: jnp.ndarray [1,M,D] or [R,M,D]
            Real valued vector.
            
        :return: jnp.ndarray [R, K, M]
            The solved intergal.
        """
        Amu_a = jnp.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        Dmu_d = jnp.einsum('cab,cb-> ca', D_mat, self.mu) + d_vec
        ASigmaB = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', self.Sigma, B_mat))
        CSigmaD = jnp.einsum('cab,cbd->cad', C_mat, jnp.einsum('abc,adc->abd', self.Sigma, D_mat))
        ASigmaC = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', self.Sigma, C_mat))
        BSigmaD = jnp.einsum('cab,cbd->cad', B_mat, jnp.einsum('abc,adc->abd', self.Sigma, D_mat))
        ASigmaD = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', self.Sigma, D_mat))
        BSigmaC = jnp.einsum('cab,cbd->cad', B_mat, jnp.einsum('abc,adc->abd', self.Sigma, C_mat))
        AmuaBmub = jnp.einsum('ab,ac->abc', Amu_a, Bmu_b)
        CmucDmud = jnp.einsum('ab,ac->abc', Cmu_c, Dmu_d)
        AmuaCmuc = jnp.einsum('ab,ac->abc', Amu_a, Cmu_c)
        BmubDmud = jnp.einsum('ab,ac->abc', Bmu_b, Dmu_d)
        BmubCmuc = jnp.einsum('ab,ab->a', Bmu_b, Cmu_c)
        AmuaDmud = jnp.einsum('ab,ac->abc', Amu_a, Dmu_d)
        first_term = jnp.einsum('abc,acd->abd', ASigmaB + AmuaBmub, CSigmaD + CmucDmud)
        second_term = jnp.einsum('abc,acd->abd', ASigmaC + AmuaCmuc, BSigmaD + BmubDmud)
        third_term = BmubCmuc[:,None,None] * (ASigmaD - AmuaDmud)
        fourth_term = self.get_trace(BSigmaC)[:,None,None] * (ASigmaD + AmuaDmud)
        return first_term + second_term + third_term + fourth_term
        
    def integrate_general_quartic_outer(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None, 
                                       B_mat: jnp.ndarray=None, b_vec: jnp.ndarray=None, 
                                       C_mat: jnp.ndarray=None, c_vec: jnp.ndarray=None, 
                                       D_mat: jnp.ndarray=None, d_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the quartic integral.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' du(x).
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param D_mat: jnp.ndarray [M,D] or [R,M,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param d_vec: jnp.ndarray [M,D] or [R,M,D]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: jnp.ndarray [R, K, M]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        D_mat, d_vec = self._get_default(D_mat, d_vec)
        constant = self.integral()
        return constant[:,None,None] * self._expectation_general_quartic_outer(A_mat,a_vec,B_mat,b_vec,C_mat,c_vec,D_mat,d_vec)
    
    def _expectation_general_quartic_inner(self, A_mat: jnp.ndarray, a_vec: jnp.ndarray, 
                                           B_mat: jnp.ndarray, b_vec: jnp.ndarray, 
                                           C_mat: jnp.ndarray, c_vec: jnp.ndarray, 
                                           D_mat: jnp.ndarray, d_vec: jnp.ndarray) -> jnp.ndarray:
        """ Computes the quartic expectation.
        
            int (Ax+a)'(Bx+b)(Cx+c)'(Dx+d) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param a_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param B_mat: jnp.ndarray [1,K,D] or [R,K,D]
            Real valued matrix.
        :param b_vec: jnp.ndarray [1,K] or [R,K]
            Real valued vector.
        :param C_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param c_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
        :param D_mat: jnp.ndarray [1,L,D] or [R,L,D]
            Real valued matrix.
        :param d_vec: jnp.ndarray [1,L] or [R,L]
            Real valued vector.
            
        :return: jnp.ndarray [R]
            The solved intergal.
        """
        Amu_a = jnp.einsum('cab,cb-> ca', A_mat, self.mu) + a_vec
        Bmu_b = jnp.einsum('cab,cb-> ca', B_mat, self.mu) + b_vec
        Cmu_c = jnp.einsum('cab,cb-> ca', C_mat, self.mu) + c_vec
        Dmu_d = jnp.einsum('cab,cb-> ca', D_mat, self.mu) + d_vec
        ASigmaB = jnp.einsum('cab,cbd->cad', A_mat, jnp.einsum('abc,adc->abd', self.Sigma, B_mat))
        CSigmaD = jnp.einsum('cab,cbd->cad', C_mat, jnp.einsum('abc,adc->abd', self.Sigma, D_mat))
        
        AmuaBmub = jnp.einsum('ab,ab->a', Amu_a, Bmu_b)
        CmucDmud = jnp.einsum('ab,ab->a', Cmu_c, Dmu_d)
        CD = jnp.einsum('abc,abd->acd', C_mat, D_mat)
        CD_DC = CD + jnp.swapaxes(CD, axis1=1, axis2=2)
        SCD_DCS = jnp.einsum('abc,acd->abd', jnp.einsum('abc,acd->abd', self.Sigma, CD_DC), self.Sigma)
        ASCD_DCSB = jnp.einsum('abc,adc->abd', jnp.einsum('cab,cbd->cad', A_mat, SCD_DCS), B_mat)
        Am_aB = jnp.einsum('ab,abc->ac', Amu_a, B_mat)
        Bm_bA = jnp.einsum('ab,abc->ac', Bmu_b, A_mat)
        CDm_d = jnp.einsum('cab,ca->cb', C_mat, Dmu_d)
        DCm_c = jnp.einsum('cab,ca->cb', D_mat, Cmu_c)
        first_term = self.get_trace(ASCD_DCSB)
        second_term = jnp.einsum('ab,ab->a', jnp.einsum('ab,abc->ac', Am_aB + Bm_bA, self.Sigma), CDm_d + DCm_c)
        third_term = (self.get_trace(ASigmaB) + AmuaBmub) * (self.get_trace(CSigmaD) + CmucDmud)
        return first_term + second_term + third_term
        
    def integrate_general_quartic_inner(self, A_mat: jnp.ndarray=None, a_vec: jnp.ndarray=None, 
                                       B_mat: jnp.ndarray=None, b_vec: jnp.ndarray=None, 
                                       C_mat: jnp.ndarray=None, c_vec: jnp.ndarray=None, 
                                       D_mat: jnp.ndarray=None, d_vec: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the quartic integral.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' du(x).
            
        :param A_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param B_mat: jnp.ndarray [K,D] or [R,K,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: jnp.ndarray [K] or [R,K]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param C_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
        :param D_mat: jnp.ndarray [L,D] or [R,L,D]
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param d_vec: jnp.ndarray [L] or [R,L]
            Real valued vector. If None, it is assumed zeros. (Default=None)
            
        :return: jnp.ndarray [R]
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
        self.Sigma = jnp.diag(1. / self.Lambda.diagonal(axis1=1,axis2=2))
        self.ln_det_Lambda = jnp.sum(jnp.log(self.Lambda.diagonal()))
        self.ln_det_Sigma = -self.ln_det_Lambda   
        
    @staticmethod
    def invert_diagonal(A: jnp.ndarray) -> (jnp.ndarray,jnp.ndarray):
        A_inv = jnp.concatenate([jnp.diag(mat)[None] for mat in  1./A.diagonal(axis1=1, axis2=2)], axis=0)
        ln_det_A = jnp.sum(jnp.log(A.diagonal(axis1=1, axis2=2)), axis=1)
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