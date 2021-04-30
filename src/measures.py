import numpy
import factors
from scipy.special import logsumexp

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
        
    def slice(self, indices: list):
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
        
    def evaluate_ln(self, x: numpy.ndarray):
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
    
    def evaluate(self, x: numpy.ndarray):
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
    
    def integrate(self, expr:str='1', **kwargs):
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
                                 'Ax_aBx_bCx_cDx_d_inner': self.integrate_general_quartic_inner,
                                 'Ax_aBx_bCx_cDx_d_outer': self.integrate_general_quartic_outer}
        
    def slice(self, indices: list):
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
    
    def integrate(self, expr:str='1', **kwargs):
        """ Integrates the indicated expression with respect to the Gaussian measure.
        
        :param expr: str
            Indicates the expression that should be integrated. Check measure's integration dict. Default='1'.
        :kwargs:
            All parameters, that are required to evaluate the expression.
        """
        return self.integration_dict[expr](**kwargs)
    
    
    ### This needs to be unified with the functionality of conjugate factors.
    def multiply_squared_exponential_term(self, Lambda: numpy.ndarray, nu: numpy.ndarray=None, ln_beta: numpy.ndarray=None, r: list=[]):
        """ Multiplies an exponential factor with another one.
        
        u(x) * beta * exp(-.5 * x'Lambda x + x'nu) 
        
        :param Lambda: numpy.ndarray [R2, D, D]
            Matrix of the second order term.
        :param nu: numpy.ndarray [R2, D] or None
            Vector of the linear order term. If None, zero vector is assumed. (Default=None)
        :param ln_beta: numpy.ndarray [R2] or None
            Log constant. If None, zero is assumed. (Default=None)
        :param r: list
            Indices of densities that need to be evaluated. If empty, all densities are evaluated. (Default=[])
        
        :return: SquaredExponential
            The resulting object.
        """
        if len(r) == 0:
            r = range(self.R)
        R1, R2 = len(r), Lambda.shape[0]
        Lambda_new = (self.Lambda[r,None] + Lambda[None]).reshape((R1 * R2, self.D, self.D))
        if nu is None:
            nu_new = numpy.tile(self.nu[r,None], (1,R2,1)).reshape((R1 * R2, self.D))
        else:
            nu_new = (self.nu[r,None] + nu[None]).reshape((R1 * R2, self.D))
        if ln_beta is None:
            ln_beta_new = numpy.tile(self.ln_beta[r,None], (1, R2)).reshape((R1 * R2))
        else:
            ln_beta_new = (self.ln_beta[r,None] + ln_beta[None]).reshape((R1 * R2))
        product = GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        return product
    
    def multiply_linear_exponential_term(self, nu: numpy.ndarray, ln_beta: numpy.ndarray=None, r: list=[]):
        """ Multiplies an exponential factor with another one.
        
        u(x) * beta * exp(x'nu) 
        
        :param nu: numpy.ndarray [R2, D] or None
            Vector of the linear order term. If None, zero vector is assumed. (Default=None)
        :param ln_beta: numpy.ndarray [R2] or None
            Log constant. If None, zero is assumed. (Default=None)
        :param r: list
            Indices of densities that need to be evaluated. If empty, all densities are evaluated. (Default=[])
        
        :return: SquaredExponential
            The resulting object.
        """
        
        ### Remark: Discrepancy to sampling becomes big, if nu shifts the mean strongly. Check if that is only a numerical issue.
        if len(r) == 0:
            r = range(self.R)
        R1, R2 = len(r), nu.shape[0]
        Lambda_new = numpy.tile(self.Lambda[r,None], (1, R2, 1, 1)).reshape((R1 * R2, self.D, self.D))
        nu_new = (self.nu[r,None] + nu[None]).reshape((R1 * R2, self.D))
        if ln_beta is None:
            ln_beta_new = numpy.tile(self.ln_beta[r,None], (1, R2)).reshape((R1 * R2))
        else:
            ln_beta_new = (self.ln_beta[r,None] + ln_beta[None]).reshape((R1 * R2))
        product = GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        product.invert_lambda()
        if self.Sigma is not None:
            product.Sigma = numpy.tile(self.Sigma[r,None], (1, R2, 1, 1)).reshape((R1 * R2, self.D, self.D))
        if self.ln_det_Sigma is not None:
            product.ln_det_Sigma = numpy.tile(self.ln_det_Sigma[r,None], (1, R2)).reshape((R1 * R2))
            product.ln_det_Lambda = -product.ln_det_Sigma
        elif self.ln_det_Lambda is not None:
            product.ln_det_Lambda = numpy.tile(self.ln_det_Lambda[r,None], (1, R2)).reshape((R1 * R2))
            product.ln_det_Sigma = -product.ln_det_Lambda
            
        return product
    
    def multiply_constant_term(self, ln_beta: numpy.ndarray, r: list=[]):
        """ Multiplies an exponential factor with another one.
        
        u(x) * beta
        
        :param nu: numpy.ndarray [R2, D] or None
            Vector of the linear order term. If None, zero vector is assumed. (Default=None)
        :param ln_beta: numpy.ndarray [R2] or None
            Log constant. If None, zero is assumed. (Default=None)
        :param r: list
            Indices of densities that need to be evaluated. If empty, all densities are evaluated. (Default=[])
        
        :return: SquaredExponential
            The resulting object.
        """
        if len(r) == 0:
            r = range(self.R)
        R1, R2 = len(r), ln_beta.shape[0]
        Lambda_new = numpy.tile(self.Lambda[r,None], (1, R2, 1, 1)).reshape((R1 * R2, self.D, self.D))
        nu_new = numpy.tile(self.nu[r,None], (1,R2,1)).reshape((R1 * R2, self.D))
        ln_beta_new = (self.ln_beta[r,None] + ln_beta[None]).reshape((R1 * R2))
        product = GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        if self.Sigma is not None:
            product.Sigma = numpy.tile(self.Sigma[r,None], (1, R2, 1, 1)).reshape((R1 * R2, self.D, self.D))
        if self.ln_det_Sigma is not None:
            product.ln_det_Sigma = numpy.tile(self.ln_det_Sigma[r,None], (1, R2)).reshape((R1 * R2))
            product.ln_det_Lambda = -product.ln_det_Sigma
        elif self.ln_det_Lambda is not None:
            product.ln_det_Lambda = numpy.tile(self.ln_det_Lambda[r,None], (1, R2)).reshape((R1 * R2))
            product.ln_det_Sigma = -product.ln_det_Lambda
        return product
    
    def multiply_rank_one(self, U: numpy.ndarray, G: numpy.ndarray, nu: numpy.ndarray=None, ln_beta: numpy.ndarray=None, r: list=[]):
        """ Multiplies the exponential term with another exponential term, where the Lambda is rank 1, i.e.
        
        Lambda = U G U'
            
        Where G is an [1 x 1] diagonal matrix and U and [D x 1] with a vector. 
        If already computed, the covariance matrix Sigma and its log-determinant are efficiently updated.
        
        :param U: numpy.ndarray [R1, D]
            Vector of low rank matrix with orthogonal vectors.
        :param G: numpy.ndarray [R1]
            Diagonal entries of the low-rank matrix.
        :param nu: numpy.ndarray [R1, D]
            Information vector of the low rank part. If None all entries are zero. (Default=None)
        :param nu: numpy.ndarray [R1]
            Log factor of the low rank part. If None all entries are zero. (Default=None)
        :param r: list
            Indices of densities that need to be evaluated. If empty, all densities are evaluated. (Default=[])
            
        :return: Squared exponential
            The resulting product, where the number of Gaussians is self.R * R1.
        """
        if len(r) == 0:
            r = range(self.R)
        R = len(r)
        R1 = G.shape[0]
        UGU = numpy.einsum('ab,ac->abc', U, G[:,None] * U)
        Lambda_new = (self.Lambda[r,None] + UGU[None]).reshape((R * R1, self.D, self.D))
        if nu is None:
            nu_new = numpy.tile(self.nu[r,None], (1, R1, 1)).reshape((R * R1, self.D))
        else:
            nu_new = (self.nu[r,None] + nu[None]).reshape((R * R1, self.D))
        if ln_beta is None:
            ln_beta_new = numpy.tile(self.ln_beta[:,None], (1,R1)).reshape((R * R1))
        else:
            ln_beta_new = (self.ln_beta[r,None] + ln_beta[None]).reshape((R * R1))
        product = GaussianMeasure(Lambda=Lambda_new, nu=nu_new, ln_beta=ln_beta_new)
        
        # if the Sigma of the object is known the Sherman-morrison formula and the matrix determinant lemma are used for efficient update of the inverses and the log determinants.
        if self.Sigma is not None and self.ln_det_Sigma is not None:
            # Sherman morrison inversion
            Sigma_U = numpy.einsum('abc,dc->adb', self.Sigma[r], U)
            U_Sigma_U = numpy.einsum('abc,bc->ab', Sigma_U, U)
            denominator = 1. + G[None] * U_Sigma_U
            nominator = G[None,:,None,None] * numpy.einsum('abc,abd->abcd', Sigma_U, Sigma_U)
            Sigma_new = self.Sigma[r, None] - nominator / denominator[:,:,None,None]
            product.Sigma = Sigma_new.reshape((R*R1, self.D, self.D))
            # Matrix determinant lemma
            ln_det_Sigma_new = self.ln_det_Sigma[r,None] - numpy.log(denominator)
            product.ln_det_Sigma = ln_det_Sigma_new.reshape((R * R1))
            product.ln_det_Lambda = -product.ln_det_Sigma
        return product
    
    def log_integral(self):
        """ Computes the log integral of the exponential term.
        
        log \int u(x) dx.
        
        :return: numpy.ndarray [R]
            Log integral
        """
        self._prepare_integration()
        return self.lnZ + self.ln_beta
    
    def integral(self):
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
        
    def is_normalized(self):
        return numpy.equal(self.lnZ, -self.ln_beta)
    
    def compute_mu(self):
        """ Converts from information to mean vector.
        
        :return: numpy.ndarray [R, D]
            Mean vector.
        """
        if self.Sigma is None:
            self.invert_lambda()
        self.mu = numpy.einsum('abc,ac->ab', self.Sigma, self.nu)
    
    def get_density(self):
        """ Returns the corresponing normalised density object.
        
        :return: GaussianDensity
            Corresponding density object.
        """
        import densities
        self._prepare_integration()
        return densities.GaussianDensity(Sigma=self.Sigma, mu=self.mu, Lambda=self.Lambda, ln_det_Sigma=self.ln_det_Sigma)
            
        
    def _get_default(self, mat, vec):
        """ Small method to get default matrix and vector.
        """
        if mat is None:
            mat = numpy.eye(self.D)
        if vec is None:
            vec = numpy.zeros(mat.shape[0])
        return mat, vec
            
    ##### Linear integals
            
    def _expectation_x(self):
        """ Computes the expectation.
        
            int x du(x) / int du(x)
        
        :return: numpy.ndarray [R, D]
            The solved intergal.
        """
        return self.mu 
            
    def integrate_x(self):
        """ Computes the integral.
        
            int x du(x)
        
        :return: numpy.ndarray [R, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None] * self._expectation_x()
    
    def _expectation_general_linear(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray):
        """ Computes the linear expectation.
        
            int (Ax+a) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
            
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        return numpy.einsum('ab,cb->ca', A_mat, self.mu) + a_vec
    
    def integrate_general_linear(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None):
        """ Computes the linear expectation.
        
            int (Ax+a) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
            
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_linear(A_mat, a_vec)
        
    
    ##### Quadratic integrals
    
    def _expectation_xxT(self):
        """ Computes the expectation.
        
            int xx' du(x) / int du(x)
        
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        return self.Sigma + numpy.einsum('ab,ac->acb', self.mu, self.mu)
    
    def integrate_xxT(self):
        """ Computes the integral.
        
            int xx' du(x)
        
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxT()
    
    def _expectation_general_quadratic_inner(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, B_mat: numpy.ndarray, b_vec: numpy.ndarray):
        """ Computes the quartic expectation.
        
            int (Ax+a)'(Bx+b) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        AB = numpy.dot(A_mat.T, B_mat)
        ABSigma_trace = self.get_trace(numpy.einsum('ab,cbd->cad', AB, self.Sigma))
        mu_AB_mu = numpy.einsum('ab,ab->a', numpy.einsum('ab, bc-> ac', self.mu, AB), self.mu)
        muAb = numpy.einsum('ab,b->a', numpy.einsum('ab,cb->ac', self.mu, A_mat), b_vec)
        aBm_b = numpy.einsum('a, ba->b', a_vec, self._expectation_general_linear(B_mat, b_vec))
        return ABSigma_trace + mu_AB_mu + muAb + aBm_b
    
    def integrate_general_quadratic_inner(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None):
        """ Computes the quadratic expectation.
        
            int (Ax+a)'(Bx+b) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
        :param B_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return constant * self._expectation_general_quadratic_inner(A_mat, a_vec, B_mat, b_vec)
    
    def _expectation_general_quadratic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, B_mat: numpy.ndarray, b_vec: numpy.ndarray):
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [L,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        Exx = self._expectation_xxT()
        AxxB = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,dc->abd', Exx, B_mat))
        Axb = numpy.einsum('ab,c->abc', numpy.einsum('ab,cb->ca', A_mat, self.mu), b_vec)
        aBx_b = numpy.einsum('a, bc->bac', a_vec, self._expectation_general_linear(B_mat, b_vec))
        return AxxB + Axb + aBx_b
    
    def integrate_general_quadratic_outer(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                          B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None):
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
        :param B_mat: numpy.ndarray [L,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
            
        :return: numpy.ndarray [R,K,L]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        constant = self.integral()
        return constant[:,None,None] * self._expectation_general_quadratic_outer(A_mat, a_vec, B_mat, b_vec)
    
    ##### Cubic integrals
    
    def _expectation_xbxx(self, b_vec: numpy.ndarray):
        """ Computes the cubic expectation.
        
            int xb'xx' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param b_vec: numpy.ndarray [D]
            Vector of 
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        Exx = self._expectation_xxT()
        mub_outer = numpy.einsum('ab,c->abc', self.mu, b_vec)
        mbExx = numpy.einsum('abc,acd->abd', mub_outer, Exx)
        bmu_inner = numpy.einsum('ab,b->a', self.mu, b_vec)
        bmSigma = numpy.einsum('a,abc->abc', bmu_inner, self.Sigma)
        bmu_outer = numpy.einsum('a,bc->bac', b_vec, self.mu)
        Sigmabm = numpy.einsum('abd,ade->abe', self.Sigma, bmu_outer)
        return mbExx + bmSigma + Sigmabm
        
    
    def intergate_xbxx(self, b_vec: numpy.ndarray):
        """ Computes the cubic integral.
        
            int xb'xx' du(x)
        :param b_vec: numpy.ndarray [D,]
            Vector of 
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xbxx(b_vec)
    
    def _expectation_xAxx(self, A_mat: numpy.ndarray):
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
                                         C_mat: numpy.ndarray, c_vec: numpy.ndarray):
        """ Computes the quartic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [L]
            Real valued vector.
        :param C_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [L]
            Real valued vector.
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        Amu_a = numpy.einsum('ab,cb-> ca', A_mat, self.mu) + a_vec[None]
        Bmu_b = numpy.einsum('ab,cb-> ca', B_mat, self.mu) + b_vec[None]
        Cmu_c = numpy.einsum('ab,cb-> ca', C_mat, self.mu) + c_vec[None]
        BSigmaC = numpy.einsum('ab,cbd->cad', B_mat, numpy.einsum('abc,cd->abd', self.Sigma, C_mat.T))
        BmubCmuc = numpy.einsum('ab,ab->a', Bmu_b, Cmu_c)
        
        BCm_c = numpy.einsum('ab,ca->cb', B_mat, Cmu_c)
        CBm_b = numpy.einsum('ab,ca->cb', C_mat, Bmu_b)
        first_term = numpy.einsum('abc,ac->ab', numpy.einsum('ab,cbd->cad', A_mat, self.Sigma), BCm_c + CBm_b)
        second_term = Amu_a * (self.get_trace(BSigmaC) + BmubCmuc)[:,None]
        return first_term + second_term
    
    
    def integrate_general_cubic_inner(self, A_mat: numpy.ndarray=None, a_vec: numpy.ndarray=None, 
                                      B_mat: numpy.ndarray=None, b_vec: numpy.ndarray=None, 
                                      C_mat: numpy.ndarray=None, c_vec: numpy.ndarray=None):
        """ Computes the quadratic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c)  dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
        :param B_mat: numpy.ndarray [L,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
        :param C_mat: numpy.ndarray [L,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: numpy.ndarray [L] or None
            Real valued vector. If None, it is assumed identity. (Default=None)            
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
                                         C_mat: numpy.ndarray, c_vec: numpy.ndarray):
        """ Computes the cubic expectation.
        
            int (Ax+a)'(Bx+b)(Cx+c)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [L]
            Real valued vector.
        :param B_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [L]
            Real valued vector.
        :param C_mat: numpy.ndarray [M,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [M]
            Real valued vector.
            
        :return: numpy.ndarray [R, K, M]
            The solved intergal.
            
        # REMARK: Does the same thing as inner transposed.
        """
        Amu_a = numpy.einsum('ab,cb-> ca', A_mat, self.mu) + a_vec[None]
        Bmu_b = numpy.einsum('ab,cb-> ca', B_mat, self.mu) + b_vec[None]
        Cmu_c = numpy.einsum('ab,cb-> ca', C_mat, self.mu) + c_vec[None]
        BSigmaC = numpy.einsum('ab,cbd->cad', B_mat, numpy.einsum('abc,cd->abd', self.Sigma, C_mat.T))
        ASigmaC = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,cd->abd', self.Sigma, C_mat.T))
        ASigmaB = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,cd->abd', self.Sigma, B_mat.T))
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
                                      C_mat: numpy.ndarray=None, c_vec: numpy.ndarray=None):
        """ Computes the quadratic expectation.
        
           int (Bx+b)'(Cx+c)(Dx+d)' du(x),
            
        :param A_mat: numpy.ndarray [K,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param a_vec: numpy.ndarray [K] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
        :param B_mat: numpy.ndarray [L,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param b_vec: numpy.ndarray [L] or None
            Real valued vector. If None, it is assumed identity. (Default=None)
        :param C_mat: numpy.ndarray [L,D] or None
            Real valued matrix. If None, it is assumed identity. (Default=None)
        :param c_vec: numpy.ndarray [L] or None
            Real valued vector. If None, it is assumed identity. (Default=None)            
        :return: numpy.ndarray [R, K]
            The solved intergal.
        """
        A_mat, a_vec = self._get_default(A_mat, a_vec)
        B_mat, b_vec = self._get_default(B_mat, b_vec)
        C_mat, c_vec = self._get_default(C_mat, c_vec)
        constant = self.integral()
        return constant[:,None] * self._expectation_general_cubic_outer(A_mat, a_vec, B_mat, b_vec, C_mat, c_vec)
    
    ##### Quartic integrals
    
    def _expectation_xxTxxT(self):
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
        
    def integrate_xxTxxT(self):
        """ Computes the quartic integral.
        
            int xx'xx' du(x).
            
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxTxxT()
    
    def _expectation_xxTAxxT(self, A_mat: numpy.ndarray):
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
    
    def integrate_xxTAxxT(self, A_mat: numpy.ndarray):
        """ Computes the quartic integral.
        
            int xx'xx' du(x)
        :return: numpy.ndarray [R, D, D]
            The solved intergal.
        """
        constant = self.integral()
        return constant[:,None,None] * self._expectation_xxTAxxT(A_mat)
    
    def _expectation_general_quartic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, B_mat: numpy.ndarray, b_vec: numpy.ndarray, 
                                           C_mat: numpy.ndarray, c_vec: numpy.ndarray, D_mat: numpy.ndarray, d_vec: numpy.ndarray):
        """ Computes the quartic expectation.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [L]
            Real valued vector.
        :param C_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [L]
            Real valued vector.
        :param D_mat: numpy.ndarray [M,D]
            Real valued matrix.
        :param d_vec: numpy.ndarray [M]
            Real valued vector.
            
        :return: numpy.ndarray [R, K, M]
            The solved intergal.
        """
        Amu_a = numpy.einsum('ab,cb-> ca', A_mat, self.mu) + a_vec[None]
        Bmu_b = numpy.einsum('ab,cb-> ca', B_mat, self.mu) + b_vec[None]
        Cmu_c = numpy.einsum('ab,cb-> ca', C_mat, self.mu) + c_vec[None]
        Dmu_d = numpy.einsum('ab,cb-> ca', D_mat, self.mu) + d_vec[None]
        ASigmaB = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,cd->abd', self.Sigma, B_mat.T))
        CSigmaD = numpy.einsum('ab,cbd->cad', C_mat, numpy.einsum('abc,cd->abd', self.Sigma, D_mat.T))
        ASigmaC = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,cd->abd', self.Sigma, C_mat.T))
        BSigmaD = numpy.einsum('ab,cbd->cad', B_mat, numpy.einsum('abc,cd->abd', self.Sigma, D_mat.T))
        ASigmaD = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,cd->abd', self.Sigma, D_mat.T))
        BSigmaC = numpy.einsum('ab,cbd->cad', B_mat, numpy.einsum('abc,cd->abd', self.Sigma, C_mat.T))
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
        
    def integrate_general_quartic_outer(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, 
                                       B_mat: numpy.ndarray, b_vec: numpy.ndarray, 
                                       C_mat: numpy.ndarray, c_vec: numpy.ndarray, 
                                       D_mat: numpy.ndarray, d_vec: numpy.ndarray):
        """ Computes the quartic integral.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [L]
            Real valued vector.
        :param C_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [L]
            Real valued vector.
        :param D_mat: numpy.ndarray [M,D]
            Real valued matrix.
        :param d_vec: numpy.ndarray [M]
            Real valued vector.
            
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
                                           D_mat: numpy.ndarray, d_vec: numpy.ndarray):
        """ Computes the quartic expectation.
        
            int (Ax+a)'(Bx+b)(Cx+c)'(Dx+d) dphi(x),
            
            with phi(x) = u(x) / int du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [L]
            Real valued vector.
        :param C_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [L]
            Real valued vector.
        :param D_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param d_vec: numpy.ndarray [L]
            Real valued vector.
            
        :return: numpy.ndarray [R]
            The solved intergal.
        """
        Amu_a = numpy.einsum('ab,cb-> ca', A_mat, self.mu) + a_vec[None]
        Bmu_b = numpy.einsum('ab,cb-> ca', B_mat, self.mu) + b_vec[None]
        Cmu_c = numpy.einsum('ab,cb-> ca', C_mat, self.mu) + c_vec[None]
        Dmu_d = numpy.einsum('ab,cb-> ca', D_mat, self.mu) + d_vec[None]
        ASigmaB = numpy.einsum('ab,cbd->cad', A_mat, numpy.einsum('abc,cd->abd', self.Sigma, B_mat.T))
        CSigmaD = numpy.einsum('ab,cbd->cad', C_mat, numpy.einsum('abc,cd->abd', self.Sigma, D_mat.T))
        
        AmuaBmub = numpy.einsum('ab,ab->a', Amu_a, Bmu_b)
        CmucDmud = numpy.einsum('ab,ab->a', Cmu_c, Dmu_d)
        CD = numpy.dot(C_mat.T, D_mat)
        CD_DC = CD + CD.T
        SCD_DCS = numpy.einsum('abc,acd->abd', numpy.einsum('abc,cd->abd', self.Sigma, CD_DC), self.Sigma)
        ASCD_DCSB = numpy.einsum('abc,dc->abd', numpy.einsum('ab,cbd->cad', A_mat, SCD_DCS), B_mat)
        Am_aB = numpy.einsum('ab,bc->ac', Amu_a, B_mat)
        Bm_bA = numpy.einsum('ab,bc->ac', Bmu_b, A_mat)
        CDm_d = numpy.einsum('ab,ca->cb', C_mat, Dmu_d)
        DCm_c = numpy.einsum('ab,ca->cb', D_mat, Cmu_c)
        first_term = self.get_trace(ASCD_DCSB)
        second_term = numpy.einsum('ab,ab->a', numpy.einsum('ab,abc->ac', Am_aB + Bm_bA, self.Sigma), CDm_d + DCm_c)
        third_term = (self.get_trace(ASigmaB) + AmuaBmub) * (self.get_trace(CSigmaD) + CmucDmud)
        return first_term + second_term + third_term
        
    def integrate_general_quartic_inner(self, A_mat: numpy.ndarray, a_vec: numpy.ndarray, 
                                       B_mat: numpy.ndarray, b_vec: numpy.ndarray, 
                                       C_mat: numpy.ndarray, c_vec: numpy.ndarray, 
                                       D_mat: numpy.ndarray, d_vec: numpy.ndarray):
        """ Computes the quartic integral.
        
            int (Ax+a)(Bx+b)'(Cx+c)(Dx+d)' du(x).
            
        :param A_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param a_vec: numpy.ndarray [K]
            Real valued vector.
        :param B_mat: numpy.ndarray [K,D]
            Real valued matrix.
        :param b_vec: numpy.ndarray [K]
            Real valued vector.
        :param C_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param c_vec: numpy.ndarray [L]
            Real valued vector.
        :param D_mat: numpy.ndarray [L,D]
            Real valued matrix.
        :param d_vec: numpy.ndarray [L]
            Real valued vector.
            
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
    def invert_diagonal(A: numpy.ndarray):
        A_inv = numpy.concatenate([numpy.diag(mat)[None] for mat in  1./A.diagonal(axis1=1, axis2=2)], axis=0)
        ln_det_A = numpy.sum(numpy.log(A.diagonal(axis1=1, axis2=2)), axis=1)
        return A_inv, ln_det_A
    
    def slice(self, indices: list):
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