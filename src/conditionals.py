import numpy

class ConditionalGaussianDensity:
    
    def __init__(self, M, b, Sigma=None, Lambda=None, ln_det_Sigma=None):
        """ A conditional Gaussian density
            
            p(y|x) = N(mu(x), Sigma)
            
            with the conditional mean function mu(x) = M x + b.
                
        :param M: numpy.ndarray [R, Dy, Dx]
            Matrix in the mean function.
        :param b: numpy.ndarray [R, Dy]
            Vector in the conditional mean function.
        :param Sigma: numpy.ndarray [R, Dy, Dy]
            The covariance matrix of the conditional. (Default=None)
        :param Lambda: numpy.ndarray [R, Dy, Dy] or None
            Information (precision) matrix of the Gaussians. (Default=None)
        :param ln_det_Sigma: numpy.ndarray [R] or None
            Log determinant of the covariance matrix. (Default=None)
        """
        
        self.R, self.Dy, self.Dx = M.shape
        self.M = M
        self.b = b
        if Sigma is None and Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif Sigma is not None:
            self.Sigma = Sigma
            if Lambda is None or ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = self.invert_matrix(self.Sigma)
            else:
                self.Lambda, self.ln_det_Sigma = Lambda, ln_det_Sigma
            self.ln_det_Lambda = -self.ln_det_Sigma
        else:
            self.Lambda = Lambda
            if Sigma is None or ln_det_Sigma is None:
                self.Sigma, self.ln_det_Lambda = self.invert_matrix(self.Sigma)
            else:
                self.Sigma, self.ln_det_Lambda = Lambda, ln_det_Sigma
            self.ln_det_Sigma = -self.ln_det_Lambda
            
    def slice(self, indices: list) -> 'ConditionalGaussianDensity':
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: ConditionalGaussianDensity
            The resulting Gaussian diagonal density.
        """
        M_new = self.M[indices]
        b_new = self.b[indices]
        Lambda_new = self.Lambda[indices]
        Sigma_new = self.Sigma[indices]
        ln_det_Sigma_new = self.ln_det_Sigma[indices]
        new_measure = ConditionalGaussianDensity(M_new, b_new, Sigma_new, Lambda_new, ln_det_Sigma_new)
        return new_measure
            
    def get_conditional_mu(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Computest the conditional mu function
        
            mu(x) = M x + b.
            
        :param y: numpy.ndarray [N, Dx]
            Instances, the mu should be conditioned on.
        
        :return: numpy.ndarray [R, N, Dy]
            Conditional means.
        """
        mu_y = numpy.einsum('abc,dc->adb', self.M, x) + self.b[:,None]
        return mu_y
    
    def condition_on_x(self, x: numpy.ndarray) -> 'GaussianDensity':
        """ Generates the corresponding Gaussian Density conditioned on x.
        
        :param x: numpy.ndarray [N, Dx]
            Instances, the mu should be conditioned on.
        
        :return: GaussianDensity
            The density conditioned on x.
        """
        from densities import GaussianDensity
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((self.R * N, self.Dy))
        Sigma_new = numpy.tile(self.Sigma[:,None], (1,N,1,1)).reshape(self.R * N, self.Dy, self.Dy)
        Lambda_new = numpy.tile(self.Lambda[:,None], (1,N,1,1)).reshape(self.R * N, self.Dy, self.Dy)
        ln_det_Sigma_new = numpy.tile(self.ln_det_Sigma[:,None], (1,N)).reshape(self.R * N)
        return GaussianDensity(Sigma=Sigma_new, mu=mu_new, Lambda=Lambda_new, ln_det_Sigma=ln_det_Sigma_new)
        
    @staticmethod
    def invert_matrix(A: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        L = numpy.linalg.cholesky(A)
        # TODO: Check whether we can make it mor efficienty with solve_triangular.
        #L_inv = solve_triangular(L, numpy.eye(L.shape[0]), lower=True,
        #                         check_finite=False)
        L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[1])[None])
        A_inv = numpy.einsum('acb,acd->abd', L_inv, L_inv)
        ln_det_A = 2. * numpy.sum(numpy.log(L.diagonal(axis1=1, axis2=2)), axis=1)
        return A_inv, ln_det_A