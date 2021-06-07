##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for conditional Gaussian densities, that can be seen as          #
# operators.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

import numpy
#from densities import GaussianDensity
import factors

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
    
    def affine_joint_transformation(self, p_x: 'GaussianDensity') -> 'GaussianDensity':
        """ Returns the joint density 
        
            p(x,y) = p(y|x)p(x),
            
            where p(y|x) is the object itself.
            
        :param p_x: GaussianDensity
            Marginal density over x.
        
        :return: GaussianDensity
            The joint density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of multiple marginals
        # and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError('The combination of combining multiple marginals with multiple conditionals is not implemented.')
        from densities import GaussianDensity
        R = p_x.R * self.R
        D_xy = p_x.D + self.Dy
        # Mean
        mu_x = numpy.tile(p_x.mu[None], (self.R, 1, 1,)).reshape((R, p_x.D))
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        mu_xy = numpy.hstack([mu_x, mu_y])
        # Sigma
        Sigma_x = numpy.tile(p_x.Sigma[None], (self.R, 1, 1, 1)).reshape(R, p_x.D, p_x.D)
        MSigma_x = numpy.einsum('abc,dce->adbe', self.M, p_x.Sigma) # [R1,R,Dy,D]
        MSigmaM = numpy.einsum('abcd,aed->abce', MSigma_x, self.M)
        Sigma_y = (self.Sigma[:,None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        C_xy = MSigma_x.reshape((R, self.Dy, p_x.D))
        Sigma_xy = numpy.empty((R, D_xy, D_xy))
        Sigma_xy[:,:p_x.D,:p_x.D] = Sigma_x
        Sigma_xy[:,p_x.D:,p_x.D:] = Sigma_y
        Sigma_xy[:,p_x.D:,:p_x.D] = C_xy
        Sigma_xy[:,:p_x.D,p_x.D:] = numpy.swapaxes(C_xy, 1, 2)
        # Lambda
        Lambda_y = numpy.tile(self.Lambda[:,None], (1, p_x.R, 1, 1)).reshape((R, self.Dy, self.Dy))
        Lambda_yM = numpy.einsum('abc,abd->acd', self.Lambda, self.M) # [R1,Dy,D]
        MLambdaM = numpy.einsum('abc,abd->acd', self.M, Lambda_yM)
        Lambda_x = (p_x.Lambda[None] + MLambdaM[:,None]).reshape((R, p_x.D, p_x.D))
        L_xy = numpy.tile(-Lambda_yM[:,None], (1, p_x.R, 1, 1)).reshape((R, self.Dy, p_x.D))
        Lambda_xy = numpy.empty((R, D_xy, D_xy))
        Lambda_xy[:,:p_x.D,:p_x.D] = Lambda_x
        Lambda_xy[:,p_x.D:,p_x.D:] = Lambda_y
        Lambda_xy[:,p_x.D:,:p_x.D] = L_xy
        Lambda_xy[:,:p_x.D,p_x.D:] = numpy.swapaxes(L_xy, 1, 2)
        # Log determinant
        if p_x.D > self.Dy:
            CLambda_x = numpy.einsum('abcd,bde->abce', MSigma_x, p_x.Lambda) # [R1,R,Dy,D]
            CLambdaC = numpy.einsum('abcd,abed->abce', CLambda_x, MSigma_x) # [R1,R,Dy,Dy]
            delta_ln_det = numpy.linalg.slogdet(Sigma_y[:,None] - CLambdaC)[1].reshape((R,))
            ln_det_Sigma_xy = p_x.ln_det_Sigma + delta_ln_det
        else:
            Sigma_yL = numpy.einsum('abc,acd->abd', self.Sigma, -Lambda_yM) # [R1,Dy,Dy] x [R1, Dy, D] = [R1, Dy, D]
            LSigmaL = numpy.einsum('abc,abd->acd', -Lambda_yM, Sigma_yL) # [R1, Dy, D] x [R1, Dy, D] = [R1, D, D]
            LSigmaL = numpy.tile(LSigmaL[:,None], (1, p_x.R)).reshape((R, p_x.D, p_x.D))
            delta_ln_det = numpy.linalg.slogdet(Lambda_x - LSigmaL)[1]
            ln_det_Sigma_xy = -(numpy.tile(self.ln_det_Lambda[:,None], (1, p_x.R)).reshape((R,)) + delta_ln_det)
        return GaussianDensity(Sigma_xy, mu_xy, Lambda_xy, ln_det_Sigma_xy)
    
    def affine_marginal_transformation(self, p_x: 'ConditionalGaussianDensity') -> 'GaussianDensity':
        """ Returns the marginal density p(y) given  p(y|x) and p(x), 
            where p(y|x) is the object itself.
            
        :param p_x: GaussianDensity
            Marginal density over x.
        
        :return: GaussianDensity
            The marginal density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of multiple marginals
        # and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError('The combination of combining multiple marginals with multiple conditionals is not implemented.')
        from densities import GaussianDensity
        R = p_x.R * self.R
        # Mean
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        # Sigma
        MSigma_x = numpy.einsum('abc,dce->adbe', self.M, p_x.Sigma) # [R1,R,Dy,D]
        MSigmaM = numpy.einsum('abcd,aed->abce', MSigma_x, self.M)
        Sigma_y = (self.Sigma[:,None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        return GaussianDensity(Sigma_y, mu_y)
    
    def affine_conditional_transformation(self, p_x: 'ConditionalGaussianDensity') -> 'ConditionalGaussianDensity':
        """ Returns the conditional density p(x|y), given p(y|x) and p(x),           
            where p(y|x) is the object itself.
            
        :param p_x: GaussianDensity
            Marginal density over x.
        
        :return: GaussianDensity
            The marginal density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of multiple marginals
        # and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError('The combination of combining multiple marginals with multiple conditionals is not implemented.')
        R = p_x.R * self.R
        # TODO: Could be flexibly made more effiecient here.
        # Marginal Sigma y
        # MSigma_x = numpy.einsum('abc,dce->adbe', self.M, p_xSigma) # [R1,R,Dy,D]
        # MSigmaM = numpy.einsum('abcd,aed->abce', MSigma_x, self.M)
        # Sigma_y = (self.Sigma[:,None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        # Lambda_y, ln_det_Sigma_y = p_x.invert_matrix(Sigma_y)
        # Lambda
        Lambda_yM = numpy.einsum('abc,abd->acd', self.Lambda, self.M) # [R1,Dy,D]
        MLambdaM = numpy.einsum('abc,abd->acd', self.M, Lambda_yM)
        Lambda_x = (p_x.Lambda[None] + MLambdaM[:,None]).reshape((R, p_x.D, p_x.D))
        # Sigma
        Sigma_x, ln_det_Lambda_x = p_x.invert_matrix(Lambda_x)
        # M_x
        M_Lambda_y = numpy.einsum('abc,abd->acd', self.M, self.Lambda) # [R1, D, Dy]
        M_x = numpy.einsum('abcd,ade->abce', Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), M_Lambda_y) #[R1, R, D, Dy]
        b_x = - numpy.einsum('abcd,ad->abc', M_x, self.b) # [R1, R, D, Dy] x [R1, Dy] = [R1, R, D]
        b_x += numpy.einsum('abcd,bd->abc', Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), p_x.nu).reshape((R, p_x.D))
        M_x = M_x.reshape((R, p_x.D, self.Dy))
        
        return ConditionalGaussianDensity(M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x)
    
    
class LSEMGaussianConditional(ConditionalGaussianDensity):
    
    def __init__(self, M: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray, 
                 Sigma: numpy.ndarray=None, Lambda: numpy.ndarray=None, 
                 ln_det_Sigma: numpy.ndarray=None):
        """ A conditional Gaussian density, with a linear squared exponential mean (LSEM) function,
            
            p(y|x) = N(mu(x), Sigma)
            
            with the conditional mean function mu(x) = M phi(x) + b. 
            phi(x) is a feature vector of the form
            
            phi(x) = (1,x_1,...,x_m,k(h_1(x)),...,k(h_n(x))),
            
            with
            
            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.
            
            Note, that the affine transformations will be approximated via moment matching.
            
            :param M: numpy.ndarray [1, Dy, Dphi]
                Matrix in the mean function.
            :param b: numpy.ndarray [1, Dy]
                Vector in the conditional mean function.
            :param W: numpy.ndarray [Dphi, Dx + 1]
                Parameters for linear mapping in the nonlinear functions
            :param Sigma: numpy.ndarray [1, Dy, Dy]
                The covariance matrix of the conditional. (Default=None)
            :param Lambda: numpy.ndarray [1, Dy, Dy] or None
                Information (precision) matrix of the Gaussians. (Default=None)
            :param ln_det_Sigma: numpy.ndarray [1] or None
                Log determinant of the covariance matrix. (Default=None)
        """
        super().__init__(M, b, Sigma, Lambda, ln_det_Sigma)
        self.w0 = W[:,0]
        self.W = W[:,1:]
        self.Dx = self.W.shape[1]
        self.Dk = self.W.shape[0]
        self.Dphi = self.Dk + self.Dx
        self._setup_phi()
        
    def _setup_phi(self):
        """ Sets up the non-linear kernel function in phi(x).
        """
        v = self.W
        nu = self.W * self.w0[:,None]
        ln_beta = - .5 * self.w0 ** 2
        self.k_func = factors.OneRankFactor(v=v, nu=nu, ln_beta=ln_beta)
        
    def evaluate_phi(self, x: numpy.ndarray):
        """ Evaluates the phi
        
        phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).
        
        :param x: numpy.ndarray [N, Dx]
            Points where f should be evaluated.
            
        :return: numpy.ndarray [N, Dphi]
            Deature vector.
        """
        N = x.shape[0]
        phi_x = numpy.empty((N, self.Dphi))
        phi_x[:,:self.Dx] = x
        phi_x[:,self.Dx:] = self.k_func.evaluate(x).T
        return phi_x
    
    def get_conditional_mu(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Computes the conditional mu function
        
            mu(x) = mu(x) = M phi(x) + b
            
        :param x: numpy.ndarray [N, Dx]
            Instances, the mu should be conditioned on.
        
        :return: numpy.ndarray [1, N, Dy]
            Conditional means.
        """
        phi_x = self.evaluate_phi(x)
        mu_y = numpy.einsum('ab,cb->ca', self.M[0], phi_x) + self.b[0][None]
        return mu_y
    
    def get_expected_moments(self, p_x: 'GaussianDensity') -> numpy.ndarray:
        """ Computes the expected covariance
        
            Sigma_y = E[yy'] - E[y]E[y]'
            
        :param p_x: GaussianDensity
            The density which we average over.

        :return: numpy.ndarray [p_R, Dy, Dy]
            Returns the expected mean
        """
        
        #### E[f(x)] ####
        # E[x] [R, Dx]
        Ex = p_x.integrate('x')
        # E[k(x)] [R, Dphi - Dx]
        p_k = p_x.multiply(self.k_func)
        Ekx = p_k.integrate().reshape((p_x.R, self.Dphi - self.Dx))
        # E[f(x)]
        Ef = numpy.concatenate([Ex, Ekx], axis=1)
        
        #### E[f(x)f(x)'] ####
        Eff = numpy.empty([p_x.R, self.Dphi, self.Dphi])
        # Linear terms E[xx']
        Eff[:,:self.Dx,:self.Dx] = p_x.integrate('xx')
        # Cross terms E[x k(x)']
        Ekx = p_k.integrate('x').reshape((p_x.R, self.Dk, self.Dx))
        Eff[:,:self.Dx,self.Dx:] = numpy.swapaxes(Ekx, axis1=1, axis2=2)
        Eff[:,self.Dx:,:self.Dx] = Ekx
        # kernel terms E[k(x)k(x)']
        Ekk = p_x.multiply(self.k_func).multiply(self.k_func).integrate().reshape((p_x.R, self.Dk, self.Dk))
        Eff[:,self.Dx:,self.Dx:] = Ekk
        
        ### mu_y = E[mu(x)] = ME[f(x)] + b ###
        mu_y = numpy.einsum('ab,cb->ca', self.M[0], Ef) + self.b[0][None]
        
        ### Sigma_y = E[yy'] - mu_ymu_y' = Sigma + E[mu(x)mu(x)'] - mu_ymu_y'
        #                                = Sigma + ME[f(x)f(x)']M' + bE[f(x)']M' + ME[f(x)]b' + bb' - mu_ymu_y'
        Sigma_y = numpy.tile(self.Sigma, (p_x.R, 1, 1))
        Sigma_y += numpy.einsum('ab,cbd->cad', self.M[0], numpy.einsum('abc,dc->abd', Eff, self.M[0]))
        MEfb = numpy.einsum('ab,c->abc', numpy.einsum('ab,cb->ca', self.M[0], Ef), self.b[0])
        Sigma_y += MEfb + numpy.swapaxes(MEfb, axis1=1, axis2=2)
        Sigma_y += (self.b[0,None] * self.b[0,:,None])[None]
        Sigma_y -= mu_y[:,None] * mu_y[:,:,None]
        return mu_y, Sigma_y
    
    def get_expected_cross_terms(self, p_x: 'GaussianDensity') -> numpy.ndarray:
        """ Computes
        
            E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M f(x) + b)x' p(x) dx
            
        :param p_x: GaussianDensity
            The density which we average over.

        :return: numpy.ndarray [p_R, Dx, Dy]
            Returns the cross expectations.
        """
        
        # E[xx']
        Exx = p_x.integrate('xx')
        # E[k(x)x']
        Ekx = p_x.multiply(self.k_func).integrate('x').reshape((p_x.R, self.Dk, self.Dx))
        # E[f(x)x']
        Ef_x = numpy.concatenate([Exx, Ekx], axis=1)
        # M E[f(x)x']
        MEf_x = numpy.einsum('ab,cbd->cad', self.M[0], Ef_x)
        # bE[x']
        bEx = self.b[0][None,:,None] * p_x.integrate('x')[:,None]
        # E[yx']
        Eyx = MEf_x + bEx
        return Eyx
        
    def affine_joint_transformation(self, p_x: 'GaussianDensity') -> 'GaussianDensity':
        """ Gets an approximation of the joint density
        
            p(x,y) ~= N(mu_{xy},Sigma_{xy}),
            
        The mean is given by
            
            mu_{xy} = (mu_x, mu_y)'
            
        with mu_y = E[mu_y(x)]. The covariance is given by
            
            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').
                          
        :param p_x: GaussianDensity
            Marginal Gaussian density over x.
        
        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        from densities import GaussianDensity
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:,:,None] * mu_x[:,None]
        mu_xy = numpy.concatenate([mu_x, mu_y], axis=1)
        Sigma_xy = numpy.empty((p_x.R, self.Dy + self.Dx, self.Dy + self.Dx))
        Sigma_xy[:,:self.Dx,:self.Dx] = p_x.Sigma
        Sigma_xy[:,self.Dx:,:self.Dx] = cov_yx
        Sigma_xy[:,:self.Dx,self.Dx:] = numpy.swapaxes(cov_yx, axis1=1, axis2=2)
        Sigma_xy[:,self.Dx:,self.Dx:] = Sigma_y
        p_xy = GaussianDensity(Sigma=Sigma_y, mu=mu_y)
        return p_xy
    
    def affine_conditional_transformation(self, p_x: 'GaussianDensity') -> 'ConditionalGaussianDensity':
        """ Gets an approximation of the joint density via moment matching
        
            p(x|y) ~= N(mu_{x|y},Sigma_{x|y}),
    
        :param p_x: GaussianDensity
            Marginal Gaussian density over x.
        
        :return: ConditionalDensity
            Returns the conditional density of x given y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = self.invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:,:,None] * mu_x[:,None]
        M_new = numpy.einsum('abc,abd->acd', cov_yx, Lambda_y)
        b_new = mu_x - numpy.einsum('abc,ac->ab', M_new, mu_y)
        Sigma_new = p_x.Sigma - numpy.einsum('abc,acd->abd', M_new, cov_yx)
        cond_p_xy = ConditionalGaussianDensity(M=M_new, b=b_new, Sigma=Sigma_new)
        return cond_p_xy
        
    def affine_marginal_transformation(self, p_x: 'GaussianDensity') -> 'GaussianDensity':
        """ Gets an approximation of the marginal density
        
            p(y) ~= N(mu_y,Sigma_y),
            
        The mean is given by
            
            mu_y = E[mu_y(x)]. 
            
        The covariance is given by
            
            Sigma_y = E[yy'] - mu_ymu_y'.
                          
        :param p_x: GaussianDensity
            Marginal Gaussian density over x.
        
        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        from densities import GaussianDensity
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = GaussianDensity(Sigma=Sigma_y, mu=mu_y)
        return p_y
