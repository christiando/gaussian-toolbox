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
from densities import GaussianDensity

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
        R = p_x.R * self.R
        D_xy = p_x.D + self.Dyp_x
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
        
        return conditionals.ConditionalGaussianDensity(M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x)