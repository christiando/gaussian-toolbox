##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the class to fit observation models that can be incroporated in the SSM-framwork.  #
#                                                                                                #
# Implemented so far:                                                                            #
#       + LinearObservationModel (Gaussian Emission)                                             #
#       + HCCovObservationModel (Gaussian Emission with state dependent covariance)              #
# Yet to be implemented:                                                                         #
#       - LSEMObservationModel (Gaussian Emission with non linear mean)                          #
#       - HCCovLSEMObservationModel (Gaussian Emission with non linear mean and state dependent  #
#                                    covariance)                                                 #
#       - BernoulliObservationModel (Emissions for binary data)                                  #
#       - PoissonObservationModel (Emissions for count data)                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

import scipy
from scipy.optimize import minimize
from autograd import numpy
from autograd import value_and_grad
import sys
sys.path.append('../src/')
import densities, conditionals, factors

    
def logcosh(x):
    # s always has real part >= 0
    s = numpy.sign(x) * x
    p = numpy.exp(-2 * s)
    return s + numpy.log1p(p) - numpy.log(2)

class ObservationModel:
    
    def __init__(self):
        """ This is the template class for observation models in state space models. 
        Basically these classes should contain all functionality for the mapping between 
        the latent variables z, and observations x, i.e. p(x_t|z_t). The object should 
        have an attribute `emission_density`, which is be a `ConditionalDensity`. 
        Furthermore, it should be possible to optimize hyperparameters, when provided 
        with a density over the latent space.
        """
        self.emission_density = None
    
    def filtering(self, prediction_density: 'GaussianDensity', x_t: numpy.ndarray, **kwargs) -> 'GaussianDensity':
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
        
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        raise NotImplementedError('Filtering for observation model not implemented.')
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray, **kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        raise NotImplementedError('Hyperparameter updates for observation model not implemented.')
        
    
    def evalutate_llk(self, p_z: 'GaussianDensity', X: numpy.ndarray, **kwargs) -> numpy.ndarray:
        """ Computes the log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: numpy.ndarray [T, Dx]
            Observations.
        """
        raise NotImplementedError('Log likelihood not implemented for observation model.')
        

class LinearObservationModel(ObservationModel):
    
    def __init__(self, Dx: int, Dz: int, noise_x: float=1.):
        """ This class implements a linear observation model, where the observations are generated as
        
            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx).
            
        :param Dx: int
            Dimensionality of observations.
        :param Dz: int
            Dimensionality of latent space.
        :param noise_x: float
            Intial isoptropic std. on the observations.
        """
        self.Dx, self.Dz = Dx, Dz
        if Dx == Dz:
            self.C = numpy.eye(Dx)
        else:
            self.C = numpy.random.randn(Dx, Dz)
        self.d = numpy.zeros(Dx)
        self.Qx = noise_x ** 2 * numpy.eye(self.Dx)
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([self.C]), 
                                                                        numpy.array([self.d]), 
                                                                        numpy.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
        
    def pca_init(self, X: numpy.ndarray, smooth_window: int=10):
        """ Sets the model parameters to an educated initial guess, based on principal component analysis.
            More specifically `d` is set to the mean of the data and `C` to the first principal components 
            of the (smoothed) data. The covariance `Qx` is set to the empirical covariance of the residuals.
        
        :param X: numpy.ndarray [T, Dx]
            Data.
        :param smoothed_window: int
            Width of the box car filter data are smoothed with. (Default=10)
        """
        self.d = numpy.mean(X, axis=0)
        T = X.shape[0]
        X_smoothed = numpy.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = numpy.convolve(X[:,i], 
                                             numpy.ones(smooth_window) / smooth_window, 
                                             mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(numpy.dot((X_smoothed-self.d[None]).T, 
                                                         X_smoothed-self.d[None]), 
                                               eigvals=(self.Dx-self.Dz, self.Dx-1))
        self.C =  eig_vecs * eig_vals / T
        z_hat = numpy.dot(numpy.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - numpy.dot(z_hat, self.C.T) - self.d
        self.Qx = numpy.dot(delta_X.T, delta_X)
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([self.C]), 
                                                                        numpy.array([self.d]), 
                                                                        numpy.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
        
    def filtering(self, prediction_density: 'GaussianDensity', x_t: numpy.ndarray, **kwargs) -> 'GaussianDensity':
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
            Observation.
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        # p(z_t| x_t, x_{1:t-1})
        p_z_given_x = self.emission_density.affine_conditional_transformation(prediction_density)
        # Condition on x_t
        cur_filter_density = p_z_given_x.condition_on_x(x_t)
        return cur_filter_density
    
    def gappy_filtering(self, prediction_density: 'GaussianDensity', x_t: numpy.ndarray, **kwargs) -> 'GaussianDensity':
        """ Here the filtering density is calculated for incomplete data. Not observed values should be nans.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
            Observation, where unobserved dimensions are filled with NANs.
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        # In case all data are unobserved
        if numpy.alltrue(numpy.isnan(x_t[0])):
            return prediction_density
        # In case all data are observed
        elif numpy.alltrue(numpy.logical_not(numpy.isnan(x_t[0]))):
            cur_filter_density = self.filtering(prediction_density, x_t)
            return cur_filter_density
        # In case we have only partial observations
        else:
            observed_dims = numpy.where(numpy.logical_not(numpy.isnan(x_t[0])))[0]
            # p(z_t, x_t| x_{1:t-1})
            p_zx = self.emission_density.affine_joint_transformation(prediction_density)
            # p(z_t, x_t (observed) | x_{1:t-1})
            marginal_dims = numpy.concatenate([numpy.arange(self.Dz), self.Dz + observed_dims])
            p_zx_observed = p_zx.get_marginal(marginal_dims)
            # p(z_t | x_t (observed), x_{1:t-1})
            conditional_dims = numpy.arange(self.Dz,self.Dz + len(observed_dims))
            p_z_given_x_observed = p_zx_observed.condition_on(conditional_dims)
            cur_filter_density = p_z_given_x_observed.condition_on_x(x_t[:,observed_dims])
            return cur_filter_density
        
    def gappy_data_density(self, p_z: 'GaussianDensity', x_t: numpy.ndarray, **kwargs):
        """ Here the data density is calculated for incomplete data. Not observed values should be nans.
        
         p(x_t) = p(x_t|z_t)p(z_t) dz_t
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
            Observation, where unobserved dimensions are filled with NANs.
        :return: (numpy.ndarray, numpy.ndarray)
            Mean and variance of unobserved entries.
        """
        # In case all data are unobserved
        if numpy.alltrue(numpy.isnan(x_t[0])):
            p_x = self.emission_density.affine_marginal_transformation(p_z)
            return p_x.mu[0], numpy.sqrt(p_x.Sigma[0].diagonal(axis1=-1, axis2=-2))
        # In case all data are observed
        elif numpy.alltrue(numpy.logical_not(numpy.isnan(x_t[0]))):
            return numpy.array([]), numpy.array([])
        # In case we have only partial observations
        else:
            observed_dims = numpy.where(numpy.logical_not(numpy.isnan(x_t[0])))[0]
            # Density over unobserved variables
            p_x = self.emission_density.affine_marginal_transformation(p_z)
            p_ux_given_ox = p_x.condition_on(observed_dims)
            p_ux = p_ux_given_ox.condition_on_x(x_t[:,observed_dims])
            return p_ux.mu[0], numpy.sqrt(p_ux.Sigma.diagonal(axis1=-1, axis2=-2))
    
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray, **kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        self.update_C(smoothing_density, X)
        self.update_d(smoothing_density, X)
        self.update_Qx(smoothing_density, X)
        self.update_emission_density()
           
    def update_Qx(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the covariance of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        T = X.shape[0]
        A = -self.C
        a_t = X - self.d[None]
        Exx = numpy.zeros((self.Dx, self.Dx))
        for t in range(1, T+1):
            cur_smooth_density = smoothing_density.slice([t])
            Exx += cur_smooth_density.integrate('Ax_aBx_b_outer', A_mat=A, 
                                                a_vec=a_t[t-1], B_mat=A, 
                                                b_vec=a_t[t-1])[0]
        self.Qx = Exx / T
        
    def update_C(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the transition matrix of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        Ezz = numpy.sum(smoothing_density.integrate('xx')[1:], axis=0)
        Ez = smoothing_density.integrate('x')[1:]
        zx = numpy.sum(Ez[:,:,None] * (X[:,None] - self.d[None,None]), axis=0)
        self.C = numpy.linalg.solve(Ezz, zx).T
        
    def update_d(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the transition offset of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """
        self.d = numpy.mean(X - numpy.dot(smoothing_density.mu[1:], self.C.T), axis=0)
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([self.C]),
                                                                        numpy.array([self.d]),
                                                                        numpy.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
        
        
    def evaluate_llk(self, p_z: 'GaussianDensity', X: numpy.ndarray, **kwargs) -> float:
        """ Computes the log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: numpy.ndarray [T, Dx]
            Observations.
            
        :return: float
            Log likelihood.
        """
        T = X.shape[0]
        llk = 0
        p_x = self.emission_density.affine_marginal_transformation(p_z)
        for t in range(0,T):
            cur_p_x = p_x.slice([t])
            llk += cur_p_x.evaluate_ln(X[t:t+1])[0,0]
        return llk
        
        
        
class HCCovObservationModel(LinearObservationModel):
    
    
    def __init__(self, Dx: int, Dz: int, Du: int, noise_x: float=1.):
        """ This class implements a linear observation model, where the observations are generated as
        
            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx(z_t)),
            
        where 
        
            Qx(z) = sigma_x^2 I + \sum_i U_i D_i(z) U_i',
            
        with D_i(z) = 2 * beta_i * cosh(h_i(z)) and h_i(z) = w_i'z + b_i
            
        :param Dx: int
            Dimensionality of observations.
        :param Dz: int
            Dimensionality of latent space.
        :param noise_x: float
            Intial isoptropic std. on the observations.
        """
        self.Dx, self.Dz, self.Du = Dx, Dz, Du
        if Dx == Dz:
            self.C = numpy.eye(Dx)
        else:
            self.C = numpy.random.randn(Dx, Dz)
        self.d = numpy.zeros(Dx)
        self.U = numpy.eye(Dx)[:,:Du]
        self.W = 1e-4 * numpy.random.randn(self.Du, self.Dz + 1)
        self.beta = numpy.ones(self.Du)
        self.sigma_x = noise_x
        self.emission_density = conditionals.HCCovGaussianConditional(M = numpy.array([self.C]), 
                                                                      b = numpy.array([self.d]), 
                                                                      sigma_x = self.sigma_x,
                                                                      U = self.U,
                                                                      W = self.W,
                                                                      beta = self.beta)
        
        
    def pca_init(self, X: numpy.ndarray, smooth_window: int=10):
        """ Sets the model parameters to an educated initial guess, based on principal component analysis.
            More specifically `d` is set to the mean of the data and `C` to the first principal components 
            of the (smoothed) data. 
            Then `U` is initialized with the first pricinpal components of the empirical covariance of the 
            residuals.
        
        :param X: numpy.ndarray [T, Dx]
            Data.
        :param smoothed_window: int
            Width of the box car filter data are smoothed with. (Default=10)
        """
        self.d = numpy.mean(X, axis=0)
        T = X.shape[0]
        X_smoothed = numpy.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = numpy.convolve(X[:,i], 
                                             numpy.ones(smooth_window) / smooth_window, 
                                             mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(numpy.dot((X_smoothed-self.d[None]).T, 
                                                         X_smoothed-self.d[None]), 
                                               eigvals=(self.Dx-self.Dz, self.Dx-1))
        self.C =  eig_vecs * eig_vals / T
        z_hat = numpy.dot(numpy.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - numpy.dot(z_hat, self.C.T) - self.d
        cov = numpy.dot(delta_X.T, delta_X)
        self.U = scipy.linalg.eigh(cov, eigvals=(self.Dx-self.Du, self.Dx-1))[1]
        self.emission_density = conditionals.HCCovGaussianConditional(M = numpy.array([self.C]), 
                                                                      b = numpy.array([self.d]), 
                                                                      sigma_x = self.sigma_x,
                                                                      U = self.U,
                                                                      W = self.W,
                                                                      beta = self.beta)
        
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray, **kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        self.update_C(smoothing_density, X)
        self.update_d(smoothing_density, X)
        self.update_U(smoothing_density, X)
        self.update_sigma_beta_W(smoothing_density, X)
        self.update_emission_density()
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.emission_density = conditionals.HCCovGaussianConditional(M = numpy.array([self.C]), 
                                                                      b = numpy.array([self.d]), 
                                                                      sigma_x = self.sigma_x,
                                                                      U = self.U,
                                                                      W = self.W,
                                                                      beta = self.beta)
        
    def update_C(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ Updates the `C` by maximizing the Q-function numerically (by L-BFGS-B).
        
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: numpy.ndarray [T, Dx]
            Data.
        """
        T = X.shape[0]
        C_old = numpy.copy(self.C)
        phi = smoothing_density.slice(range(1,T+1))
        intD_inv_z, intD_inv_zz = numpy.zeros((self.Du, T, self.Dz)), numpy.zeros((self.Du, self.Dz, self.Dz))
        for iu in range(self.Du):
            intD_inv_z_i, intD_inv_zz_i = self.get_lb_i(iu, phi, X, update='C')
            intD_inv_z[iu] = intD_inv_z_i
            intD_inv_zz[iu] += numpy.sum(intD_inv_zz_i, axis=0)
        Ez = phi.integrate('x')
        Ezz = numpy.sum(phi.integrate('xx'), axis=0)
        Ezx_d = numpy.einsum('ab,ac->bc', Ez, X - self.d)
        UU = numpy.einsum('ab,cb->bac', self.U, self.U)
        intD_inv_zx_d = numpy.einsum('abc,bd->adc', intD_inv_z, X - self.d)
        
        def Q_C_func(params: numpy.ndarray) -> (float, numpy.ndarray):
            """ Function computing the terms of the (negative) Q-function that depend on `C` and the gradients.
            """
            C = numpy.reshape(params, (self.Dx, self.Dz))
            tr_CEzx_d = numpy.trace(numpy.dot(C, Ezx_d))
            tr_CC_Ezz = numpy.trace(numpy.dot(numpy.dot(C.T, C), Ezz))
            tr_uu_CC_Dinv_zx_d = numpy.sum(numpy.trace(numpy.einsum('abc,acd->abd', UU, numpy.einsum('ab,cdb->cad', C, intD_inv_zx_d)), axis1=1, axis2=2))
            CD_inv_zz = numpy.einsum('ab,cbd->cad', C, intD_inv_zz)
            CD_inv_zzC = numpy.einsum('abc,dc->abd', CD_inv_zz, C)
            uCD_inv_zzCu = numpy.sum(numpy.einsum('ab,ba->b',self.U, numpy.einsum('abc,ca->ab', CD_inv_zzC, self.U)))
            Q_C = 2 * tr_CEzx_d - tr_CC_Ezz - 2 * tr_uu_CC_Dinv_zx_d + uCD_inv_zzCu #- 2 * tr_uu_CC_Dinv_zx_d + uCD_inv_zzCu
            Q_C /= 2 * self.sigma_x ** 2
            C_Ezz = numpy.dot(C, Ezz)
            UU_C_Dinv_zz = numpy.sum(numpy.einsum('abc,abd->acd', UU, CD_inv_zz), axis=0)
            UU_Dinv_zx_d = numpy.sum(numpy.einsum('abc,abd->acd', UU, intD_inv_zx_d), axis=0)
            dQ_C = Ezx_d.T - UU_Dinv_zx_d + UU_C_Dinv_zz - C_Ezz #- UU_Dinv_zx_d + UU_C_Dinv_zz
            dQ_C /= self.sigma_x ** 2
            return -Q_C, -dQ_C.flatten() 
        
        x0 = self.C.flatten()
        result = minimize(Q_C_func, x0, method='L-BFGS-B', jac=True)
        if not result.success:
            print(result)
            print('C did not converge!! Falling back to old C.')
            self.C = C_old
        else:
            self.C = result.x.reshape(self.Dx, self.Dz)
            
    def update_d(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ Updates the `d` by maximizing the Q-function analytically.
        
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: numpy.ndarray [T, Dx]
            Data.
        """
        T = X.shape[0]
        phi = smoothing_density.slice(range(1,T+1))
        intD_inv, intD_inv_z = numpy.zeros((self.Du, T)), numpy.zeros((self.Du, T, self.Dz))
        for iu in range(self.Du):
            intD_inv_i, intD_inv_z_i = self.get_lb_i(iu, phi, X, update='d')
            intD_inv[iu] = intD_inv_i
            intD_inv_z[iu] += intD_inv_z_i
        Ez = phi.integrate('x')
        CEz = numpy.dot(self.C, numpy.sum(Ez,axis=0))
        UU = numpy.einsum('ab,cb->bac', self.U, self.U)
        A = numpy.eye(self.Dx) * T - numpy.sum(numpy.sum(intD_inv, axis=1)[:,None,None] * UU, axis=0)
        sum_X = numpy.sum(X, axis=0)
        intDinv_X_UU = numpy.sum(numpy.einsum('ab,abc->ac', numpy.einsum('ab,bc->ac',intD_inv[:,], X), UU), axis=0)
        UU_C_intDinv_z = numpy.sum(numpy.einsum('abc,ac->ab', UU, numpy.einsum('ab,cb->ca', self.C, numpy.sum(intD_inv_z,axis=1))), axis=0)
        b = sum_X - intDinv_X_UU - CEz + UU_C_intDinv_z
        self.d = numpy.linalg.solve(A,b)
    
        
    def update_sigma_beta_W(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ Updates the `sigma_x`, `beta`, and `W` by maximizing the Q-function numerically (by L-BFGS-B).
        
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: numpy.ndarray [T, Dx]
            Data.
        """
        x0 = numpy.concatenate([numpy.array([numpy.log(self.sigma_x ** 2)]), numpy.log(self.beta), self.W.flatten()])
        bounds = [(None, 10)] + [(-10, 10)] * self.Du + [(None,None)] * (self.Du * (self.Dz + 1))
        objective = lambda x: self.parameter_optimization_sigma_beta_W(x, smoothing_density, X)
        result = minimize(objective, x0, jac=True, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': 100})
        #print(result)
        #if not result.success:
        #    raise RuntimeError('Sigma, beta, W did not converge!!')
        self.sigma_x = numpy.exp(.5*result.x[0])
        self.beta = numpy.exp(result.x[1:self.Du + 1])
        self.W = result.x[self.Du + 1:].reshape((self.Du, self.Dz+1))
        
    def _U_lagrange_func(self, x, R):
        
        U = x[:self.Du * self.Dx].reshape((self.Dx, self.Du))
        lagrange_multipliers = x[self.Du * self.Dx:].reshape((self.Du, self.Du))
        dL_dU = -numpy.einsum('abc,ca->ab', R, self.U) + numpy.dot(U, lagrange_multipliers).T
        dL_dmultipliers = numpy.dot(U.T, U) - numpy.eye(self.Du)
        objective = numpy.sum(dL_dU ** 2) + numpy.sum(dL_dmultipliers ** 2)
        return objective
        
        
    def update_U(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        T = X.shape[0]
        x0 = numpy.zeros(self.Du * self.Dx + self.Du * self.Du)
        x0[:self.Du * self.Dx] = self.U.flatten()
        R = numpy.empty([self.Du, self.Dx, self.Dx])
        phi = smoothing_density.slice(range(1,T+1))
        for iu in range(self.Du):
            R[iu] = self.get_lb_i(iu, phi, X, update='U')
            R[iu] /= numpy.amax(R[iu])
        objective = lambda x: self._U_lagrange_func(x, R)
        result = minimize(value_and_grad(objective), x0,
                          method='L-BFGS-B', jac=True, options={'disp': True})
        self.U = result.x[:self.Du * self.Dx].reshape((self.Dx, self.Du))
        
        
    def update_U2(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ Updates the `U` by maximizing the Q-function. One component at a time is updated analytically.
        
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: numpy.ndarray [T, Dx]
            Data.
        """
        converged = False
        T = X.shape[0]
        phi = smoothing_density.slice(range(1,T+1))
        R = numpy.empty([self.Du, self.Dx, self.Dx])
        for iu in range(self.Du):
            R[iu] = self.get_lb_i(iu, phi, X, update='U')
            R[iu] /= numpy.amax(R[iu])
        num_iter = 0
        Q_u = numpy.sum(numpy.einsum('ab,ba->a', self.U, numpy.einsum('abc,ca->ab', R, self.U)))
        while not converged and num_iter < 50:
            #print(Q_u)
            U_old = numpy.copy(self.U)
            for iu in range(self.Du):
                U_not_i = numpy.delete(self.U, [iu], axis=1)
                V = self.partial_gs(U_not_i)
                #A = numpy.hstack([U_not_i, numpy.eye(self.Dx)[:,-self.Du-1:]])
                #V_full = numpy.linalg.qr(A)[0]
                #print(numpy.dot(V_full[:,:-self.Du-1].T, U_not_i))
                #V = V_full[:,-self.Du-1:]
                VRV = numpy.dot(numpy.dot(V.T, R[iu]), V)
                VRV /= numpy.amax(VRV)
                #alpha = scipy.linalg.eigh(VRV, eigvals=(VRV.shape[0]-1,VRV.shape[0]-1))[1]
                alpha = numpy.linalg.eig(VRV)[1][:,:1]
                u_new = numpy.dot(V, alpha)[:,0]
                U_new = numpy.copy(self.U)
                U_new[:,iu] = u_new
                if numpy.allclose(numpy.dot(U_new.T, U_new), numpy.eye(self.Du)):
                    self.U[:,iu] = u_new
                else:
                    print('Warning: U not orthonormal')
            Q_u_old = Q_u
            Q_u = numpy.sum(numpy.einsum('ab,ba->a', self.U, numpy.einsum('abc,ca->ab', R, self.U)))
            converged = (Q_u - Q_u_old) < 1e-4
            num_iter += 1
        if (Q_u - Q_u_old) < 0:
            self.U = U_old
        
    ####################### FUNCTIONS FOR OPTIMIZING sigma_x, beta, and W ################################################
    def update_parameters_sigma_beta_W(self, params):
        """ Mapping function from scipy-vector to model fields.
        
        :param params: numpy.ndarray [1+Du+Du*(Dz + 1)]
            Parameters in vector form. (log(sigma_x), log(beta), W)
        """
        self.sigma_x = numpy.exp(.5 * params[0])
        self.beta = numpy.exp(params[1:self.Du + 1])
        self.W = params[self.Du + 1:].reshape((self.Du, self.Dz + 1))
        
    def parameter_optimization_sigma_beta_W(self, params: numpy.ndarray, 
                                          smoothing_density: 'GaussianDensity', 
                                          X: numpy.ndarray):
        """ Computes (negative) Q-function (only terms depending on `sigma_x`, `beta`, and `W`) and the 
        corresponding gradients.
        
        :param params: numpy.ndarray [1+Du+Du*(Dz + 1)]
            Parameters in vector form. (log(sigma_x), log(beta), W)
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: numpy.ndarray [T, Dx]
            Data.
            
        :return: (float, numpy.ndarray [1+Du+Du*(Dz + 1)])
            Negative Q-function and gradients.
        """
        T = X.shape[0]
        self.update_parameters_sigma_beta_W(params)
        dW = numpy.zeros(self.W.shape)
        dln_beta = numpy.zeros(self.Du)
        dlnsigma2_x = numpy.zeros(1)
        phi = smoothing_density.slice(range(1,T+1))
        # E[epsilon(z)^2]
        mat = -self.C
        vec = X - self.d
        E_epsilon2 = numpy.sum(phi.integrate('Ax_aBx_b_inner', A_mat=mat, a_vec=vec, B_mat=mat, b_vec=vec), axis=0)
        dlnsigma2_x += .5 * E_epsilon2 / self.sigma_x ** 2
        # E[D_inv epsilon(z)^2(z)] & E[log(sigma^2 + f(h))]
        E_D_inv_epsilon2 = 0
        E_ln_sigma2_f = 0
        for iu in range(self.Du):
            uRu_i, log_lb_sum_i, dw_i, dln_beta_i, dlnsigma2_i  = self.get_lb_i(iu, phi, X, update='sigma_beta_W')
            E_D_inv_epsilon2 += uRu_i
            E_ln_sigma2_f += log_lb_sum_i
            dW[iu] = dw_i
            dln_beta[iu] = dln_beta_i
            dlnsigma2_x += dlnsigma2_i
        # data part
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / self.sigma_x ** 2
        # determinant part
        Qm -= .5 * E_ln_sigma2_f + .5 * T * (self.Dx - self.Du) * numpy.log(self.sigma_x ** 2)
        # constant part
        Qm -= T * self.Dx * numpy.log(2 * numpy.pi)
        dlnsigma2_x -= .5 * T * (self.Dx - self.Du) #/ self.sigma_x ** 2
        #print(numpy.array([dlnsigma2_x]).shape, dln_beta.shape)
        gradients = numpy.concatenate([dlnsigma2_x, dln_beta, dW.flatten()])
        return -Qm, -gradients
        
    ####################### FUNCTIONS FOR OPTIMIZING U ################################################        
    @staticmethod
    def gen_lin_ind_vecs(U):
        """ Generates linearly independent vectors from U.
        
        :param U: numpy.ndarray [N,M]
            Set of M vectors.
        
        :return: numpy.ndarray [N,N-M]
            return N-M linear inpendent vectors.
        """
        N, M = U.shape
        rand_vecs = numpy.random.rand(N, N - M)
        V_fixed = numpy.hstack([U, rand_vecs])
        V = numpy.copy(V_fixed)
        for m in range(N - M):
            v = rand_vecs[:,m]
            V[:,M+m] -= numpy.dot(V_fixed.T, v) / numpy.sqrt(numpy.sum(v ** 2))
        return V[:,M:]
    
    @staticmethod
    def proj(U, v):
        """ Projects v on U.
        
            proj_U(v) = (vU)/|U| U'
            
        :param U: numpy.ndarray [N,M]
            Set of M vectors.
        :param v: numpy.ndarray [N]
            Vector U is projected on.
        
        :return: numpy.ndarray [N]
            Projection of v on U.
        """
        return numpy.dot(numpy.dot(v, U) / numpy.linalg.norm(U, axis=0), U.T)

    def partial_gs(self, U):
        """ Partial Gram-Schmidt process, which generates orthonormal vectors (also to U).
        
        :param U: numpy.ndarray [N,M]
            Set of M orthonormal vectors.
            
        :return: numpy.ndarray [N,N-M]
            Orthonormal vectors (orthonormal to itself and U).
        """
        N, M = U.shape
        V = numpy.empty((N, N - M))
        #I = self.gen_lin_ind_vecs(U)#numpy.random.randn(N,N-M)
        I = numpy.eye(N)[:,M:]
        #I[-1,0] = 1
        for d in range(N - M):
            v = I[:,d]
            V[:,d] = v - self.proj(U, v) - self.proj(V[:,:d], v)
            V[:,d] /= numpy.sqrt(numpy.sum(V[:,d] ** 2))  
        return V
    
    ####################### Functions for bounds of non tractable terms in the Q-function ################################################  
    def f(self, h, beta):
        """ Computes the function
            
            f(h) = 2 * beta * cosh(h)
            
        :param h: numpy.ndarray
            Activation functions.
        :param beta: numpy.ndarray
            Scaling factor.
        
        :return: numpy.ndarray
            Evaluated functions.
        """
        return 2 * beta * numpy.cosh(h)
    
    def f_prime(self, h, beta):
        """ Computes the derivative of f
            
            f'(h) = 2 * beta * sinh(h)
            
        :param h: numpy.ndarray
            Activation functions.
        :param beta: numpy.ndarray
            Scaling factor.
        :return: numpy.ndarray
            Evaluated derivative functions.
        """
        return 2 * beta * numpy.sinh(h)
    
    def g(self, omega, beta):
        """ Computes the function
        
            g(omega) = f'(omega) / (sigma_x^2 + f(omega)) / |omega|
            
            for the variational boind
            
        :param omega: numpy.ndarray
            Free variational parameter.
        :param beta: numpy.ndarray
            Scaling factor.
        """
        return self.f_prime(omega, beta) / (self.sigma_x ** 2 + self.f(omega, beta)) / numpy.abs(omega)
        
    def get_lb_i(self, iu: int, phi: densities.GaussianDensity, X: numpy.ndarray, conv_crit: float=1e-4, update: str=None):
        """ Computes the variational lower bounds for the log determinant and inverse term. In addition it provides terms, 
        that are needed for updating different model parameters.
        
        :param iu: int
            Index of the component the bound should be computed for.
        :param phi: GaussianDensity
            Density over latent variables.
        :param X: numpy.ndarray [T, Dx]
            Data.
        :param conv_crit: float
            When bounds are considered to be converged (max. change in omega, Default=1e-4).
        :param update: str
            Determines which additional terms should be provided. If None only the lower bounds for the inverse term and the log determinant are provided. If
            `sigma_beta_W`, `C`, `d`, `U` terms that are required for updating the corresponding model parameters are provided. (Default=None)
            
        :return: tuple
            Returns terms for lower bound or for its gradients.
        """
        T = X.shape[0]
        w_i = self.W[iu:iu+1,1:]
        v = numpy.tile(w_i, (T, 1))
        b_i = self.W[iu:iu+1,0]
        u_i = self.U[:,iu:iu+1]
        beta = self.beta[iu:iu+1]
        uC = numpy.dot(u_i.T, -self.C)
        ux_d = numpy.dot(u_i.T, X.T-self.d[:,None])
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = numpy.sqrt(phi.integrate('Ax_aBx_b_inner', A_mat=w_i, a_vec=b_i,
                                                                  B_mat=w_i, b_vec=b_i))
        f_omega_dagger = self.f(omega_dagger, beta)
        log_lb = numpy.log(self.sigma_x ** 2 + f_omega_dagger)
        # Lower bound for E[f(h) / (sigma_x^2 + f(h)) * (u'epsilon(z))^2]
        omega_star = numpy.ones(T)
        converged = False
        num_iter = 0
        while not converged and num_iter < 50:
            # From the lower bound term
            g_omega = self.g(omega_star, beta)
            nu_plus = (1. - g_omega[:,None] * b_i) * w_i
            nu_minus = (-1. - g_omega[:,None] * b_i) * w_i
            ln_beta = - numpy.log(self.sigma_x ** 2 + self.f(omega_star, beta)) - .5 * g_omega * (b_i ** 2 - omega_star ** 2) + numpy.log(beta)
            ln_beta_plus = ln_beta + b_i
            ln_beta_minus = ln_beta - b_i
            # Create OneRankFactors
            exp_factor_plus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
            exp_factor_minus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
            # Create the two measures
            exp_phi_plus = phi.hadamard(exp_factor_plus)
            exp_phi_minus = phi.hadamard(exp_factor_minus)
            # Fourth order integrals E[h^2 (x-Cz-d)^2]
            mat1 = uC
            vec1 = ux_d.T
            mat2 = w_i
            vec2 = b_i
            quart_int_plus = exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1, 
                                                                              C_mat=mat2, c_vec=vec2, D_mat=mat2, d_vec=vec2)
            quart_int_minus = exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1, 
                                                                              C_mat=mat2, c_vec=vec2, D_mat=mat2, d_vec=vec2)
            quart_int = quart_int_plus + quart_int_minus
            # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate('Ax_aBx_b_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
            quad_int_minus = exp_phi_minus.integrate('Ax_aBx_b_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
            quad_int = quad_int_plus + quad_int_minus
            omega_old = omega_star
            #omega_star = numpy.amin([numpy.amax([numpy.sqrt(quart_int / quad_int), 1e-10]), 1e2])
            #quad_int[quad_int < 1e-10] = 1e-10
            omega_star = numpy.sqrt(numpy.abs(quart_int / quad_int))
            # For numerical stability
            #omega_star[omega_star < 1e-10] = 1e-10
            #omega_star[omega_star > 30] = 30
            #print(numpy.amax(numpy.abs(omega_star - omega_old)))
            converged = numpy.amax(numpy.abs(omega_star - omega_old)) < conv_crit
            num_iter += 1
        #print(numpy.amax(numpy.abs(omega_star - omega_old)))
        mat1 = -self.C
        vec1 = X - self.d[None]
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R = R_plus + R_minus
        uRu = numpy.sum(u_i * numpy.dot(numpy.sum(R, axis=0), u_i))
        log_lb_sum = numpy.sum(log_lb)
        if update == 'sigma_beta_W':
            uRu = numpy.sum(u_i * numpy.dot(numpy.sum(R, axis=0), u_i))
            ##### w_i gradiend ######################################################################
            # E[f'(h)exp(-k(h,omega^*)) dh/dw (u'epsilon(z))^2]
            # Matrix and vector for dh/dw
            dW = numpy.zeros((self.Dz + 1, self.Dz))
            dW[1:] = numpy.eye(self.Dz)
            db = numpy.zeros(self.Dz + 1)
            db[0] = 1
            dw_i = numpy.sum(exp_phi_plus.integrate('Ax_aBx_bCx_c_outer', A_mat=uC, a_vec=ux_d.T,
                                                    B_mat=uC, b_vec=ux_d.T, C_mat=dW, c_vec=db), axis=0)
            dw_i -= numpy.sum(exp_phi_minus.integrate('Ax_aBx_bCx_c_outer', A_mat=uC, a_vec=ux_d.T, 
                                                      B_mat=uC, b_vec=ux_d.T, C_mat=dW, c_vec=db), axis=0)
            # -g(omega) * E[f(h)exp(-k(h,omega^*)) h dh/dw (u'epsilon(z))^2]
            dw_i -= numpy.einsum('a,ab->b', g_omega, exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_outer', A_mat=w_i, a_vec=b_i,
                                                                            B_mat=uC, b_vec=ux_d.T, C_mat=uC, c_vec=ux_d.T,
                                                                            D_mat=dW, d_vec=db)[:,0])
            dw_i -= numpy.einsum('a,ab->b', g_omega, exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_outer', A_mat=w_i, a_vec=b_i,
                                                                             B_mat=uC, b_vec=ux_d.T, C_mat=uC, c_vec=ux_d.T,
                                                                             D_mat=dW, d_vec=db)[:,0])
            dw_i /= self.sigma_x ** 2
            # g(omega^+)E[h dh/dw]
            dw_i -= numpy.einsum('a,ab->b', self.g(omega_dagger, beta), phi.integrate('Ax_aBx_b_outer', A_mat=w_i, a_vec=b_i, 
                                                                                      B_mat=dW, b_vec=db)[:,0])
            dw_i /= 2.
            ###########################################################################################
            ##### beta_i gradient #####################################################################
            weighted_R = numpy.einsum('abc,a->bc', R, 1. / (self.sigma_x ** 2 + self.f(omega_star, beta))) 
            #  u'R u / (sigma_x^2 + f(omega^*))
            dln_beta_i = numpy.sum(u_i * numpy.dot(weighted_R, u_i))
            dln_beta_i -= numpy.sum(f_omega_dagger / (self.sigma_x ** 2 + f_omega_dagger))
            dln_beta_i /= 2.
            ##### sigma_x ** 2 gradient ###############################################################
            dlnsigma2 =  - uRu / self.sigma_x ** 2
            dlnsigma2 -= numpy.sum(u_i * numpy.dot(weighted_R, u_i))
            dlnsigma2 -= numpy.sum(self.sigma_x ** 2 / (self.sigma_x ** 2 + f_omega_dagger))
            dlnsigma2 /= 2.
            return uRu, log_lb_sum, dw_i, dln_beta_i, dlnsigma2
        elif update == 'C':
            intD_inv_zz_plus = exp_phi_plus.integrate('xx')
            intD_inv_zz_minus = exp_phi_minus.integrate('xx')
            intD_inv_zz = intD_inv_zz_plus + intD_inv_zz_minus
            intD_inv_z_plus = exp_phi_plus.integrate('x')
            intD_inv_z_minus = exp_phi_minus.integrate('x')
            intD_inv_z = intD_inv_z_plus + intD_inv_z_minus
            return intD_inv_z, intD_inv_zz
        elif update == 'd':
            intD_inv_z_plus = exp_phi_plus.integrate('x')
            intD_inv_z_minus = exp_phi_minus.integrate('x')
            intD_inv_z = intD_inv_z_plus + intD_inv_z_minus
            intD_inv_plus = exp_phi_plus.integrate()
            intD_inv_minus = exp_phi_minus.integrate()
            intD_inv = intD_inv_plus + intD_inv_minus
            return intD_inv, intD_inv_z
        elif update == 'U':
            return numpy.sum(R, axis=0)
        else:
            return uRu, log_lb_sum
        

class BernoulliObservationModel(ObservationModel):
    
    def __init__(self, Dx: int, Dz: int, Dphi_u: int=0):
        """ This class implements an observation model for Bernoulli data `x\in(0,1)`, where the observations 
        are generated as
        
            x_{t,i} \sim \sigma(h_{t,i}).
            
            with h_{t,i} = \theta_i' phi_i(z_t),
            
            phi_i(z) = (1,z,u_i)
            
        :param Dx: int
            Dimensionality of observations.
        :param Dz: int
            Dimensionality of latent space.
        :param Dphi_u: int
            Dimensionality of control variable features. (Default=0)
        """
        self.Dx, self.Dz, self.Dphi_u = Dx, Dz, Dphi_u
        self.Dphi = Dphi_u + Dz + 1
        self.Theta = numpy.random.randn(self.Dx, self.Dphi)
        
    def compute_feature_vector(self, z: numpy.ndarray, ux: numpy.ndarray=None) -> numpy.ndarray:
        """ Constructs the feature vector
        
            phi_i(z) = (1,z,u_i)
            
        :param z: numpy.ndarray [T, Dz]
            Instantiation of latent variables.
        :param uz: numpy.ndarray [T, Dphi_u] or [T, Dphi_u, Dx]
            Control variables. (Default=None)
            
        :return: numpy.ndarray [T, Dphi, Dx]
            Feature vector.
        """
        T = z.shape[0]
        phi = numpy.zeros((T, self.Dphi, self.Dx))
        phi[:,0] = 1
        phi[:,1:self.Dz+1] = z
        if ux is not None:
            phi[:,self.Dz+1:] = ux
        return phi
    
    def compute_expected_feature_vector(self, density: 'GaussianDensity', ux: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the expected feature vector
        
            E[phi_i(z)] = (1,E[z],u_i)
            
        :param density: GaussianDensity
            Density over z.
        :param uz: numpy.ndarray [T, Dphi_u] or [T, Dphi_u, Dx]
            Control variables. (Default=None)
            
        :return: numpy.ndarray [T, Dx, Dphi]
            Expected feature vector.
        """
        T = density.R
        Ephi = numpy.zeros((T, self.Dx, self.Dphi))
        Ephi[:,:,0] = 1 
        Ephi[:,:,1:self.Dz+1] = density.integrate('x')[:,None]
        if ux is not None:
            Ephi[:,:,self.Dz+1:] = ux
        return Ephi
    
    def compute_expected_feature_outer_product(self, density: 'GaussianDensity', ux: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the expected feature vector
        
            E[phi_i(z) phi_i(z)'] = (1,    E[z'],   u_i',
                                     E[z], E[zz'],  E[z]u_i',
                                     u_i,  E[z]u_i, u_iu_i')
            
        :param density: GaussianDensity
            Density over z.
        :param ux: numpy.ndarray [T, Dx, Dphi_u] or [T, Dphi_u]
            Control variables. (Default=None)
            
        :return: numpy.ndarray [T,  Dx, Dphi, Dphi]
            Expected feature vector.
        """
        T = density.R
        
        Ez = density.integrate('x')
        Ezz = density.integrate('xx')
        
        Ephi_outer = numpy.zeros((T, self.Dx, self.Dphi, self.Dphi))
        Ephi_outer[:,:,0,0] = 1                                              # 1
        Ephi_outer[:,:,1:self.Dz+1,0] = Ez                                   # E[z']
        Ephi_outer[:,:,0,1:self.Dz+1] = Ez                                   # E[z]
        Ephi_outer[:,:,1:self.Dz+1,1:self.Dz+1] = Ezz                        # E[zz']
        if ux is not None:
            if ux.ndim == 2:
                ux = ux.reshape((uz.shape[0],1,uz.shape[1]))
            Ez_ux = Ez[:,None,:,None] * ux[:,:,None]
            uxux = ux[:,:,None] * ux[:,:,:,None]
            Ephi_outer[:,:,self.Dz+1:,0] = ux                                # u'
            Ephi_outer[:,:,0,self.Dz+1:] = ux                                # u
            Ephi_outer[:,:,1:self.Dz+1,self.Dz+1:] = Ez_ux                   # E[z] u'
            Ephi_outer[:,:,self.Dz+1:,1:self.Dz+1] = numpy.swapaxes(
                Ez_ux, axis1=2, axis2=3)                                     # E[z'] u
            Ephi_outer[:,:,self.Dz+1:,self.Dz+1:] = uxux                     # uu'
        return Ephi_outer
    
    def get_omega_star(self, density: 'GaussianDensity', x_t: numpy.ndarray, ux_t: numpy.ndarray=None, conv_crit: float=1e-4) -> numpy.ndarray:
        """ Gets the optimal variational parameter.
        """
        
        omega_old = numpy.ones((self.Dx))
        converged = False
        v = self.Theta[:,1:self.Dz+1]
        while not converged:
            g = 1. / numpy.abs(omega_old) * numpy.tanh(.5 * omega_old)
            sign = 2. * x_t[0] - 1.
            nu = self.Theta[:,1:self.Dz+1] * (.5 * sign - g * self.Theta[:,0])[:,None]
            #ln_beta = numpy.log(2 * numpy.cosh(.5 * omega_old)) + .5 * sign * self.Theta[:,0] - .5 * g * (self.Theta[:,0] ** 2 - omega_old ** 2)
            if ux_t is not None:
                theta_uz = numpy.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
                nu = nu - self.Theta[:,1:self.Dz+1] * (g * theta_ux)[:, None]
                #ln_beta = ln_beta + .5 * sign * theta_uz - g * (.5 * theta_uz ** 2 + self.Theta[:,0] * ux_t[0])           
                
            sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu)
            sigma_density = density.multiply(sigma_lb).get_density()
            A_mat = self.Theta[:,1:self.Dz+1]
            a_vec = self.Theta[:,0]
            if ux_t is not None:
                a_vec = a_vec + theta_ux
            Eh2 = sigma_density.integrate('Ax_aBx_b_inner', A_mat=A_mat, a_vec=a_vec, B_mat=A_mat, b_vec=a_vec)
            omega_star = numpy.sqrt(Eh2)
            omega_star[omega_star < 1e-10] = 1e-10
            converged = numpy.amax(numpy.abs(omega_star - omega_old)) < conv_crit
            omega_old = numpy.copy(omega_star)
        return omega_star
    
        
    def filtering(self, prediction_density: 'GaussianDensity', x_t: numpy.ndarray, ux_t: numpy.ndarray=None,**kwargs) -> 'GaussianDensity':
        """ Here the variational approximation of filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
            Observation.
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        omega_star = self.get_omega_star(prediction_density, x_t, ux_t)
        v = self.Theta[:,1:self.Dz+1]
        g = 1. / numpy.abs(omega_star) * numpy.tanh(.5 * omega_star)
        sign = 2. * x_t[0] - 1.
        nu = self.Theta[:,1:self.Dz+1] * (.5 * sign - g * self.Theta[:,0])[:,None]
        if ux_t is not None:
            theta_ux = numpy.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
            nu = nu - self.Theta[:,1:self.Dz+1] * (g * theta_ux)[:, None]
        sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu)
        filter_measure = prediction_density
        for idx in range(self.Dx):
            filter_measure = filter_measure.hadamard(sigma_lb.slice([idx]))
        filter_density = filter_measure.get_density()
        return filter_density
    
    def get_omega_dagger(self, density: 'GaussianDensity', ux_t: numpy.ndarray=None, conv_crit: float=1e-4) -> numpy.ndarray:
        """ Gets the optimal variational parameter.
        """
    
        A_mat = self.Theta[:,1:self.Dz+1]
        a_vec = self.Theta[:,0]
        if ux_t is not None:
            theta_ux = numpy.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
            a_vec = a_vec + theta_ux
        Eh2 = density.integrate('Ax_aBx_b_inner', A_mat=A_mat, a_vec=a_vec, B_mat=A_mat, b_vec=a_vec)
        omega_dagger = numpy.sqrt(Eh2)
        omega_dagger[omega_dagger < 1e-10] = 1e-10
        return omega_dagger
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray, u_x: numpy.ndarray=None,**kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        :param u_x: numpy.ndarray [T, ...]
            Control parameters. (Default=None)
        """ 
        A_theta = numpy.zeros((self.Dx, self.Dphi, self.Dphi))
        b_theta = numpy.zeros((self.Dx, self.Dphi))
        T = X.shape[0]
        for t in range(T):
            density_t = smoothing_density.slice([t+1])
            if u_x is not None:
                ux_t = u_x[t:t+1]
            else:
                ux_t = None
            omega_dagger = self.get_omega_dagger(density_t, ux_t=ux_t)
            g = 1. / numpy.abs(omega_dagger) * numpy.tanh(.5 * omega_dagger)
            Ephiphi = self.compute_expected_feature_outer_product(density_t, ux=ux_t)[0]
            A_theta = A_theta + g[:,None,None] * Ephiphi
            Ephi = self.compute_expected_feature_vector(density_t, ux=ux_t)[0]
            sign = 2. * X[t] - 1.
            b_theta = b_theta + .5 * sign[:,None] * Ephi
        #A_theta += 1e-4 * numpy.eye(self.Dphi)[None]
        self.Theta = numpy.linalg.solve(A_theta, b_theta)
        
    def get_lb_sigma(self, density: 'GaussianDensity', x_t: numpy.ndarray, ux_t: numpy.ndarray=None) -> numpy.ndarray:
        """ Computes the lower bounds for the data probability.
        """
        omega_star = self.get_omega_star(density, x_t, ux_t)
        v = self.Theta[:,1:self.Dz+1]
        g = 1. / numpy.abs(omega_star) * numpy.tanh(.5 * omega_star)
        sign = 2. * x_t[0] - 1.
        nu = self.Theta[:,1:self.Dz+1] * (.5 * sign - g * self.Theta[:,0])[:,None]
        ln_beta = - numpy.log(2) -  logcosh(.5 * omega_star) + .5 * sign * self.Theta[:,0] - .5 * g * (self.Theta[:,0] ** 2 - omega_star ** 2)
        if ux_t is not None:
            theta_ux = numpy.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
            nu = nu - self.Theta[:,1:self.Dz+1] * (g * theta_ux)[:, None]
            ln_beta = ln_beta + .5 * sign * theta_ux - g * (.5 * theta_ux ** 2 + self.Theta[:,0] * ux_t[0])
        sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu, ln_beta=ln_beta)
        measure = density
        for idx in range(self.Dx):
            measure = measure.hadamard(sigma_lb.slice([idx]))
        prob_lb = measure.integrate()[0]
        return prob_lb

        
    def evaluate_llk(self, p_z: 'GaussianDensity', X: numpy.ndarray, u_x: numpy.ndarray=None,**kwargs) -> float:
        """ Computes the lower bound of log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: numpy.ndarray [T, Dx]
            Observations.
            
        :return: float
            Log likelihood lower bound.
        """
        T = X.shape[0]
        llk = 0
        #p_x = self.emission_density.affine_marginal_transformation(p_z)
        for t in range(0,T):
            if u_x is not None:
                ux_t = u_x[t:t+1]
            else:
                ux_t = None
            prob_lb = self.get_lb_sigma(p_z.slice([t]), X[t:t+1], ux_t=ux_t)
            llk += numpy.log(prob_lb)
        return llk