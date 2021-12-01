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
import sys
sys.path.append('../src_jax/')
import scipy
from scipy.optimize import minimize, NonlinearConstraint, minimize_scalar
from jax import numpy as jnp
from jax import scipy as jsc
import numpy as np
from jax import lax
from jax import jit, value_and_grad, vmap
from functools import partial

import densities, conditionals, factors
#from pathos.multiprocessing import ProcessingPool as Pool


def recommend_dims(X, smooth_window=20, cut_off=.99):
    X_mean = jnp.mean(X, axis=0)
    T = X.shape[0]
    X_smoothed = jnp.empty(X.shape)
    for i in range(X.shape[1]):
        X_smoothed.at[:,i].set(jnp.convolve(X[:,i],
                                         jnp.ones(smooth_window) / smooth_window, 
                                         mode='same'))
    eig_vals_X, eig_vecs_X = scipy.linalg.eigh(jnp.dot((X_smoothed-X_mean[None]).T, 
                                                        X_smoothed-X_mean[None]))
    Dz = jnp.searchsorted(jnp.cumsum(eig_vals_X[::-1]) / jnp.sum(eig_vals_X), cut_off) + 1
    C =  eig_vecs_X[:,-Dz:] * eig_vals_X[-Dz:] / T
    z_hat = jnp.dot(jnp.linalg.pinv(C), (X_smoothed - X_mean).T).T
    delta_X = X - jnp.dot(z_hat, C.T) - X_mean
    cov = jnp.dot(delta_X.T, delta_X)
    eig_vals_deltaX, eig_vecs_deltaX = scipy.linalg.eigh(cov)
    Du = jnp.searchsorted(jnp.cumsum(eig_vals_deltaX[::-1]) / jnp.sum(eig_vals_deltaX), cut_off)
    return Dz, Du
    
def logcosh(x):
    # s always has real part >= 0
    s = jnp.sign(x) * x
    p = jnp.exp(-2 * s)
    return s + jnp.log1p(p) - jnp.log(2)

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
    
    def filtering(self, prediction_density: 'GaussianDensity', x_t: jnp.ndarray, **kwargs) -> 'GaussianDensity':
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
        
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        raise NotImplementedError('Filtering for observation model not implemented.')
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray, **kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """  
        raise NotImplementedError('Hyperparameter updates for observation model not implemented.')
        
    
    def evalutate_llk(self, p_z: 'GaussianDensity', X: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """ Computes the log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: jnp.ndarray [T, Dx]
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
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
        self.d = jnp.zeros(Dx)
        self.Qx = noise_x ** 2 * jnp.eye(self.Dx)
        self.emission_density = conditionals.ConditionalGaussianDensity(jnp.array([self.C]), 
                                                                        jnp.array([self.d]), 
                                                                        jnp.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
        
    def pca_init(self, X: jnp.ndarray, smooth_window: int=10):
        """ Sets the model parameters to an educated initial guess, based on principal component analysis.
            More specifically `d` is set to the mean of the data and `C` to the first principal components 
            of the (smoothed) data. The covariance `Qx` is set to the empirical covariance of the residuals.
        
        :param X: jnp.ndarray [T, Dx]
            Data.
        :param smoothed_window: int
            Width of the box car filter data are smoothed with. (Default=10)
        """
        self.d = jnp.mean(X, axis=0)
        T = X.shape[0]
        X_smoothed = jnp.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = jnp.convolve(X[:,i], 
                                             jnp.ones(smooth_window) / smooth_window, 
                                             mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(jnp.dot((X_smoothed-self.d[None]).T, 
                                                         X_smoothed-self.d[None]), 
                                               eigvals=(self.Dx-jnp.amin([self.Dz,self.Dx]), self.Dx-1))
        self.C[:,:jnp.amin([self.Dz,self.Dx])] =  eig_vecs * eig_vals / T
        z_hat = jnp.dot(jnp.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - jnp.dot(z_hat, self.C.T) - self.d
        self.Qx = jnp.dot(delta_X.T, delta_X)
        self.emission_density = conditionals.ConditionalGaussianDensity(jnp.array([self.C]), 
                                                                        jnp.array([self.d]), 
                                                                        jnp.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
        
    def filtering(self, prediction_density: 'GaussianDensity', x_t: jnp.ndarray, **kwargs) -> 'GaussianDensity':
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
            Observation.
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        # p(z_t| x_t, x_{1:t-1})
        p_z_given_x = self.emission_density.affine_conditional_transformation(prediction_density)
        # Condition on x_t
        cur_filter_density = p_z_given_x.condition_on_x(x_t)
        return cur_filter_density
    
    def gappy_filtering(self, prediction_density: 'GaussianDensity', x_t: jnp.ndarray, **kwargs) -> 'GaussianDensity':
        """ Here the filtering density is calculated for incomplete data. Not observed values should be nans.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
            Observation, where unobserved dimensions are filled with NANs.
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        # In case all data are unobserved
        if jnp.alltrue(jnp.isnan(x_t[0])):
            return prediction_density
        # In case all data are observed
        elif jnp.alltrue(jnp.logical_not(jnp.isnan(x_t[0]))):
            cur_filter_density = self.filtering(prediction_density, x_t)
            return cur_filter_density
        # In case we have only partial observations
        else:
            observed_dims = jnp.where(jnp.logical_not(jnp.isnan(x_t[0])))[0]
            # p(z_t, x_t| x_{1:t-1})
            p_zx = self.emission_density.affine_joint_transformation(prediction_density)
            # p(z_t, x_t (observed) | x_{1:t-1})
            marginal_dims = jnp.concatenate([jnp.arange(self.Dz), self.Dz + observed_dims])
            p_zx_observed = p_zx.get_marginal(marginal_dims)
            # p(z_t | x_t (observed), x_{1:t-1})
            conditional_dims = jnp.arange(self.Dz,self.Dz + len(observed_dims))
            p_z_given_x_observed = p_zx_observed.condition_on(conditional_dims)
            cur_filter_density = p_z_given_x_observed.condition_on_x(x_t[:,observed_dims])
            return cur_filter_density
        
    def gappy_data_density(self, p_z: 'GaussianDensity', x_t: jnp.ndarray, **kwargs):
        """ Here the data density is calculated for incomplete data. Not observed values should be nans.
        
         p(x_t) = p(x_t|z_t)p(z_t) dz_t
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
            Observation, where unobserved dimensions are filled with NANs.
        :return: (jnp.ndarray, jnp.ndarray)
            Mean and variance of unobserved entries.
        """
        # In case all data are unobserved
        if jnp.alltrue(jnp.isnan(x_t[0])):
            p_x = self.emission_density.affine_marginal_transformation(p_z)
            return p_x.mu[0], jnp.sqrt(p_x.Sigma[0].diagonal(axis1=-1, axis2=-2))
        # In case all data are observed
        elif jnp.alltrue(jnp.logical_not(jnp.isnan(x_t[0]))):
            return jnp.array([]), jnp.array([])
        # In case we have only partial observations
        else:
            observed_dims = jnp.where(jnp.logical_not(jnp.isnan(x_t[0])))[0]
            # Density over unobserved variables
            p_x = self.emission_density.affine_marginal_transformation(p_z)
            p_ux_given_ox = p_x.condition_on(observed_dims)
            p_ux = p_ux_given_ox.condition_on_x(x_t[:,observed_dims])
            return p_ux.mu[0], jnp.sqrt(p_ux.Sigma.diagonal(axis1=-1, axis2=-2))
    
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray, **kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """  
        self.update_C(smoothing_density, X)
        self.update_d(smoothing_density, X)
        self.update_Qx(smoothing_density, X)
        self.update_emission_density()

           
    def update_Qx(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray):
        """ This procedure updates the covariance of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """  
        T = X.shape[0]
        A = -self.C
        a_t = jnp.concatenate([self.d[None], X]) - self.d[None]
        # Exx = jnp.zeros((self.Dx, self.Dx))

        Exx = jnp.sum(smoothing_density.integrate('Ax_aBx_b_outer', A_mat=A,
                                                 a_vec=a_t, B_mat=A,
                                                 b_vec=a_t)[1:], axis=0)
        # for t in range(1, T+1):
        #     cur_smooth_density = smoothing_density.slice(jnp.array([t]))
        #     Exx += cur_smooth_density.integrate('Ax_aBx_b_outer', A_mat=A,
        #                                         a_vec=a_t[t-1], B_mat=A,
        #                                         b_vec=a_t[t-1])[0]
        self.Qx = Exx / T
        
    def update_C(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray):
        """ This procedure updates the transition matrix of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """  
        Ezz = jnp.sum(smoothing_density.integrate('xx')[1:], axis=0)
        Ez = smoothing_density.integrate('x')[1:]
        zx = jnp.sum(Ez[:,:,None] * (X[:,None] - self.d[None,None]), axis=0)
        self.C = jnp.linalg.solve(Ezz, zx).T
        
    def update_d(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray):
        """ This procedure updates the transition offset of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """
        self.d = jnp.mean(X - jnp.dot(smoothing_density.mu[1:], self.C.T), axis=0)
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.emission_density = conditionals.ConditionalGaussianDensity(jnp.array([self.C]),
                                                                        jnp.array([self.d]),
                                                                        jnp.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]

    @staticmethod
    def llk_step(t: int, p_x: 'GaussianDensity', X: jnp.ndarray):
        cur_p_x = p_x.slice(jnp.array([t]))
        return cur_p_x.evaluate_ln(X[t].reshape((1, -1)))[0, 0]
        
    def evaluate_llk(self, p_z: 'GaussianDensity', X: jnp.ndarray, **kwargs) -> float:
        """ Computes the log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: jnp.ndarray [T, Dx]
            Observations.
            
        :return: float
            Log likelihood.
        """
        T = X.shape[0]
        llk = 0
        p_x = self.emission_density.affine_marginal_transformation(p_z)
        llk_step = jit(lambda t: self.llk_step(t, p_x, X))
        for t in range(0,T):
            llk += llk_step(t)
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
            # self.C = jnp.eye(Dx)[:,:Dz]
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
            self.C = self.C / jnp.sqrt(jnp.sum(self.C ** 2, axis=0))[None]
        self.d = jnp.zeros(Dx)
        rand_mat = np.random.rand(self.Dx, self.Dx) - .5
        Q, R = np.linalg.qr(rand_mat)
        self.U = jnp.array(Q[:,:self.Du])
        #self.U = jnp.eye(Dx)[:, :Du]
        W = 1e-4 * np.random.randn(self.Du, self.Dz + 1)
        W[:,0] = 0
        self.W = jnp.array(W)
        self.beta = noise_x ** 2 * jnp.ones(self.Du)
        self.sigma_x = jnp.array([noise_x])
        self.omega_star = None
        self.emission_density = conditionals.HCCovGaussianConditional(M = jnp.array([self.C]), 
                                                                      b = jnp.array([self.d]), 
                                                                      sigma_x = self.sigma_x,
                                                                      U = self.U,
                                                                      W = self.W,
                                                                      beta = self.beta)
        
        
    def pca_init(self, X: jnp.ndarray, smooth_window: int=10):
        """ Sets the model parameters to an educated initial guess, based on principal component analysis.
            More specifically `d` is set to the mean of the data and `C` to the first principal components 
            of the (smoothed) data. 
            Then `U` is initialized with the first pricinpal components of the empirical covariance of the 
            residuals.
        
        :param X: jnp.ndarray [T, Dx]
            Data.
        :param smoothed_window: int
            Width of the box car filter data are smoothed with. (Default=10)
        """
        self.d = jnp.mean(X, axis=0)
        T = X.shape[0]
        X_smoothed = np.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = np.convolve(X[:,i],
                                             np.ones(smooth_window) / smooth_window,
                                             mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(jnp.dot((X_smoothed-self.d[None]).T, 
                                                         X_smoothed-self.d[None]), 
                                               eigvals=(self.Dx-np.amin([self.Dz,self.Dx]), self.Dx-1))
        C = np.array(self.C)
        C[:,:np.amin([self.Dz,self.Dx])] =  eig_vecs * eig_vals / T
        self.C = jnp.array(self.C)
        z_hat = jnp.dot(jnp.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - jnp.dot(z_hat, self.C.T) - self.d
        cov = jnp.dot(delta_X.T, delta_X)
        self.U = jnp.array(scipy.linalg.eigh(cov, eigvals=(self.Dx-self.Du, self.Dx-1))[1])
        self.emission_density = conditionals.HCCovGaussianConditional(M = jnp.array([self.C]), 
                                                                      b = jnp.array([self.d]), 
                                                                      sigma_x = self.sigma_x,
                                                                      U = self.U,
                                                                      W = self.W,
                                                                      beta = self.beta)
        
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray, iteration: int, **kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """  

        if iteration < 30:
            num_resets = 10
        else:
            num_resets = 10
        if self.omega_star is None:
            self.omega_star = jnp.ones((self.Du, X.shape[0]))
        #print(self.C.shape, self.d.shape, self.U.shape, self.sigma_x.shape, self.beta.shape, self.W.shape)
        phi = smoothing_density.slice(jnp.arange(1, smoothing_density.R))
        get_omegas = jit(vmap(HCCovObservationModel.get_omegas_i, in_axes=(None, None, 0, 0, 1, 0, None, None, None),
                          out_axes=(0,0,None), axis_name='i'), static_argnums=(0))
        #get_omegas = vmap(HCCovObservationModel.get_omegas_i, in_axes=(None, None, 0, 1, 0, None, None, None),
        #                  out_axes=(0,0), axis_name='i')
        val_old = -np.inf
        converged = False
        num_iter = 0
        func1 = jit(HCCovObservationModel.Qfunc, static_argnums=(1,6,7,8))
        grad_func1 = jit(value_and_grad(HCCovObservationModel.Qfunc), static_argnums=(1,6,7,8))
        func2 = jit(HCCovObservationModel.Qfunc_U_ls, static_argnums=(1,5,6,7))
        grad_func2 = jit(value_and_grad(HCCovObservationModel.Qfunc_U_ls), static_argnums=(1,5,6,7))
        while not converged and num_iter < 50:
            val = self.step(func1, grad_func1, func2, grad_func2, phi, X, get_omegas, num_iter)
            converged = jnp.abs((val - val_old) / val_old) < 1e-5
            val_old = val
            num_iter += 1
        self.update_emission_density()
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.emission_density = conditionals.HCCovGaussianConditional(M = jnp.array([self.C]), 
                                                                      b = jnp.array([self.d]), 
                                                                      sigma_x = self.sigma_x,
                                                                      U = self.U,
                                                                      W = self.W,
                                                                      beta = self.beta)
        
    def step(self, func1, grad_func1, func2, grad_func2, phi, X, get_omegas, iteration, step_size: float=.001):
        omega_dagger, omega_star, num_iter = get_omegas(phi, X, self.omega_star, self.W, self.U, self.beta, self.C, self.d, self.sigma_x)
        self.omega_star = omega_star
        #print(num_iter)
        params = HCCovObservationModel.params_to_vector(self.C, self.d, self.sigma_x, self.beta, self.W)
        val, euclid_grad = grad_func1(params, phi, X, self.U, omega_dagger, omega_star, self.Dx, self.Dz, self.Du)
        params_new = self.euclid_step(func1, params, euclid_grad, phi, X, get_omegas)
        #params_new = self.apply_step(params, euclid_grad, step_size, phi, X, get_omegas)
        C, d, sigma_x, beta, W = HCCovObservationModel.vector_to_params(params_new, self.Dx, self.Dz, self.Du)
        self.C = C
        self.d = d
        self.sigma_x = sigma_x
        self.beta = beta
        self.W = W
        #dW = W - self.W
        #dW = dW.at[dW > .005].set(.005)
        #dW = dW.at[dW < -.005].set(-.005)
        #self.W += dW
        if self.Dx > 1:
            omega_dagger, omega_star, num_iter = get_omegas(phi, X, self.omega_star, self.W, self.U, self.beta, self.C, self.d, self.sigma_x)
            self.omega_star = omega_star
            val, euclid_grad_U = grad_func2(self.U, phi, X, omega_dagger, omega_star, self.Dx, self.Dz, self.Du,
                                            self.C, self.d, self.sigma_x, self.beta, self.W)
            U = self.stiefel_step(func2, euclid_grad_U, phi, X, get_omegas, step_size)
            self.U = U
        return val
    
    def euclid_step(self, func1, params, param_grad: jnp.array, phi, X, get_omegas):
        
        def objective(t):
            params_new = params + t * param_grad
            C, d, sigma_x, beta, W = HCCovObservationModel.vector_to_params(params_new, self.Dx, self.Dz, self.Du)
            omega_dagger, omega_star, num_iter = get_omegas(phi, X, self.omega_star, W, self.U, beta, C, d, sigma_x)
            val = - func1(params_new, phi, X, self.U, omega_dagger, omega_star, self.Dx, self.Dz, self.Du)
            return val
        min_res = minimize_scalar(objective)
        opt_t = min_res.x
        params_new = params + opt_t * param_grad
        return params_new
        
        
    def stiefel_step(self, func2, euclid_dU: jnp.ndarray, phi, X, get_omegas, step_size):
        grad_stiefel = euclid_dU - jnp.dot(jnp.dot(self.U, euclid_dU.T), self.U)
        tangent = jnp.dot(self.U.T, grad_stiefel)
        tangent = .5 * (tangent - tangent.T)
        geodesic = lambda t: jnp.dot(self.U, jsc.linalg.expm(t * tangent))
        #omega_dagger, omega_star, num_iter = get_omegas(phi, X, self.omega_star, self.W, U, self.beta, self.C, self.d, self.sigma_x)
        def objective(t):
            U = geodesic(t)
            omega_dagger, omega_star, num_iter = get_omegas(phi, X, self.omega_star, self.W, U, self.beta, self.C, self.d, self.sigma_x)
            
            val = -func2(U, phi, X, omega_dagger, omega_star, 
                         self.Dx, self.Dz, self.Du, self.C, self.d, self.sigma_x, self.beta, self.W)
            return val
        
        min_res = minimize_scalar(objective)
        opt_t = min_res.x
        U_new = geodesic(opt_t)
        return U_new
    
    def apply_step(self, params, euclid_grad, step_size: float, phi, X, get_omegas):
        params = params + step_size * euclid_grad
        if self.Dx > 1:
            num_params = (self.Dz + 1) * self.Dx
            cur_params = self.Dx * self.Du
            euclid_dU = euclid_grad[jnp.arange(num_params, num_params + cur_params)].reshape((self.Dx, self.Du))
            U_new = self.stiefel_step(euclid_dU, phi, X, get_omegas, step_size)
            params = params.at[jnp.arange(num_params, num_params + cur_params)].set(U_new.flatten())
        else:
            num_params = (self.Dz + 1) * self.Dx
            cur_params = self.Dx * self.Du
            params = params.at[jnp.arange(num_params, num_params + cur_params)].set(1)
        num_params = (self.Dz + 1) * self.Dx + self.Dx * self.Du + 1
        cur_params = self.Du
        ln_y = params[jnp.arange(num_params, num_params + cur_params)]
        ln_y = ln_y.at[ln_y < jnp.log(.25)].set(jnp.log(.25))
        params = params.at[jnp.arange(num_params, num_params + cur_params)].set(ln_y)
        return params

    @staticmethod
    def vector_to_params(params: jnp.ndarray, Dx: int, Dz: int, Du: int):
        num_params = 0
        cur_params = Dz * Dx
        C = params[jnp.arange(num_params, num_params + cur_params)].reshape((Dx, Dz))
        num_params += cur_params
        cur_params = Dx
        d = params[jnp.arange(num_params, num_params + cur_params)]
        num_params += cur_params
        cur_params = 1
        sigma_x = jnp.exp(.5 * params[jnp.arange(num_params, num_params + cur_params)])
        num_params += cur_params
        cur_params = Du
        y = jnp.exp(params[jnp.arange(num_params, num_params + cur_params)])
        beta = sigma_x ** 2 * y
        num_params += cur_params
        cur_params = Du * (Dz + 1)
        W = params[jnp.arange(num_params, num_params + cur_params)].reshape((Du, Dz + 1))
        return C, d, sigma_x, beta, W
    
    @staticmethod
    def params_to_vector(C: jnp.ndarray, d: jnp.ndarray, sigma_x: jnp.ndarray, beta: jnp.ndarray, W: jnp.ndarray):
        C_flattened = C.flatten()
        ln_sigma2_x = 2. * jnp.log(sigma_x)
        ln_y = jnp.log(beta / sigma_x ** 2)
        W_flattened = W.flatten()
        params = jnp.concatenate([C_flattened, d, ln_sigma2_x, ln_y, W_flattened])
        return params
        
    
    @staticmethod
    def Qfunc(params: jnp.ndarray, phi: 'GaussianDensity', X: jnp.ndarray, U: jnp.array, omega_dagger: jnp.array, omega_star: jnp.array, Dx: int, Dz: int, Du: int):
        C, d, sigma_x, beta, W = HCCovObservationModel.vector_to_params(params, Dx, Dz, Du)
        T = X.shape[0]
        vec = X - d
        E_epsilon2 = jnp.sum(phi.integrate('Ax_aBx_b_inner', A_mat=-C, a_vec=vec, B_mat=-C, b_vec=vec), axis=0)
        uRu, log_lb_sum = HCCovObservationModel.get_lb_i(phi, X, W, U, beta, omega_dagger, omega_star, C, d, sigma_x)
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / sigma_x ** 2
        # determinant part
        Qm = Qm - .5 * E_ln_sigma2_f + .5 * T * (Du - Dx) * jnp.log(sigma_x ** 2)
        return jnp.squeeze(Qm) / T
    
    @staticmethod
    def Qfunc_U_ls(U, phi: 'GaussianDensity', X: jnp.ndarray, omega_dagger: jnp.array, omega_star: jnp.array, Dx: int, Dz: int, Du: int,
                   C, d, sigma_x, beta, W):
        T = X.shape[0]
        vec = X - d
        E_epsilon2 = jnp.sum(phi.integrate('Ax_aBx_b_inner', A_mat=-C, a_vec=vec, B_mat=-C, b_vec=vec), axis=0)
        uRu, log_lb_sum = HCCovObservationModel.get_lb_i(phi, X, W, U, beta, omega_dagger, omega_star, C, d, sigma_x)
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / sigma_x ** 2
        # determinant part
        Qm = Qm - .5 * E_ln_sigma2_f + .5 * T * (Du - Dx) * jnp.log(sigma_x ** 2)
        return jnp.squeeze(Qm) / T

    def update_U4(self, phi: 'GaussianDensity', X: jnp.ndarray, get_omegas):
        """ Updates the `U` by maximizing the Q-function. One component at a time is updated analytically.

        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: numpy.ndarray [T, Dx]
            Data.
        """
        converged = False
        T = X.shape[0]
        get_R = lambda U, omega_star: vmap(HCCovObservationModel.get_R,
                                           in_axes=(None, None, 0, 0, 1, 0, None, None, None), axis_name='i')(phi, X,
                                                                                                              self.W,
                                                                                                              self.beta,
                                                                                                              U,
                                                                                                              omega_star,
                                                                                                              self.C,
                                                                                                              self.d,
                                                                                                              self.sigma_x)
        get_omegas_U = lambda U: get_omegas(phi, X, self.W, U, self.beta, self.C, self.d, self.sigma_x)[1]
        omega_star = get_omegas_U(self.U)
        R = get_R(self.U, omega_star)
        num_iter = 0
        Q_u = jnp.sum(jnp.einsum('ab,ba->a', self.U, jnp.einsum('abc,ca->ab', R, self.U)))
        U_new = self.U
        Q_u_old_old = Q_u
        while not converged and num_iter < 50:
            # print(Q_u)
            U_old = U_new
            for iu in range(self.Du):
                U_not_i = jnp.delete(U_new, jnp.array([iu]), axis=1)
                V = self.partial_gs(U_not_i)
                # A = numpy.hstack([U_not_i, numpy.eye(self.Dx)[:,-self.Du-1:]])
                # V_full = numpy.linalg.qr(A)[0]
                # print(numpy.dot(V_full[:,:-self.Du-1].T, U_not_i))
                # V = V_full[:,-self.Du-1:]
                VRV = jnp.dot(jnp.dot(V.T, R[iu]), V)
                # VRV /= jnp.amax(VRV)
                # alpha = scipy.linalg.eigh(VRV, eigvals=(VRV.shape[0]-1,VRV.shape[0]-1))[1]
                alpha = jnp.real(jnp.linalg.eig(VRV)[1][:, :1])
                u_new = jnp.dot(V, alpha)[:, 0]
                if jnp.allclose(jnp.dot(U_not_i.T, u_new), 0, rtol=1e-4):
                    U_new = U_new.at[:, iu].set(u_new)
                else:
                    print('Warning: U not orthonormal')
            Q_u_old = Q_u
            Q_u = jnp.sum(jnp.einsum('ab,ba->a', U_new, jnp.einsum('abc,ca->ab', R, U_new)))
            converged = (Q_u - Q_u_old) < 1e-4
            num_iter += 1
        if (Q_u - Q_u_old_old) < 0:
            self.U = U_new
            print(jnp.dot(self.U.T, self.U))



    def update_U2(self, phi: 'GaussianDensity', X: jnp.ndarray):
        """ Updates the `U` by maximizing the Q-function. One component at a time is updated analytically.
        
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: jnp.ndarray [T, Dx]
            Data.
        """
        R = vmap(HCCovObservationModel.get_R, in_axes=(None, None, 0, 0, None, None, None),
                                        axis_name='i')(phi, X, self.W, self.beta, self.C, self.d,
                                                       self.sigma_x)
        U_new = jnp.empty(self.U.shape)

        for iu in range(self.Du):
            R_tmp = R[iu] / jnp.amax(R[iu])
            if iu > 0:
                U_not_i = U_new[:,:iu]
                # print(U_not_i)
                V = self.partial_gs(U_not_i)
                # A = jnp.hstack([U_not_i, jnp.eye(self.Dx)[:,-self.Du-1:]])
                # V_full = jnp.linalg.qr(A)[0]
                # print(jnp.dot(V_full[:,:-self.Du-1].T, U_not_i))
                # V = V_full[:,-self.Du-1:]
                # print(V)
                VRV = jnp.dot(jnp.dot(V.T, R_tmp), V)
                VRV /= jnp.amax(VRV)
                # alpha = scipy.linalg.eigh(VRV, eigvals=(VRV.shape[0]-1,VRV.shape[0]-1))[1]
                alpha = jnp.real(jnp.linalg.eig(VRV)[1])[:, :1]
                u_new = jnp.dot(V, alpha)[:, 0]
            else:
                u_new = jnp.real(jnp.linalg.eig(R_tmp)[1])[:, 0]
            # U_new = jnp.copy(self.U)
            # print(u_new)
            U_new = U_new.at[:, iu].set(u_new)

        if jnp.allclose(jnp.dot(U_new.T, U_new), jnp.eye(self.Du)):
            self.U = U_new
        else:
            print('Warning: U not orthonormal')
        # Q_u = jnp.sum(jnp.einsum('ab,ba->a', self.U, jnp.einsum('abc,ca->ab', R, self.U)))


    ####################### FUNCTIONS FOR OPTIMIZING sigma_x, beta, and W ##############################################


    ###################### Lower bound functions #######################################################################
    @staticmethod
    def get_omegas_i(phi: densities.GaussianDensity, X: jnp.ndarray, omega_star_init: jnp.ndarray,
                     W_i, u_i, beta, C, d, sigma_x, conv_crit: float = 1e-4):
        T = X.shape[0]
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        uC = jnp.dot(u_i.T, -C)
        ux_d = jnp.dot(u_i.T, (X - d).T)
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(phi.integrate('Ax_aBx_b_inner', A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i))
        # omega_star = 1e-8 * jnp.ones(T)
        omega_star = omega_star_init
        omega_old = 10 * jnp.ones(T)

        def body_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # From the lower bound term
            g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
            nu_plus = (1. - g_omega[:, None] * b_i) * w_i
            nu_minus = (-1. - g_omega[:, None] * b_i) * w_i
            ln_beta = - jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta)) - .5 * g_omega * (
                    b_i ** 2 - omega_star ** 2) + jnp.log(beta)
            ln_beta_plus = ln_beta + b_i
            ln_beta_minus = ln_beta - b_i
            # Create OneRankFactors
            exp_factor_plus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
            exp_factor_minus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
            # Create the two measures
            exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
            exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
            # Fourth order integrals E[h^2 (x-Cz-d)^2]
            quart_int_plus = exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC,
                                                    b_vec=ux_d.T,
                                                    C_mat=w_i, c_vec=b_i, D_mat=w_i, d_vec=b_i)
            quart_int_minus = exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC,
                                                      b_vec=ux_d.T,
                                                      C_mat=w_i, c_vec=b_i, D_mat=w_i, d_vec=b_i)
            quart_int = quart_int_plus + quart_int_minus
            # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate('Ax_aBx_b_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T)
            quad_int_minus = exp_phi_minus.integrate('Ax_aBx_b_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T)
            quad_int = quad_int_plus + quad_int_minus
            omega_old = omega_star
            omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))
            num_iter = num_iter + 1
            return omega_star, omega_old, num_iter

        def cond_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            return lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old)), 'i') > conv_crit
            #return jnp.logical_and(lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old) / omega_star), 'i') > conv_crit,
            #                       num_iter < 100)

        num_iter = 0
        init_val = (omega_star, omega_old, num_iter)
        omega_star, omega_old, num_iter = lax.while_loop(cond_fun, body_fun, init_val)

        return omega_dagger, omega_star, num_iter

    @staticmethod
    @partial(vmap, in_axes=(None, None, 0, 1, 0, 0, 0, None, None, None), out_axes=(0, 0))
    def get_lb_i(phi: densities.GaussianDensity, X: jnp.ndarray, W_i, u_i, beta, omega_dagger, omega_star, C, d, sigma_x):
        # beta = self.beta[iu:iu + 1]
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        T = X.shape[0]
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        uC = jnp.dot(u_i.T, -C)
        ux_d = jnp.dot(u_i.T, (X - d).T)
        # Lower bound for E[ln (sigma_x^2 + f(h))]

        omega_dagger = jnp.sqrt(phi.integrate('Ax_aBx_b_inner', A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i))
        g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
        nu_plus = (1. - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1. - g_omega[:, None] * b_i) * w_i
        ln_beta = - jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta)) - .5 * g_omega * (
                b_i ** 2 - omega_star ** 2) + jnp.log(beta)
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
        exp_factor_minus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
        # Create the two measures
        exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
        # Fourth order integrals E[h^2 (x-Cz-d)^2]
        quart_int_plus = exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC,
                                                b_vec=ux_d.T,
                                                C_mat=w_i, c_vec=b_i, D_mat=w_i, d_vec=b_i)
        quart_int_minus = exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC,
                                                  b_vec=ux_d.T,
                                                  C_mat=w_i, c_vec=b_i, D_mat=w_i, d_vec=b_i)
        quart_int = quart_int_plus + quart_int_minus
        # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
        quad_int_plus = exp_phi_plus.integrate('Ax_aBx_b_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T)
        quad_int_minus = exp_phi_minus.integrate('Ax_aBx_b_inner', A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T)
        quad_int = quad_int_plus + quad_int_minus
        omega_old = omega_star
        omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))

        f_omega_dagger = HCCovObservationModel.f(omega_dagger, beta)
        log_lb = jnp.log(sigma_x ** 2 + f_omega_dagger)
        g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
        nu_plus = (1. - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1. - g_omega[:, None] * b_i) * w_i
        ln_beta = - jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta)) - .5 * g_omega * (
                b_i ** 2 - omega_star ** 2) + jnp.log(beta)
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
        exp_factor_minus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
        # Create the two measures
        exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
        mat1 = -C
        vec1 = X - d
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R = R_plus + R_minus
        R = jnp.sum(R, axis=0)
        #R = .5 * (R + R.T)
        uRu = jnp.sum(u_i * jnp.dot(R, u_i))
        log_lb_sum = jnp.sum(log_lb)
        return uRu, log_lb_sum


    @staticmethod
    @partial(vmap, in_axes=(None, None, 0, 1, 0, 0, 0, None, None, None), out_axes=(0, 0))
    def get_R(phi: densities.GaussianDensity, X: jnp.ndarray, W_i, beta, u_i, omega_star, C, d, sigma_x):

        # Lower bound for E[ln (sigma_x^2 + f(h))]
        T = X.shape[0]
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
        nu_plus = (1. - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1. - g_omega[:, None] * b_i) * w_i
        ln_beta = - jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta)) - .5 * g_omega * (
                b_i ** 2 - omega_star ** 2) + jnp.log(beta)
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
        exp_factor_minus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
        # Create the two measures
        exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
        mat1 = -C
        vec1 = X - d
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R = R_plus + R_minus
        R = jnp.mean(R, axis=0)
        # R = R / beta
        R = .5 * (R + R.T)
        return R

    ####################### FUNCTIONS FOR OPTIMIZING U ################################################
    @staticmethod
    def gen_lin_ind_vecs(U):
        """ Generates linearly independent vectors from U.
        
        :param U: jnp.ndarray [N,M]
            Set of M vectors.
        
        :return: jnp.ndarray [N,N-M]
            return N-M linear inpendent vectors.
        """
        N, M = U.shape
        rand_vecs = jnp.array(np.random.rand(N, N - M))
        V_fixed = jnp.hstack([U, rand_vecs])
        V = V_fixed
        for m in range(N - M):
            v = rand_vecs[:,m]
            V = V.at[:,M+m].add(-jnp.dot(V_fixed.T, v) / jnp.sqrt(jnp.sum(v ** 2)))
        return V[:,M:]
    
    @staticmethod
    def proj(U, v):
        """ Projects v on U.
        
            proj_U(v) = (vU)/|U| U'
            
        :param U: jnp.ndarray [N,M]
            Set of M vectors.
        :param v: jnp.ndarray [N]
            Vector U is projected on.
        
        :return: jnp.ndarray [N]
            Projection of v on U.
        """
        return jnp.dot(jnp.dot(v, U) / jnp.linalg.norm(U, axis=0), U.T)

    def partial_gs(self, U):
        """ Partial Gram-Schmidt process, which generates orthonormal vectors (also to U).
        
        :param U: jnp.ndarray [N,M]
            Set of M orthonormal vectors.
            
        :return: jnp.ndarray [N,N-M]
            Orthonormal vectors (orthonormal to itself and U).
        """
        N, M = U.shape
        V = jnp.empty((N, N - M))
        I = jnp.tril(np.random.randn(N, N - M), -1)
        I = I + jnp.eye(N)[:,:N - M]
        # I = jnp.tri(N,N-M)
        #
        # I = self.gen_lin_ind_vecs(U)#jnp.random.randn(N,N-M)
        # I = jnp.eye(N)[:,M:]
        #I[-1,0] = 1
        for d in range(N - M):
            v = I[:,d]
            V = V.at[:,d].set(v - self.proj(U, v) - self.proj(V[:,:d], v))
            V = V.at[:,d].set(V[:,d] / jnp.sqrt(jnp.sum(V[:,d] ** 2)))
        return V
    
    ####################### Functions for bounds of non tractable terms in the Q-function ################################################  
    @staticmethod
    def f(h, beta):
        """ Computes the function
            
            f(h) = 2 * beta * cosh(h)
            
        :param h: jnp.ndarray
            Activation functions.
        :param beta: jnp.ndarray
            Scaling factor.
        
        :return: jnp.ndarray
            Evaluated functions.
        """
        return 2 * beta * jnp.cosh(h)

    @staticmethod
    def f_prime(h, beta):
        """ Computes the derivative of f
            
            f'(h) = 2 * beta * sinh(h)
            
        :param h: jnp.ndarray
            Activation functions.
        :param beta: jnp.ndarray
            Scaling factor.
        :return: jnp.ndarray
            Evaluated derivative functions.
        """
        return 2 * beta * jnp.sinh(h)

    @staticmethod
    def g(omega, beta, sigma_x):
        """ Computes the function
        
            g(omega) = f'(omega) / (sigma_x^2 + f(omega)) / |omega|
            
            for the variational boind
            
        :param omega: jnp.ndarray
            Free variational parameter.
        :param beta: jnp.ndarray
            Scaling factor.
        """
        return HCCovObservationModel.f_prime(omega, beta) / (sigma_x ** 2 + HCCovObservationModel.f(omega, beta)) / jnp.abs(omega)

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
        self.Theta = jnp.array(np.random.randn(self.Dx, self.Dphi))

    def compute_feature_vector(self, z: jnp.ndarray, ux: jnp.ndarray=None) -> jnp.ndarray:
        """ Constructs the feature vector
        
            phi_i(z) = (1,z,u_i)
            
        :param z: jnp.ndarray [T, Dz]
            Instantiation of latent variables.
        :param uz: jnp.ndarray [T, Dphi_u] or [T, Dphi_u, Dx]
            Control variables. (Default=None)
            
        :return: jnp.ndarray [T, Dphi, Dx]
            Feature vector.
        """
        T = z.shape[0]
        phi = jnp.zeros((T, self.Dphi, self.Dx))
        phi[:,0] = 1
        phi[:,1:self.Dz+1] = z
        if ux is not None:
            phi[:,self.Dz+1:] = ux
        return phi
    
    def compute_expected_feature_vector(self, density: 'GaussianDensity', ux: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the expected feature vector
        
            E[phi_i(z)] = (1,E[z],u_i)
            
        :param density: GaussianDensity
            Density over z.
        :param uz: jnp.ndarray [T, Dphi_u] or [T, Dphi_u, Dx]
            Control variables. (Default=None)
            
        :return: jnp.ndarray [T, Dx, Dphi]
            Expected feature vector.
        """
        T = density.R
        Ephi = jnp.zeros((T, self.Dx, self.Dphi))
        Ephi[:,:,0] = 1 
        Ephi[:,:,1:self.Dz+1] = density.integrate('x')[:,None]
        if ux is not None:
            Ephi[:,:,self.Dz+1:] = ux
        return Ephi
    
    def compute_expected_feature_outer_product(self, density: 'GaussianDensity', ux: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the expected feature vector
        
            E[phi_i(z) phi_i(z)'] = (1,    E[z'],   u_i',
                                     E[z], E[zz'],  E[z]u_i',
                                     u_i,  E[z]u_i, u_iu_i')
            
        :param density: GaussianDensity
            Density over z.
        :param ux: jnp.ndarray [T, Dx, Dphi_u] or [T, Dphi_u]
            Control variables. (Default=None)
            
        :return: jnp.ndarray [T,  Dx, Dphi, Dphi]
            Expected feature vector.
        """
        T = density.R
        
        Ez = density.integrate('x')
        Ezz = density.integrate('xx')
        
        Ephi_outer = jnp.zeros((T, self.Dx, self.Dphi, self.Dphi))
        Ephi_outer[:,:,0,0] = 1                                              # 1
        Ephi_outer[:,:,1:self.Dz+1,0] = Ez                                   # E[z']
        Ephi_outer[:,:,0,1:self.Dz+1] = Ez                                   # E[z]
        Ephi_outer[:,:,1:self.Dz+1,1:self.Dz+1] = Ezz                        # E[zz']
        if ux is not None:
            if ux.ndim == 2:
                ux = ux.reshape((ux.shape[0],1,ux.shape[1]))
            Ez_ux = Ez[:,None,:,None] * ux[:,:,None]
            uxux = ux[:,:,None] * ux[:,:,:,None]
            Ephi_outer[:,:,self.Dz+1:,0] = ux                                # u'
            Ephi_outer[:,:,0,self.Dz+1:] = ux                                # u
            Ephi_outer[:,:,1:self.Dz+1,self.Dz+1:] = Ez_ux                   # E[z] u'
            Ephi_outer[:,:,self.Dz+1:,1:self.Dz+1] = jnp.swapaxes(
                Ez_ux, axis1=2, axis2=3)                                     # E[z'] u
            Ephi_outer[:,:,self.Dz+1:,self.Dz+1:] = uxux                     # uu'
        return Ephi_outer
    
    def get_omega_star(self, density: 'GaussianDensity', x_t: jnp.ndarray, ux_t: jnp.ndarray=None, conv_crit: float=1e-4) -> jnp.ndarray:
        """ Gets the optimal variational parameter.
        """
        
        omega_old = jnp.ones((self.Dx))
        converged = False
        v = self.Theta[:,1:self.Dz+1]
        while not converged:
            g = 1. / jnp.abs(omega_old) * jnp.tanh(.5 * omega_old)
            sign = 2. * x_t[0] - 1.
            nu = self.Theta[:,1:self.Dz+1] * (.5 * sign - g * self.Theta[:,0])[:,None]
            #ln_beta = jnp.log(2 * jnp.cosh(.5 * omega_old)) + .5 * sign * self.Theta[:,0] - .5 * g * (self.Theta[:,0] ** 2 - omega_old ** 2)
            if ux_t is not None:
                theta_uz = jnp.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
                nu = nu - self.Theta[:,1:self.Dz+1] * (g * theta_ux)[:, None]
                #ln_beta = ln_beta + .5 * sign * theta_uz - g * (.5 * theta_uz ** 2 + self.Theta[:,0] * ux_t[0])           
                
            sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu)
            sigma_density = density.multiply(sigma_lb).get_density()
            A_mat = self.Theta[:,1:self.Dz+1]
            a_vec = self.Theta[:,0]
            if ux_t is not None:
                a_vec = a_vec + theta_ux
            Eh2 = sigma_density.integrate('Ax_aBx_b_inner', A_mat=A_mat, a_vec=a_vec, B_mat=A_mat, b_vec=a_vec)
            omega_star = jnp.sqrt(Eh2)
            omega_star[omega_star < 1e-10] = 1e-10
            converged = jnp.amax(jnp.abs(omega_star - omega_old)) < conv_crit
            omega_old = jnp.copy(omega_star)
        return omega_star
    
        
    def filtering(self, prediction_density: 'GaussianDensity', x_t: jnp.ndarray, ux_t: jnp.ndarray=None,**kwargs) -> 'GaussianDensity':
        """ Here the variational approximation of filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
            Observation.
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        omega_star = self.get_omega_star(prediction_density, x_t, ux_t)
        v = self.Theta[:,1:self.Dz+1]
        g = 1. / jnp.abs(omega_star) * jnp.tanh(.5 * omega_star)
        sign = 2. * x_t[0] - 1.
        nu = self.Theta[:,1:self.Dz+1] * (.5 * sign - g * self.Theta[:,0])[:,None]
        if ux_t is not None:
            theta_ux = jnp.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
            nu = nu - self.Theta[:,1:self.Dz+1] * (g * theta_ux)[:, None]
        sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu)
        filter_measure = prediction_density
        for idx in range(self.Dx):
            filter_measure = filter_measure.hadamard(sigma_lb.slice([idx]))
        filter_density = filter_measure.get_density()
        return filter_density
    
    def get_omega_dagger(self, density: 'GaussianDensity', ux_t: jnp.ndarray=None, conv_crit: float=1e-4) -> jnp.ndarray:
        """ Gets the optimal variational parameter.
        """
    
        A_mat = self.Theta[:,1:self.Dz+1]
        a_vec = self.Theta[:,0]
        if ux_t is not None:
            theta_ux = jnp.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
            a_vec = a_vec + theta_ux
        Eh2 = density.integrate('Ax_aBx_b_inner', A_mat=A_mat, a_vec=a_vec, B_mat=A_mat, b_vec=a_vec)
        omega_dagger = jnp.sqrt(Eh2)
        omega_dagger[omega_dagger < 1e-10] = 1e-10
        return omega_dagger
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: jnp.ndarray, u_x: jnp.ndarray=None,**kwargs):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        :param u_x: jnp.ndarray [T, ...]
            Control parameters. (Default=None)
        """ 
        A_theta = jnp.zeros((self.Dx, self.Dphi, self.Dphi))
        b_theta = jnp.zeros((self.Dx, self.Dphi))
        T = X.shape[0]
        for t in range(T):
            density_t = smoothing_density.slice([t+1])
            if u_x is not None:
                ux_t = u_x[t:t+1]
            else:
                ux_t = None
            omega_dagger = self.get_omega_dagger(density_t, ux_t=ux_t)
            g = 1. / jnp.abs(omega_dagger) * jnp.tanh(.5 * omega_dagger)
            Ephiphi = self.compute_expected_feature_outer_product(density_t, ux=ux_t)[0]
            A_theta = A_theta + g[:,None,None] * Ephiphi
            Ephi = self.compute_expected_feature_vector(density_t, ux=ux_t)[0]
            sign = 2. * X[t] - 1.
            b_theta = b_theta + .5 * sign[:,None] * Ephi
        #A_theta += 1e-4 * jnp.eye(self.Dphi)[None]
        self.Theta = jnp.linalg.solve(A_theta, b_theta)
        
    def get_lb_sigma(self, density: 'GaussianDensity', x_t: jnp.ndarray, ux_t: jnp.ndarray=None) -> jnp.ndarray:
        """ Computes the lower bounds for the data probability.
        """
        omega_star = self.get_omega_star(density, x_t, ux_t)
        v = self.Theta[:,1:self.Dz+1]
        g = 1. / jnp.abs(omega_star) * jnp.tanh(.5 * omega_star)
        sign = 2. * x_t[0] - 1.
        nu = self.Theta[:,1:self.Dz+1] * (.5 * sign - g * self.Theta[:,0])[:,None]
        ln_beta = - jnp.log(2) -  logcosh(.5 * omega_star) + .5 * sign * self.Theta[:,0] - .5 * g * (self.Theta[:,0] ** 2 - omega_star ** 2)
        if ux_t is not None:
            theta_ux = jnp.einsum('ab,b->a', self.Theta[:,self.Dz + 1:], ux_t[0])
            nu = nu - self.Theta[:,1:self.Dz+1] * (g * theta_ux)[:, None]
            ln_beta = ln_beta + .5 * sign * theta_ux - g * (.5 * theta_ux ** 2 + self.Theta[:,0] * ux_t[0])
        sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu, ln_beta=ln_beta)
        measure = density
        for idx in range(self.Dx):
            measure = measure.hadamard(sigma_lb.slice([idx]))
        prob_lb = measure.integrate()[0]
        return prob_lb

        
    def evaluate_llk(self, p_z: 'GaussianDensity', X: jnp.ndarray, u_x: jnp.ndarray=None,**kwargs) -> float:
        """ Computes the lower bound of log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: jnp.ndarray [T, Dx]
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
            llk += jnp.log(prob_lb)
        return llk