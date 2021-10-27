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
sys.path.append('../')
import scipy
from scipy.optimize import minimize, NonlinearConstraint
from jax import numpy as jnp
import numpy as np
from jax import lax
from jax import jit, value_and_grad, vmap

from src_jax import densities, conditionals, factors


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
    
    
    def __init__(self, Dx: int, Dz: int, Du: int, noise_x: float=.1):
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
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
        self.d = jnp.zeros(Dx)
        self.U = jnp.eye(Dx)[:,:Du]
        self.W = 1e-3 * jnp.array(np.random.randn(self.Du, self.Dz + 1))
        self.beta = noise_x ** 2 * jnp.ones(self.Du)
        self.sigma_x = noise_x
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

        phi = smoothing_density.slice(jnp.arange(1, smoothing_density.R))
        get_omegas = jit(vmap(HCCovObservationModel.get_omegas_i, in_axes=(None, None, 0, 1, 0, None, None, None),
                          out_axes=(0,0), axis_name='i'), static_argnums=(0))
        self.update_C(phi, X, get_omegas)
        self.update_d(phi, X, get_omegas)
        if self.Dx > 1:
            self.update_U3(phi, X, get_omegas)
            # if iteration % 2 == 0:
            #     self.update_U3(phi, X, get_omegas)
            # else:
            #     self.update_U4(phi, X, get_omegas)
        # for i in range(10):
        #     self.update_U3(phi, X, get_omegas)
        self.update_sigma_beta_W(phi, X, get_omegas)
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

    def update_C(self, phi: 'GaussianDensity', X: jnp.ndarray, get_omegas):
        """ Updates the `sigma_x`, `beta`, and `W` by maximizing the Q-function numerically (by L-BFGS-B).

        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: jnp.ndarray [T, Dx]
            Data.
        """
        x0 = np.array(self.C.flatten())
        objective = lambda C_flattened: self.value_and_grad_Qfunc_C(C_flattened, phi, X, self.W, self.U, self.beta,
                                                                    self.d, self.sigma_x, self.Dx, self.Dz, self.Du,
                                                                    get_omegas)
        objective(x0)
        # objective = jit(lambda x: value_and_grad(self.parameter_optimization_C(x, phi, X)))
        result = minimize(objective, x0, jac=True, method='L-BFGS-B', options={'disp': False, 'maxiter': 10})
        self.C = jnp.array(result.x.reshape((self.Dx, self.Dz)))

    def update_d(self, phi: 'GaussianDensity', X: jnp.ndarray, get_omegas):
        """ Updates the `d` by maximizing the Q-function analytically.
        
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: jnp.ndarray [T, Dx]
            Data.
        """
        T = X.shape[0]
        omega_dagger, omega_star = get_omegas(phi, X, self.W, self.U, self.beta, self.C, self.d, self.sigma_x)

        get_intDinv = vmap(HCCovObservationModel.get_intDinv, in_axes=(None, None, 0, 0, 0, None))
        intD_inv, intD_inv_z = get_intDinv(phi, X, self.W, self.beta, omega_star, self.sigma_x)
        Ez = phi.integrate('x')
        CEz = jnp.dot(self.C, jnp.sum(Ez,axis=0))
        UU = jnp.einsum('ab,cb->bac', self.U, self.U)
        A = jnp.eye(self.Dx) * T - jnp.sum(jnp.sum(intD_inv, axis=1)[:,None,None] * UU, axis=0)
        sum_X = jnp.sum(X, axis=0)
        intDinv_X_UU = jnp.sum(jnp.einsum('ab,abc->ac', jnp.einsum('ab,bc->ac',intD_inv, X), UU), axis=0)
        UU_C_intDinv_z = jnp.sum(jnp.einsum('abc,ac->ab', UU, jnp.einsum('ab,cb->ca', self.C, jnp.sum(intD_inv_z,axis=1))), axis=0)
        b = sum_X - intDinv_X_UU - CEz + UU_C_intDinv_z
        self.d = jnp.linalg.solve(A,b)

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

    def update_U3(self, phi: 'GaussianDensity', X: jnp.ndarray, get_omegas):
        x0 = np.array(self.U.flatten())
        triu_indices = np.triu_indices(self.Du)

        def constraint_fun(params):
            U = params.reshape((self.Dx, self.Du))
            return (np.eye(self.Du)-np.dot(U.T, U))[triu_indices]

        constraint = {'type': 'eq', 'fun': constraint_fun}

        get_R = lambda U, omega_star: vmap(HCCovObservationModel.get_R, in_axes=(None, None, 0, 0, 1, 0, None, None, None), axis_name='i')(phi, X, self.W, self.beta, U, omega_star, self.C, self.d, self.sigma_x)
        get_omegas_U = lambda U: get_omegas(phi, X, self.W, U, self.beta, self.C, self.d, self.sigma_x)[1]
        omega_star = get_omegas_U(self.U)
        R = get_R(self.U, omega_star)
        objective = lambda x: HCCovObservationModel.value_and_grad_QU(x, self.Du, self.Dx, R)
        result = minimize(objective, x0, method='SLSQP', jac=True, constraints=constraint, options={'disp': True})
        U_new = jnp.array(result.x.reshape((self.Dx, self.Du)))
        self.U = U_new
        print(jnp.dot(self.U.T, self.U))

    @staticmethod
    def value_and_grad_QU(params, Du, Dx, R):
        params_jax = jnp.array(params)
        val, grad = value_and_grad(HCCovObservationModel.Qfunc_U)(params_jax, R, Du, Dx)
        return val, np.array(grad)

    @staticmethod
    def Qfunc_U(params, R, Du, Dx):
        U = params.reshape((Dx, Du))
        Qu = -jnp.sum(jnp.einsum('ab,bc->b', U, jnp.einsum('abc,ca->ab', R, U)))
        return Qu


    def update_U(self, phi: 'GaussianDensity', X: jnp.ndarray, get_omegas):
        x0 = np.zeros(self.Du * self.Dx + self.Du * self.Du)
        x0[:self.Du * self.Dx] = np.array(self.U.flatten())
        get_R = lambda U, omega_star: vmap(HCCovObservationModel.get_R, in_axes=(None, None, 0, 0, 1, 0, None, None, None), axis_name='i')(phi, X, self.W, self.beta, U, omega_star, self.C, self.d, self.sigma_x)
        get_omegas_U = lambda U: get_omegas(phi, X, self.W, U, self.beta, self.C, self.d, self.sigma_x)[1]
        omega_star = get_omegas_U(self.U)
        R = get_R(self.U, omega_star)
        objective = lambda x: HCCovObservationModel.value_and_grad_Lagrange_U(x, self.Du, self.Dx, R, self.U)
        result = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'disp': False})
        U_new = jnp.array(result.x[:self.Du * self.Dx].reshape((self.Dx, self.Du)))
        self.U = U_new
        # if jnp.amax(jnp.abs(jnp.dot(U_new.T, U_new) - jnp.eye(self.Du))) < 1e-5:
        #     print('U updated')
        #     self.U = U_new
        # else:
        #     print('U not updated: eps=%.2f' %(1e5 * jnp.amax(jnp.abs(jnp.dot(U_new.T, U_new) - jnp.eye(self.Du)))))


    @staticmethod
    def value_and_grad_Lagrange_U(params, Du, Dx, R, U_old):
        params_jax = jnp.array(params)
        val, grad = value_and_grad(HCCovObservationModel.Lagrange_func_U)(params_jax, R, Du, Dx, U_old)
        return val, np.array(grad)

    @staticmethod
    def Lagrange_func_U(params, R, Du, Dx, U_old):
        U = params[:Du * Dx].reshape((Dx, Du))
        lagrange_multipliers = params[Du * Dx:].reshape((Du, Du))
        dL_dU = - jnp.einsum('abc,ca->ab', R, U) + jnp.dot(U, lagrange_multipliers).T
        dL_dmultipliers = jnp.dot(U.T, U) - jnp.eye(Du)
        L_U = jnp.sum(dL_dU ** 2) + jnp.sum(dL_dmultipliers ** 2)
        return L_U

    def update_sigma_beta_W(self, phi: 'GaussianDensity', X: jnp.ndarray, get_omegas):
        """ Updates the `sigma_x`, `beta`, and `W` by maximizing the Q-function numerically (by L-BFGS-B).

        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: jnp.ndarray [T, Dx]
            Data.
        """
        x0 = np.array(jnp.concatenate(
            [jnp.array([jnp.log(self.sigma_x ** 2)]), jnp.log(self.beta) - jnp.log(self.sigma_x ** 2),
             self.W.flatten()]))
        # x0 = np.array(self.W.flatten())
        bounds = [(None, 10)] + [(np.log(.25), 10)] * self.Du + [(None, None)] * (self.Du * (self.Dz + 1))
        # omega_dagger, omega_star = get_omegas(phi, X, self.W, self.U, self.beta, self.C, self.d, self.sigma_x)
        objective = lambda params: self.value_and_grad_Qfunc_sigma_beta_W(params, phi, X, self.U, self.C,
                                                                    self.d, self.Dx, self.Dz, self.Du, get_omegas)
        # objective = jit(lambda x: value_and_grad(self.parameter_optimization_C(x, phi, X)))
        result = minimize(objective, x0, jac=True, method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': 10})
        # result = minimize(objective, x0, jac=True, method='L-BFGS-B',
        #                   options={'disp': True, 'maxiter': 10})
        opt_params = result.x
        self.sigma_x = jnp.array(np.exp(.5 * opt_params[0]))
        self.beta = jnp.array(np.exp(opt_params[1:self.Du + 1] + opt_params[0]))
        self.W = jnp.array(opt_params[self.Du + 1:].reshape((self.Du, self.Dz + 1)))
        # self.W = jnp.array(opt_params.reshape((self.Du, self.Dz + 1)))
        # print(self.W)


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
    @staticmethod
    def Qfunc_sigma_beta_W(params: jnp.ndarray, phi: 'GaussianDensity', X: jnp.ndarray, U: jnp.ndarray, omega_star, C: jnp.array,
                           d: jnp.ndarray, Dx: int, Dz: int, Du: int):
        sigma_x = jnp.exp(.5 * params[0])
        beta = jnp.exp(params[1:Du + 1] + params[0])
        W = params[Du + 1:].reshape((Du, Dz + 1))
        # W = params.reshape((Du, Dz + 1))
        T = X.shape[0]

        vec = X - d
        E_epsilon2 = jnp.sum(phi.integrate('Ax_aBx_b_inner', A_mat=-C, a_vec=vec, B_mat=-C, b_vec=vec), axis=0)
        get_lb = vmap(HCCovObservationModel.get_lb_i, in_axes=(None, None, 0, 1, 0, 0, None, None, None),
                      out_axes=(0, 0))
        uRu, log_lb_sum = get_lb(phi, X, W, U, beta, omega_star, C, d, sigma_x)
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / sigma_x ** 2
        # determinant part
        Qm = Qm - .5 * E_ln_sigma2_f + .5 * T * (Du - Dx) * jnp.log(sigma_x ** 2)
        # constant part
        # Qm -= T * Dx * jnp.log(2 * jnp.pi)
        return - Qm / T

    @staticmethod
    def value_and_grad_Qfunc_sigma_beta_W(params: jnp.array, phi: densities.GaussianDensity, X: jnp.ndarray, U, C, d,
                                Dx, Dz, Du, get_omegas):
        params_jax = jnp.array(params)
        sigma_x = jnp.exp(.5 * params[0])
        beta = jnp.exp(params[1:Du + 1] + params[0])
        W = params_jax[Du + 1:].reshape((Du, Dz + 1))
        # W = params_jax.reshape((Du, Dz + 1))
        omega_dagger, omega_star = get_omegas(phi, X, W, U, beta, C, d, sigma_x)
        # omega_dagger, omega_star = HCCovObservationModel.get_omegas_i(phi, X, W, U, beta, C, d, sigma_x)
        val, grad = value_and_grad(HCCovObservationModel.Qfunc_sigma_beta_W)(params_jax, phi, X, U, omega_star, C, d,  Dx, Dz, Du)
        return val, np.array(grad)

    ###################### FUNCTIONS FOR OPTIMIZING C ##################################################################
    @staticmethod
    def value_and_grad_Qfunc_C(params: jnp.array, phi: densities.GaussianDensity, X: jnp.ndarray, W, U, beta, d, sigma_x, Dx, Dz, Du, get_omegas):
        C = jnp.array(params.reshape((Dx, Dz)))
        omega_dagger, omega_star = get_omegas(phi, X, W, U, beta, C, d, sigma_x)
        # omega_dagger, omega_star = HCCovObservationModel.get_omegas_i(phi, X, W, U, beta, C, d, sigma_x)
        val, grad = value_and_grad(HCCovObservationModel.Qfunc_C)(params, phi, X, W, U, beta, omega_star, d, sigma_x,
                                                      Dx, Dz, Du)
        return val, np.array(grad)

    @staticmethod
    def Qfunc_C(params: jnp.ndarray, phi: 'GaussianDensity', X: jnp.ndarray, W: jnp.ndarray, U: jnp.ndarray, beta: jnp.ndarray,
                omega_star, d: jnp.ndarray, sigma_x: float, Dx: int, Dz: int, Du: int):
        """ Computes (negative) Q-function (only terms depending on `sigma_x`, `beta`, and `W`) and the 
        corresponding gradients.
        
        :param params: jnp.ndarray [1+Du+Du*(Dz + 1)]
            Parameters in vector form. (log(sigma_x), log(beta), W)
        :param smoothing_density: GaussianDensity
            The smoothing density obtained in the E-step.
        :param X: jnp.ndarray [T, Dx]
            Data.
            
        :return: (float, jnp.ndarray [1+Du+Du*(Dz + 1)])
            Negative Q-function and gradients.
        """
        C = params.reshape((Dx, Dz))
        T = X.shape[0]
        # E[epsilon(z)^2]
        vec = X - d
        E_epsilon2 = jnp.sum(phi.integrate('Ax_aBx_b_inner', A_mat=-C, a_vec=vec, B_mat=-C, b_vec=vec), axis=0)
        get_lb = vmap(HCCovObservationModel.get_lb_i, in_axes=(None, None, 0, 1, 0, 0, None, None, None), out_axes=(0, 0))
        uRu, log_lb_sum = get_lb(phi, X, W, U, beta, omega_star, C, d, sigma_x)
        # data part
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / sigma_x ** 2
        # determinant part
        Qm -= .5 * E_ln_sigma2_f + .5 * T * (Dx - Du) * jnp.log(sigma_x ** 2)
        # constant part
        Qm -= T * Dx * jnp.log(2 * jnp.pi)
        return -Qm/ T

    ###################### Lower bound functions #######################################################################
    @staticmethod
    def get_omegas_i(phi: densities.GaussianDensity, X: jnp.ndarray,
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
        omega_star = omega_dagger
        omega_old = jnp.zeros(T)

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
            return jnp.logical_and(lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old) / omega_star), 'i') > conv_crit, num_iter < 1000)

        num_iter = 0
        init_val = (omega_star, omega_old, num_iter)
        omega_star = lax.while_loop(cond_fun, body_fun, init_val)[0]
        #
        # val = init_val
        # while cond_fun(val):
        #     val = body_fun(val)
        # omega_star = val[0]

        return omega_dagger, omega_star

    @staticmethod
    def get_lb_i(phi: densities.GaussianDensity, X: jnp.ndarray, W_i, u_i, beta, omega_star, C, d, sigma_x):
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
        omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))

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
        exp_phi_plus = phi.hadamard(exp_factor_plus)
        exp_phi_minus = phi.hadamard(exp_factor_minus)
        mat1 = -C
        vec1 = X - d
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R = R_plus + R_minus
        uRu = jnp.sum(u_i * jnp.dot(jnp.sum(R, axis=0), u_i))
        log_lb_sum = jnp.sum(log_lb)
        return uRu, log_lb_sum


    @staticmethod
    def get_R(phi: densities.GaussianDensity, X: jnp.ndarray, W_i, beta, u_i, omega_star, C, d, sigma_x):

        # Lower bound for E[ln (sigma_x^2 + f(h))]
        T = X.shape[0]
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        uC = jnp.dot(u_i.T, -C)
        ux_d = jnp.dot(u_i.T, (X - d).T)

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
        omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))

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
        # R = R / jnp.amax(R)
        return R

    @staticmethod
    def get_intDinv(phi: densities.GaussianDensity, X: jnp.ndarray, W_i, beta, omega_star, sigma_x):
        T = X.shape[0]
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        # beta = self.beta[iu:iu + 1]
        # Lower bound for E[ln (sigma_x^2 + f(h))]

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
        intD_inv_z_plus = exp_phi_plus.integrate('x')
        intD_inv_z_minus = exp_phi_minus.integrate('x')
        intD_inv_z = intD_inv_z_plus + intD_inv_z_minus
        intD_inv_plus = exp_phi_plus.integrate()
        intD_inv_minus = exp_phi_minus.integrate()
        intD_inv = intD_inv_plus + intD_inv_minus
        return intD_inv, intD_inv_z

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

'''
    def get_lb_i(self, iu: int, phi: densities.GaussianDensity, X: jnp.ndarray, conv_crit: float=1e-4, update: str=None):
        """ Computes the variational lower bounds for the log determinant and inverse term. In addition it provides terms, 
        that are needed for updating different model parameters.
        
        :param iu: int
            Index of the component the bound should be computed for.
        :param phi: GaussianDensity
            Density over latent variables.
        :param X: jnp.ndarray [T, Dx]
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
        v = jnp.tile(w_i, (T, 1))
        b_i = self.W[iu:iu+1,0]
        u_i = self.U[:,iu:iu+1]
        beta = self.beta[iu:iu+1]
        uC = jnp.dot(u_i.T, -self.C)
        ux_d = jnp.dot(u_i.T, X.T-self.d[:,None])
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(phi.integrate('Ax_aBx_b_inner', A_mat=w_i, a_vec=b_i,
                                                                  B_mat=w_i, b_vec=b_i))
        f_omega_dagger = self.f(omega_dagger, beta)
        log_lb = jnp.log(self.sigma_x ** 2 + f_omega_dagger)
        # Lower bound for E[f(h) / (sigma_x^2 + f(h)) * (u'epsilon(z))^2]
        omega_star = jnp.ones(T)
        converged = False
        num_iter = 0
        while not converged and num_iter < 50:
            # From the lower bound term
            g_omega = self.g(omega_star, beta)
            nu_plus = (1. - g_omega[:,None] * b_i) * w_i
            nu_minus = (-1. - g_omega[:,None] * b_i) * w_i
            ln_beta = - jnp.log(self.sigma_x ** 2 + self.f(omega_star, beta)) - .5 * g_omega * (b_i ** 2 - omega_star ** 2) + jnp.log(beta)
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
            #omega_star = jnp.amin([jnp.amax([jnp.sqrt(quart_int / quad_int), 1e-10]), 1e2])
            #quad_int[quad_int < 1e-10] = 1e-10
            omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))
            # For numerical stability
            #omega_star[omega_star < 1e-10] = 1e-10
            #omega_star[omega_star > 30] = 30
            #print(jnp.amax(jnp.abs(omega_star - omega_old)))
            converged = jnp.amax(jnp.abs(omega_star - omega_old)) < conv_crit
            num_iter += 1
        #print(jnp.amax(jnp.abs(omega_star - omega_old)))
        mat1 = -self.C
        vec1 = X - self.d[None]
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R = R_plus + R_minus
        uRu = jnp.sum(u_i * jnp.dot(jnp.sum(R, axis=0), u_i))
        log_lb_sum = jnp.sum(log_lb)
        if update == 'sigma_beta_W':
            uRu = jnp.sum(u_i * jnp.dot(jnp.sum(R, axis=0), u_i))
            ##### w_i gradiend ######################################################################
            # E[f'(h)exp(-k(h,omega^*)) dh/dw (u'epsilon(z))^2]
            # Matrix and vector for dh/dw
            dW = jnp.zeros((self.Dz + 1, self.Dz))
            dW[1:] = jnp.eye(self.Dz)
            db = jnp.zeros(self.Dz + 1)
            db[0] = 1
            dw_i = jnp.sum(exp_phi_plus.integrate('Ax_aBx_bCx_c_outer', A_mat=uC, a_vec=ux_d.T,
                                                    B_mat=uC, b_vec=ux_d.T, C_mat=dW, c_vec=db), axis=0)
            dw_i -= jnp.sum(exp_phi_minus.integrate('Ax_aBx_bCx_c_outer', A_mat=uC, a_vec=ux_d.T, 
                                                      B_mat=uC, b_vec=ux_d.T, C_mat=dW, c_vec=db), axis=0)
            # -g(omega) * E[f(h)exp(-k(h,omega^*)) h dh/dw (u'epsilon(z))^2]
            dw_i -= jnp.einsum('a,ab->b', g_omega, exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_outer', A_mat=w_i, a_vec=b_i,
                                                                            B_mat=uC, b_vec=ux_d.T, C_mat=uC, c_vec=ux_d.T,
                                                                            D_mat=dW, d_vec=db)[:,0])
            dw_i -= jnp.einsum('a,ab->b', g_omega, exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_outer', A_mat=w_i, a_vec=b_i,
                                                                             B_mat=uC, b_vec=ux_d.T, C_mat=uC, c_vec=ux_d.T,
                                                                             D_mat=dW, d_vec=db)[:,0])
            dw_i /= self.sigma_x ** 2
            # g(omega^+)E[h dh/dw]
            dw_i -= jnp.einsum('a,ab->b', self.g(omega_dagger, beta), phi.integrate('Ax_aBx_b_outer', A_mat=w_i, a_vec=b_i, 
                                                                                      B_mat=dW, b_vec=db)[:,0])
            dw_i /= 2.
            ###########################################################################################
            ##### beta_i gradient #####################################################################
            weighted_R = jnp.einsum('abc,a->bc', R, 1. / (self.sigma_x ** 2 + self.f(omega_star, beta))) 
            #  u'R u / (sigma_x^2 + f(omega^*))
            dln_beta_i = jnp.sum(u_i * jnp.dot(weighted_R, u_i))
            dln_beta_i -= jnp.sum(f_omega_dagger / (self.sigma_x ** 2 + f_omega_dagger))
            dln_beta_i /= 2.
            ##### sigma_x ** 2 gradient ###############################################################
            dlnsigma2 =  - uRu / self.sigma_x ** 2
            dlnsigma2 -= jnp.sum(u_i * jnp.dot(weighted_R, u_i))
            dlnsigma2 -= jnp.sum(self.sigma_x ** 2 / (self.sigma_x ** 2 + f_omega_dagger))
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
            return jnp.sum(R, axis=0)
        else:
            return uRu, log_lb_sum
'''

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