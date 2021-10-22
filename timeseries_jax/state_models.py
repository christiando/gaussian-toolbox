##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the class to fit state models that can be incroporated in the SSM-framwork.        #
#                                                                                                #
# Implemented so far:                                                                            #
#       + LinearStateModel (Gaussian Transition)                                                 #
#       + LSEMStateModel (Gaussian Transition with non linear mean)                              #
# Yet to be implemented:                                                                         #
#       - ControlledLinearStateModel (Gaussian Transition, with mean and covariance dependent on #
#                                     control variables)                                         #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
import sys
sys.path.append('../')
from jax import numpy as jnp
import numpy as np
from jax import jit, value_and_grad, grad
from scipy.optimize import minimize as minimize_sc
from jax.scipy.optimize import minimize
from jax.experimental import optimizers

from src_jax import densities, conditionals, factors


class StateModel:
    
    def __init__(self):
        """ This is the template class for state transition models in state space models. 
        Basically these classes should contain all functionality for transition between time steps
        the latent variables z, i.e. p(z_{t+1}|z_t). The object should 
        have an attribute `state_density`, which is be a `ConditionalDensity`. 
        Furthermore, it should be possible to optimize hyperparameters, when provided 
        with a density over the latent space.
        """
        self.emission_density = None
        
    def prediction(self, pre_filter_density: 'GaussianDensity', **kwargs) -> 'GaussianDensity':
        """ Here the prediction density is calculated.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1
        
        :param pre_filter_density: GaussianDensity
            Density p(z_t-1|x_{1:t-1})
            
        :return: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        """
        raise NotImplementedError('Prediction for state model not implemented.')
        
    def smoothing(self, cur_filter_density: 'GaussianDensity', 
                  post_smoothing_density: 'GaussianDensity', **kwargs) -> 'GaussianDensity':
        """ Here we do the smoothing step to acquire p(z_{t} | x_{1:T}), 
        given p(z_{t+1} | x_{1:T}) and p(z_{t} | x_{1:t}).
        
        :param cur_filter_density: GaussianDensity
            Density p(z_t|x_{1:t})
        :param post_smoothing_density: GaussianDensity
            Density p(z_{t+1}|x_{1:T})
            
        :return: [GaussianDensity, GaussianDensity]
            Smoothing density p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T})
        """
        raise NotImplementedError('Smoothing for state model not implemented.')
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', 
                               two_step_smoothing_density: 'GaussianDensity', **kwargs):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        raise NotImplementedError('Hyperparamers for state model not implemented.')
        
    def update_init_density(self, init_smooth_density: 'GaussianDensity', **kwargs) -> 'GaussianDensity':
        """ Finds the optimal distribution over the initial state z_0, 
        provided with the initial smoothing density.
        
        :param init_smooth_density: GaussianDensity
            Smoothing density over z_0.
        
        :return: GaussianDensity
            The optimal initial distribution.
        """
        raise NotImplementedError('Initial distribution update for state model not implemented.')
        
        
class LinearStateModel(StateModel):
    
    def __init__(self, Dz: int, noise_z: float=1.):
        """ This implements a linear state transition model
        
            z_t = A z_{t-1} + b + zeta_t     with      zeta_t ~ N(0,Qz).
            
        :param Dz: int
            Dimensionality of latent space.
        :param noise_z: float
            Intial isoptropic std. on the state transition.
        """
        self.Dz = Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.A, self.b = jnp.eye(self.Dz), jnp.zeros((self.Dz,))
        self.state_density = conditionals.ConditionalGaussianDensity(jnp.array([self.A]),
                                                                     jnp.array([self.b]), 
                                                                     jnp.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.state_density.Lambda[0], self.state_density.ln_det_Sigma[0]
        
    def prediction(self, pre_filter_density: 'GaussianDensity', **kwargs) -> 'GaussianDensity':
        """ Here the prediction density is calculated.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1
        
        :param pre_filter_density: GaussianDensity
            Density p(z_t-1|x_{1:t-1})
            
        :return: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        """
        # p(z_t|x_{1:t-1})
        return self.state_density.affine_marginal_transformation(pre_filter_density)
    
    def smoothing(self, cur_filter_density: 'GaussianDensity', 
                  post_smoothing_density: 'GaussianDensity', **kwargs) -> 'GaussianDensity':
        """ Here we do the smoothing step.
        
        First we calculate the backward density
        
        $$
        p(z_{t} | z_{t+1}, x_{1:t}) = p(z_{t+1}|z_t)p(z_t | x_{1:t}) / p(z_{t+1}| x_{1:t}) 
        $$
        
        and finally we get the smoothing density
        
        $$
        p(z_{t} | x_{1:T}) = int p(z_{t} | z_{t+1}, x_{1:t}) p(z_{t+1}|x_{1:T}) dz_{t+1}
        $$
        
        :param cur_filter_density: GaussianDensity
            Density p(z_t|x_{1:t})
        :param post_smoothing_density: GaussianDensity
            Density p(z_{t+1}|x_{1:T})
            
        :return: [GaussianDensity, GaussianDensity]
            Smoothing density p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T})
        """
        # p(z_{t} | z_{t+1}, x_{1:t}) 
        backward_density = self.state_density.affine_conditional_transformation(cur_filter_density)
        # p(z_{t}, z_{t+1} | x_{1:T})
        cur_two_step_smoothing_density = backward_density.affine_joint_transformation(post_smoothing_density)
        # p(z_{t} | x_{1:T})
        cur_smoothing_density = cur_two_step_smoothing_density.get_marginal(jnp.arange(self.Dz, 2*self.Dz))
        
        return cur_smoothing_density, cur_two_step_smoothing_density
    
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', 
                               two_step_smoothing_density: 'GaussianDensity', **kwargs):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        self.update_A(smoothing_density, two_step_smoothing_density)
        self.update_b(smoothing_density)
        self.update_Qz(smoothing_density, two_step_smoothing_density)
        self.update_state_density()
        
    def update_A(self, smoothing_density: 'GaussianDensity', 
                 two_step_smoothing_density: 'GaussianDensity'):
        """ The transition matrix is updated here, where the the densities
        p(z_{t+1}, z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        #Ezz = smoothing_density.integrate('xx')
        mu_b = smoothing_density.mu[:-1,None] * self.b[None,:,None]
        Ezz_two_step = two_step_smoothing_density.integrate('xx')
        Ezz = Ezz_two_step[:,self.Dz:,self.Dz:]
        Ezz_cross = Ezz_two_step[:,self.Dz:,:self.Dz]
        A = jnp.mean(Ezz, axis=0) #+ 1e-2 * jnp.eye(self.Dz)
        self.A = jnp.linalg.solve(A, jnp.mean(Ezz_cross -  mu_b, axis=0)).T
        
    def update_b(self, smoothing_density: 'GaussianDensity'):
        """ The transition offset is updated here, where the the densities p(z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        """
        self.b = jnp.mean(smoothing_density.mu[1:] - jnp.dot(self.A, smoothing_density.mu[:-1].T).T, axis=0)
    
    def update_Qz(self, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity'):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        A_tilde = jnp.eye(2*self.Dz, self.Dz)
        A_tilde = A_tilde.at[self.Dz:].set(-self.A.T)
        # A_tilde = jnp.block([[jnp.eye(self.Dz), -self.A.T]])
        b_tilde = -self.b
        self.Qz = jnp.mean(two_step_smoothing_density.integrate('Ax_aBx_b_outer', 
                                                                  A_mat=A_tilde.T, 
                                                                  a_vec=b_tilde, 
                                                                  B_mat=A_tilde.T, 
                                                                  b_vec=b_tilde), axis=0)
        
    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.ConditionalGaussianDensity(jnp.array([self.A]),
                                                                     jnp.array([self.b]),
                                                                     jnp.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.state_density.Lambda[0], self.state_density.ln_det_Sigma[0]
        
    def update_init_density(self, init_smooth_density: 'GaussianDensity', **kwargs) -> 'GaussianDensity':
        """ Finds the optimal distribution over the initial state z_0, 
        provided with the initial smoothing density.
        
        :param init_smooth_density: GaussianDensity
            Smoothing density over z_0.
        
        :return: GaussianDensity
            The optimal initial distribution.
        """
        mu0 = init_smooth_density.integrate('x')
        Sigma0 = init_smooth_density.integrate('Ax_aBx_b_outer', A_mat=None, a_vec=-mu0[0], 
                                               B_mat=None, b_vec=-mu0[0])
        opt_init_density = densities.GaussianDensity(Sigma0, mu0)
        return opt_init_density
        
        
class LSEMStateModel(LinearStateModel):
    
    def __init__(self, Dz: int, Dk: int, noise_z: float=1.):
        """ This implements a linear+squared exponential mean (LSEM) state model
        
            z_t = A phi(z_{t-1}) + b + zeta_t     with      zeta_t ~ N(0,Qz).
            
            The feature function is 
            
            f(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).
            
            The kernel and linear activation function are given by
            
            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.
            
            Hence, the 
            
        :param Dz: int
            Dimensionality of latent space.
        :param Dk: int
            Number of kernels to use.
        :param noise_z: float
            Intial isoptropic std. on the state transition.
        """
        self.Dz, self.Dk = Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.A = jnp.array(np.random.randn(self.Dz, self.Dphi))
        self.A = self.A.at[:,:self.Dz].set(jnp.eye(self.Dz))
        self.b = jnp.zeros((self.Dz,))
        self.W = jnp.array(np.random.randn(self.Dk, self.Dz + 1))
        self.state_density = conditionals.LSEMGaussianConditional(M=jnp.array([self.A]), 
                                                                  b=jnp.array([self.b]), 
                                                                  W=self.W, 
                                                                  Sigma=jnp.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.state_density.Lambda[0], self.state_density.ln_det_Sigma[0]
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', 
                               two_step_smoothing_density: 'GaussianDensity', **kwargs):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms).
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        # self.Qz = jit(self.update_Qz, static_argnums=(0,1))(smoothing_density, two_step_smoothing_density)
        # self.update_state_density()
        # self.A = jit(self.update_A, static_argnums=(0,1))(smoothing_density, two_step_smoothing_density)
        # self.b = jit(self.update_b, static_argnums=(0))(smoothing_density)
        # time_start = time.perf_counter()
        self.A, self.b, self.Qz = self.update_AbQ(smoothing_density, two_step_smoothing_density)
        # print(self.b)
        # print('AbQ: Run Time %.1f' % (time.perf_counter() - time_start))
        # time_start = time.perf_counter()
        self.update_state_density()
        self.update_W(smoothing_density, two_step_smoothing_density)
        # print('W: Run Time %.1f' % (time.perf_counter() - time_start))
        # time_start = time.perf_counter()
        self.update_state_density()
        # print('Update: Run Time %.1f' % (time.perf_counter() - time_start))

    def update_AbQ(self, smoothing_density: 'GaussianDensity',
                 two_step_smoothing_density: 'GaussianDensity'):
        """ The transition matrix is updated here, where the the densities
        p(z_{t+1}, z_t|x_{1:T}) is provided.

        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        phi = smoothing_density.slice(jnp.arange(T))

        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:, :self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.mean(two_step_smoothing_density.integrate('Ax_aBx_b_outer',
                                                               A_mat=A_tilde.T,
                                                               a_vec=b_tilde,
                                                               B_mat=A_tilde.T,
                                                               b_vec=b_tilde), axis=0)
        # v_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                      self.state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                       self.state_density.k_func.nu])
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:, self.Dz:].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz:].set(self.state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta)
        two_step_k_measure = two_step_smoothing_density.multiply(joint_k_func, update_full=True)
        Ekz = jnp.mean(two_step_k_measure.integrate('x').reshape((T, self.Dk, 2 * self.Dz)), axis=0)
        Ekz_future, Ekz_past = Ekz[:, :self.Dz], Ekz[:, self.Dz:]
        phi_k = phi.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.mean(phi_k.integral_light().reshape((T, self.Dk)), axis=0)
        Qz_k_lin_err = jnp.dot(self.A[:, self.Dz:],
                               (Ekz_future - jnp.dot(self.A[:, :self.Dz], Ekz_past.T).T - Ek[:, None] *
                                self.b[None]))
        mean_Ekk = jnp.mean(phi_k.multiply(self.state_density.k_func, update_full=True).integral_light().reshape(
            (T, self.Dk, self.Dk)), axis=0)
        Qz_kk = jnp.dot(jnp.dot(self.A[:, self.Dz:], mean_Ekk), self.A[:, self.Dz:].T)
        Qz =  Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T

        # E[f(z)f(z)']
        Ekk = phi_k.multiply(self.state_density.k_func, update_full=True).integral_light().reshape((T, self.Dk, self.Dk))
        Ekz = phi_k.integrate('x').reshape((T, self.Dk, self.Dz))
        mean_Ekz = jnp.mean(Ekz, axis=0)
        mean_Ezz = jnp.mean(phi.integrate('xx'), axis=0)
        mean_Ekk = jnp.mean(Ekk, axis=0) + .0001 * jnp.eye(self.Dk)
        Eff = jnp.block([[mean_Ezz, mean_Ekz.T],
                         [mean_Ekz, mean_Ekk]])

        # mean_Ekk_reg = mean_Ekk + .0001 * jnp.eye(self.Dk)
        # mean_Ezz = jnp.mean(phi.integrate('xx'), axis=0)
        # Eff = jnp.block([[mean_Ezz, Ekz_past.T],
        #                  [Ekz_past, mean_Ekk_reg]])
        Ez = jnp.mean(phi.integrate('x'), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:, None]
        # E[z f(z)']
        Ezz_cross = jnp.mean(two_step_smoothing_density.integrate('xx')[:, self.Dz:, :self.Dz], axis=0)
        Ezf = jnp.concatenate([Ezz_cross.T, Ekz_future.T], axis=1)
        A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        b =  jnp.mean(smoothing_density.mu[1:], axis=0) - jnp.dot(A, Ef).T
        # A = self.A
        # b = self.b
        return A, b, Qz

    def update_A(self, smoothing_density: 'GaussianDensity',
                 two_step_smoothing_density: 'GaussianDensity'):
        """ The transition matrix is updated here, where the the densities
        p(z_{t+1}, z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = two_step_smoothing_density.R
        phi = smoothing_density.slice(jnp.arange(T))
        
        # E[f(z)f(z)']
        phi_k = phi.multiply(self.state_density.k_func, update_full=True)
        Ekk = phi_k.multiply(self.state_density.k_func, update_full=True).integral_light().reshape((T, self.Dk, self.Dk))
        Ekz = phi_k.integrate('x').reshape((T, self.Dk, self.Dz))
        mean_Ekz = jnp.mean(Ekz, axis=0)
        mean_Ezz = jnp.mean(phi.integrate('xx'), axis=0)
        mean_Ekk = jnp.mean(Ekk, axis=0) + .0001 * jnp.eye(self.Dk)
        Eff = jnp.block([[mean_Ezz, mean_Ekz.T],
                         [mean_Ekz, mean_Ekk]])
        # Eff = jnp.empty((self.Dphi, self.Dphi))
        # Eff[:self.Dz,:self.Dz] = jnp.mean(phi.integrate('xx'), axis=0)
        # Eff[self.Dz:,self.Dz:] = jnp.mean(Ekk, axis=0)
        # Eff[self.Dz:,:self.Dz] = jnp.mean(Ekz, axis=0)
        # Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        # E[f(z)] b'
        Ez = jnp.mean(phi.integrate('x'), axis=0)
        Ek = jnp.mean(phi_k.integral_light().reshape((T,self.Dk)), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:,None]
        # E[z f(z)']
        # v_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                      self.state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                       self.state_density.k_func.nu])
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:,self.Dz:].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz:].set(self.state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta)
        Ezz_cross = jnp.mean(two_step_smoothing_density.integrate('xx')[:,self.Dz:,:self.Dz], axis=0)
        Ezk = jnp.mean(two_step_smoothing_density.multiply(joint_k_func, update_full=True).integrate('x').reshape((T, self.Dk,
                                                                                                 (2*self.Dz)))[:,:,:self.Dz], axis=0).T
        Ezf = jnp.concatenate([Ezz_cross.T, Ezk], axis=1)
        return jnp.linalg.solve(Eff / T, (Ezf -  Ebf).T / T).T

    def update_b(self, smoothing_density: 'GaussianDensity'):
        """ The transition offset is updated here, where the the densities p(z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        Ez = smoothing_density.integrate('x')
        Ek = smoothing_density.multiply(self.state_density.k_func, update_full=True).integral_light().reshape((T+1,self.Dk))
        Ef = jnp.concatenate([Ez, Ek], axis=1)
        return jnp.mean(smoothing_density.mu[1:] - jnp.dot(self.A, Ef[:-1].T).T, axis=0)

    def update_Qz(self, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity'):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:,:self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.mean(two_step_smoothing_density.integrate('Ax_aBx_b_outer', 
                                                                  A_mat=A_tilde.T, 
                                                                  a_vec=b_tilde, 
                                                                  B_mat=A_tilde.T, 
                                                                  b_vec=b_tilde), axis=0)
        # v_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                      self.state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                       self.state_density.k_func.nu])
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:,self.Dz:].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz:].set(self.state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta)
        two_step_k_measure = two_step_smoothing_density.multiply(joint_k_func, update_full=True)
        Ekz = jnp.mean(two_step_k_measure.integrate('x').reshape((T, self.Dk, 2*self.Dz)), axis=0)
        phi_k = smoothing_density.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.mean(phi_k.integral_light().reshape((T+1, self.Dk))[:-1], axis=0)
        Qz_k_lin_err = jnp.dot(self.A[:,self.Dz:], 
                  (Ekz[:,:self.Dz] - jnp.dot(self.A[:,:self.Dz], Ekz[:,self.Dz:].T).T - Ek[:,None] * self.b[None]))
        Ekk = phi_k.multiply(self.state_density.k_func, update_full=True).integral_light().reshape((T+1, self.Dk, self.Dk))
        Qz_kk = jnp.dot(jnp.dot(self.A[:,self.Dz:], jnp.mean(Ekk[:-1], axis=0)), self.A[:,self.Dz:].T)
        return Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T
    
    @staticmethod
    def _Wfunc2(W, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity', A, b, Qz, Qz_inv, Dk, Dz) -> (float, jnp.ndarray):
        """ Computes the parts of the (negative) Q-fub

        :param W: jnp.ndarray [Dk, Dz + 1]
            The weights in the squared exponential of conditional mean function.
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).

        :return: float
            Terms of negative Q-function depending on W.
        """
        W = jnp.reshape(W, (Dk, Dz + 1))
        # print(W.shape)
        state_density = conditionals.LSEMGaussianConditional(M=jnp.array([A]),
                                                             b=jnp.array([b]),
                                                             W=W,
                                                             Sigma=jnp.array([Qz]))
        # self.state_density.update_phi()
        T = smoothing_density.R - 1
        # E[z f(z)'] A'
        # v_joint = jnp.block([jnp.zeros([Dk, int(Dz)]),
        #                      state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([Dk, int(Dz)]),
        #                       state_density.k_func.nu])
        zero_arr = jnp.zeros([Dk, 2 * Dz])
        v_joint = zero_arr.at[:,Dz:].set(state_density.k_func.v)
        nu_joint = zero_arr.at[:, Dz:].set(state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=state_density.k_func.ln_beta)
        two_step_k_measure = two_step_smoothing_density.multiply(joint_k_func, update_full=True)
        Ekz = jnp.mean(jnp.reshape(two_step_k_measure.integrate('x'), (T, Dk, 2*Dz)), axis=0)
        #Ek = jnp.mean(two_step_k_measure.integrate().reshape((T, self.Dk)), axis=0)
        Ek = jnp.mean(jnp.reshape(smoothing_density.multiply(
            state_density.k_func).integrate(), (T + 1, Dk))[:-1], axis=0)
        Qz_k_lin_err = jnp.dot(A[:,Dz:], (Ekz[:,:Dz] - jnp.dot(Ekz[:,Dz:], A[:,:Dz].T) - Ek[:,None] * b[None]))
        Ekk = jnp.reshape(smoothing_density.multiply(state_density.k_func, update_full=True).multiply(state_density.k_func, update_full=True).integrate(), (T + 1, Dk, Dk))
        Qz_kk = jnp.dot(jnp.dot(A[:,Dz:], jnp.mean(Ekk[:-1], axis=0)), A[:,Dz:].T)
        Qfunc_W = .5 * jnp.trace(jnp.dot(Qz_inv, Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T))
        return Qfunc_W
    

    @staticmethod
    def _Wfunc(W, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity', A, b, Qz, Qz_inv,
               ln_det_Qz, Dk, Dz) -> (float, jnp.ndarray):
        """ Computes the parts of the (negative) Q-fub

        :param W: jnp.ndarray [Dk, Dz + 1]
            The weights in the squared exponential of conditional mean function.
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).

        :return: float
            Terms of negative Q-function depending on W.
        """
        W = jnp.reshape(W, (Dk, Dz + 1))
        # print(W.shape)
        state_density = conditionals.LSEMGaussianConditional(M=jnp.array([A]),
                                                             b=jnp.array([b]),
                                                             W=W,
                                                             Sigma=jnp.array([Qz]),
                                                             Lambda=jnp.array([Qz_inv]),
                                                             ln_det_Sigma=jnp.array([ln_det_Qz]))
        # self.state_density.update_phi()
        T = smoothing_density.R
        A_lower = A[:, Dz:]
        # E[z f(z)'] A'
        # v_joint = jnp.block([jnp.zeros([Dk, Dz]),
        #                      state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([Dk, Dz]),
        #                       state_density.k_func.nu])
        zero_arr = jnp.zeros([Dk, 2 * Dz])
        v_joint = zero_arr.at[:,Dz:].set(state_density.k_func.v)
        nu_joint = zero_arr.at[:, Dz:].set(state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=state_density.k_func.ln_beta)
        Ekz = jnp.sum(jnp.reshape(two_step_smoothing_density.multiply(joint_k_func, update_full=True).integrate('x'), (T, Dk, 2 * Dz)), axis=0)
        phi_k = smoothing_density.multiply(state_density.k_func, update_full=True)
        Ek = jnp.sum(jnp.reshape(phi_k.integral_light(), (T, Dk)), axis=0)
        Qz_k_lin_err = jnp.dot(A_lower, jnp.subtract(jnp.subtract(Ekz[:, :Dz], jnp.dot(Ekz[:, Dz:], A[:, :Dz].T)), jnp.outer(Ek, b)))
        Ekk = jnp.sum(jnp.reshape(phi_k.multiply(state_density.k_func, update_full=True).integral_light(), (T, Dk, Dk)), axis=0)
        Qz_kk = jnp.dot(jnp.dot(A_lower, Ekk), A_lower.T)
        Qfunc_W = .5 * jnp.trace(jnp.dot(Qz_inv, jnp.subtract(Qz_kk, jnp.subtract(Qz_k_lin_err, Qz_k_lin_err.T))))
        return Qfunc_W


    def update_W(self, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity'):
        """ Updates the weights in the squared exponential of the state conditional mean.

        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        #  This compiling takes a lot of time, and is only worth it for several iterations
        # func = value_and_grad(jit(lambda W: self._Wfunc(W, smoothing_density, two_step_smoothing_density, self.A, self.b, self.Qz, self.Qz_inv, self.Dk, self.Dz)))
        # def objective(W):
        #     obj, grads = func(jnp.array(W))
        #     return obj, np.array(grads)
        # result = minimize_sc(objective, np.array(self.W.flatten()), method='BFGS', jac=True,
        #                   options={'disp': True, 'maxiter': 10})
        # self.W = jnp.array(result.x.reshape((self.Dk, self.Dz + 1)))
        phi = smoothing_density.slice(jnp.arange(0,smoothing_density.R - 1))
        # func = jit(lambda W: self._Wfunc(W, phi, two_step_smoothing_density, self.A, self.b, self.Qz,
        #                           self.Qz_inv, self.ln_det_Qz, self.Dk, self.Dz))
        # result = minimize(func, self.W.flatten(), method='BFGS', options={'maxiter': 20})
        # self.W = result.x.reshape((self.Dk, self.Dz + 1))
        func = jit(grad(lambda W: self._Wfunc(W, phi, two_step_smoothing_density, self.A, self.b, self.Qz,
                                  self.Qz_inv, self.ln_det_Qz, self.Dk, self.Dz)))
        W = self.W.flatten()
        # for i in range(10):
        #     v, g = func(W)
        #     W = W - 1e-1 * g
        # self.W = W.reshape((self.Dk, self.Dz + 1))

        opt_init, opt_update, get_params = optimizers.adam(1e-1)
        opt_state = opt_init(W)

        def step(step, opt_state):
            grads = func(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return opt_state

        for i in range(10):
            opt_state = step(i, opt_state)
        self.W = get_params(opt_state).reshape((self.Dk, self.Dz + 1))

    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.LSEMGaussianConditional(M=jnp.array([self.A]), 
                                                                  b=jnp.array([self.b]), 
                                                                  W=self.W, 
                                                                  Sigma=jnp.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.state_density.Lambda[0], self.state_density.ln_det_Sigma[0]
        
    # TODO: Optimal initial state density