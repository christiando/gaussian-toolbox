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

from autograd import numpy
from autograd import value_and_grad
from scipy.optimize import minimize
import sys
sys.path.append('../src/')
import densities, conditionals, factors


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
    
    def __init__(self, Dz: int, noise_z: float=1e-1):
        """ This implements a linear state transition model
        
            z_t = A z_{t-1} + b + zeta_t     with      zeta_t ~ N(0,Qz).
            
        :param Dz: int
            Dimensionality of latent space.
        :param noise_z: float
            Intial isoptropic std. on the state transition.
        """
        self.Dz = Dz
        self.Qz = noise_z ** 2 * numpy.eye(self.Dz)
        self.A, self.b = numpy.eye(self.Dz), numpy.zeros((self.Dz,))
        self.state_density = conditionals.ConditionalGaussianDensity(numpy.array([self.A]),
                                                                     numpy.array([self.b]), 
                                                                     numpy.array([self.Qz]))
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
        cur_smoothing_density = cur_two_step_smoothing_density.get_marginal(range(self.Dz,2*self.Dz))
        
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
        self.A = numpy.linalg.solve(numpy.sum(Ezz[:-1], axis=0), numpy.sum(Ezz_cross -  mu_b, axis=0)).T
        
    def update_b(self, smoothing_density: 'GaussianDensity'):
        """ The transition offset is updated here, where the the densities p(z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        """
        self.b = numpy.mean(smoothing_density.mu[1:] - numpy.dot(self.A, smoothing_density.mu[:-1].T).T, axis=0)
    
    def update_Qz(self, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity'):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = two_step_smoothing_density.R
        Ezz_two_step = two_step_smoothing_density.integrate('xx')
        Ezz = Ezz_two_step[:,self.Dz:,self.Dz:]
        Ezz_cross = Ezz_two_step[:,self.Dz:,:self.Dz]
        AEzz_cross = numpy.sum(numpy.einsum('ab,cbd->cad', self.A, Ezz_cross), axis=0)
        Ez_b = smoothing_density.mu[:,None] * self.b[None,:,None]
        AEz_b = numpy.sum(numpy.einsum('ab,cbd->cad', self.A, Ez_b[:-1]), axis=0)
        Az_b2 = numpy.sum(smoothing_density.integrate('Ax_aBx_b_outer', A_mat=self.A, a_vec=self.b, 
                                                      B_mat=self.A, b_vec=self.b)[:-1],axis=0)
        mu_b = numpy.sum(Ez_b[1:], axis=0)
        #AEzzA = numpy.einsum('ab,bc->ac', numpy.einsum('ab,bc->ac', self.A, numpy.sum(Ezz[:-1], axis=0)), self.A.T)
        #print(AEzzA, numpy.sum(Ezz[:-1], axis=0).shape)
        self.Qz = (numpy.sum(Ezz[1:], axis=0) + Az_b2 - mu_b - mu_b.T - AEzz_cross - AEzz_cross.T) / T
        #eigvals, eigvecs = numpy.linalg.eig(self.Qz)
        self.Qz += 1e-3 * numpy.eye(self.Dz) 
        
    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.ConditionalGaussianDensity(numpy.array([self.A]),
                                                                     numpy.array([self.b]),
                                                                     numpy.array([self.Qz]))
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
    
    def __init__(self, Dz: int, Dk: int, noise_z: float=1e-1):
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
        self.Qz = noise_z ** 2 * numpy.eye(self.Dz)
        self.A = numpy.zeros((self.Dz, self.Dphi))
        self.A[:,:self.Dz] = numpy.eye(self.Dz)
        self.b = numpy.zeros((self.Dz,))
        self.W = 1e-2 * numpy.random.randn(self.Dk, self.Dz + 1)
        self.state_density = conditionals.LSEMGaussianConditional(M=numpy.array([self.A]), 
                                                                  b=numpy.array([self.b]), 
                                                                  W=self.W, 
                                                                  Sigma=numpy.array([self.Qz]))
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
        self.update_A(smoothing_density, two_step_smoothing_density)
        self.update_b(smoothing_density)
        self.update_Qz(smoothing_density, two_step_smoothing_density)
        self.update_W(smoothing_density, two_step_smoothing_density)
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
        T = two_step_smoothing_density.R
        phi = smoothing_density.slice(range(T))
        
        # E[f(z)f(z)']
        Ekk = phi.multiply(self.state_density.k_func).multiply(self.state_density.k_func).integrate().reshape((T, 
                                                                                                               self.Dk, 
                                                                                                               self.Dk))
        Ekz = phi.multiply(self.state_density.k_func).integrate('x').reshape((T, 
                                                                              self.Dk, 
                                                                              self.Dz))
        Eff = numpy.empty((self.Dphi, self.Dphi))
        Eff[:self.Dz,:self.Dz] = numpy.sum(phi.integrate('xx'), axis=0)
        Eff[self.Dz:,self.Dz:] = numpy.sum(Ekk, axis=0)
        Eff[self.Dz:,:self.Dz] = numpy.sum(Ekz, axis=0)
        Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        # E[f(z)] b'
        Ez = numpy.sum(phi.integrate('x'), axis=0)
        Ek = numpy.sum(phi.multiply(self.state_density.k_func).integrate().reshape((T,self.Dk)), axis=0)
        Ef = numpy.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:,None]
        # E[z f(z)']
        v_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        v_joint[:,self.Dz:] = self.state_density.k_func.v
        nu_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        nu_joint[:,self.Dz:] = self.state_density.k_func.nu
        ln_beta = self.state_density.k_func.ln_beta
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=ln_beta)
        Ezz_cross = numpy.sum(two_step_smoothing_density.integrate('xx')[:,self.Dz:,:self.Dz], axis=0)
        Ezk = numpy.sum(two_step_smoothing_density.multiply(joint_k_func).integrate('x').reshape((T, self.Dk, 
                                                                                                 (2*self.Dz)))[:,:,:self.Dz], axis=0).T
        Ezf = numpy.concatenate([Ezz_cross, Ezk], axis=1)
        # This is added for stability. Could defined as prior. Check!
        Eff += 1e-6 * numpy.eye(self.Dphi)
        self.A = numpy.linalg.solve(Eff, (Ezf -  Ebf).T).T
        
    def update_b(self, smoothing_density: 'GaussianDensity'):
        """ The transition offset is updated here, where the the densities p(z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        Ez = smoothing_density.integrate('x')
        Ek = smoothing_density.multiply(self.state_density.k_func).integrate().reshape((T+1,self.Dk))
        Ef = numpy.concatenate([Ez, Ek], axis=1)
        self.b = numpy.mean(smoothing_density.mu[1:] - numpy.dot(self.A, Ef[:-1].T).T, axis=0)
        
    def update_Qz(self, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity'):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        Ezz = smoothing_density.integrate('xx')
        # E[zz']
        Ezz_sum = numpy.sum(Ezz[1:], axis=0)
        # E[z f(z)'] A'
        v_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        v_joint[:,self.Dz:] = self.state_density.k_func.v
        nu_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        nu_joint[:,self.Dz:] = self.state_density.k_func.nu
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta)
        Ezz_cross = two_step_smoothing_density.integrate('xx')[:,self.Dz:,:self.Dz]
        Ekz = two_step_smoothing_density.multiply(joint_k_func).integrate('x').reshape((T,self.Dk,int(2*self.Dz)))[:,:,:self.Dz]
        Ezf = numpy.concatenate([Ezz_cross, numpy.swapaxes(Ekz,1,2)], axis=2)
        EzfA = numpy.einsum('abc,dc->bd', Ezf, self.A)
        # E[z]b'
        Ezb = numpy.sum(smoothing_density.integrate('x')[1:,None] * self.b[None,:,None], axis=0)
        # A E[f(z)] b'
        Ez = numpy.sum(smoothing_density.integrate('x')[:-1], axis=0)
        Ek = numpy.sum(smoothing_density.multiply(self.state_density.k_func).integrate().reshape((T+1,self.Dk))[:-1], axis=0)
        Ef = numpy.concatenate([Ez, Ek])
        AEfb = numpy.dot(self.A, Ef)[:,None] * self.b[None]
        # A E[f(z)f(z)'] A'
        Ekk = smoothing_density.multiply(self.state_density.k_func).multiply(self.state_density.k_func).integrate().reshape((T+1, self.Dk, self.Dk))
        Ekz = smoothing_density.multiply(self.state_density.k_func).integrate('x').reshape((T+1, self.Dk, self.Dz))
        Eff = numpy.empty((self.Dphi, self.Dphi))
        Eff[:self.Dz,:self.Dz] = numpy.sum(Ezz[:-1], axis=0)
        Eff[self.Dz:,self.Dz:] = numpy.sum(Ekk[:-1], axis=0)
        Eff[self.Dz:,:self.Dz] = numpy.sum(Ekz[:-1], axis=0)
        Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        AEffA = numpy.dot(numpy.dot(self.A, Eff), self.A.T)
        self.Qz = (Ezz_sum - EzfA - EzfA.T + AEffA - Ezb - Ezb.T + AEfb + AEfb.T + T * self.b[:,None] * self.b[None]) / T
        # To make it stable (TO CHECK!!!)
        eigvals, eigvecs = numpy.linalg.eig(self.Qz)
        if any(eigvals <= 0):
            print('Warning: Qz not positive definite. Negative eigenvalues are set to small positive ones!')
        eigvals[eigvals < 0] = 1e-10
        self.Qz = numpy.dot(eigvecs, numpy.dot(numpy.diag(eigvals), numpy.linalg.inv(eigvecs)))
    
    def _Wfunc(self, W, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity') -> (float, numpy.ndarray):
        """ Computes the parts of the (negative) Q-fub
        
        :param W: numpy.ndarray [Dk, Dz + 1]
            The weights in the squared exponential of conditional mean function.
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
            
        :return: float
            Terms of negative Q-function depending on W.
        """
        self.W = W.reshape((self.Dk, self.Dz + 1))
        self.state_density.w0 = self.W[:,0]
        self.state_density.W = self.W[:,1:]
        self.state_density.update_phi()
        T = smoothing_density.R - 1
        Ezz = smoothing_density.integrate('xx')
        # E[z f(z)'] A'
        v_joint = numpy.zeros([self.Dk, self.Dz])
        v_joint = numpy.concatenate([v_joint, self.state_density.k_func.v], axis=1)
        nu_joint = numpy.zeros([self.Dk, self.Dz])
        nu_joint = numpy.concatenate([nu_joint, self.state_density.k_func.nu], axis=1)
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta)
        Ezz_cross = two_step_smoothing_density.integrate('xx')[:,self.Dz:,:self.Dz]
        Ekz = two_step_smoothing_density.multiply(joint_k_func).integrate('x').reshape((T,self.Dk,int(2*self.Dz)))[:,:,:self.Dz]
        Ezf = numpy.concatenate([Ezz_cross, numpy.swapaxes(Ekz,1,2)], axis=2)
        EzfA = numpy.einsum('abc,dc->bd', Ezf, self.A)
        # A E[f(z)] b'
        Ez = numpy.sum(smoothing_density.integrate('x')[:-1], axis=0)
        Ek = numpy.sum(smoothing_density.multiply(self.state_density.k_func).integrate().reshape((T+1,self.Dk))[:-1], axis=0)
        Ef = numpy.concatenate([Ez, Ek])
        AEfb = numpy.dot(self.A, Ef)[:,None] * self.b[None]
        # A E[f(z)f(z)'] A'
        Ekk = smoothing_density.multiply(self.state_density.k_func).multiply(self.state_density.k_func).integrate().reshape((T+1, self.Dk, self.Dk))
        Ekz = smoothing_density.multiply(self.state_density.k_func).integrate('x').reshape((T+1, self.Dk, self.Dz))
        #Eff = numpy.empty((self.Dphi, self.Dphi))
        #Eff[:self.Dz,:self.Dz] = numpy.sum(Ezz[:-1], axis=0)
        #Eff[self.Dz:,self.Dz:] = numpy.sum(Ekk[:-1], axis=0)
        #Eff[self.Dz:,:self.Dz] = numpy.sum(Ekz[:-1], axis=0)
        #Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        Eff1 = numpy.concatenate([numpy.sum(Ezz[:-1], axis=0), numpy.sum(Ekz[:-1], axis=0)], axis=0)
        Eff2 = numpy.concatenate([numpy.sum(Ekz[:-1], axis=0).T, numpy.sum(Ekk[:-1], axis=0)], axis=0)
        Eff = numpy.concatenate([Eff1, Eff2], axis=1)
        AEffA = numpy.dot(numpy.dot(self.A, Eff), self.A.T)
        Qfunc_W = .5 * numpy.trace(numpy.dot(self.Qz_inv, 
                                                   - EzfA - EzfA.T + AEffA + AEfb + 
                                                   AEfb.T))
    
        return Qfunc_W
        
        
    def update_W(self, smoothing_density: 'GaussianDensity', two_step_smoothing_density: 'GaussianDensity'):
        """ Updates the weights in the squared exponential of the state conditional mean.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        objective = lambda W: self._Wfunc(W, smoothing_density, two_step_smoothing_density)
        result = minimize(value_and_grad(objective), self.W.flatten(),
                          method='L-BFGS-B', jac=True, options={'disp': True})
        self.W = result.x.reshape((self.Dk, self.Dz + 1))
        
    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.LSEMGaussianConditional(M=numpy.array([self.A]), 
                                                                  b=numpy.array([self.b]), 
                                                                  W=self.W, 
                                                                  Sigma=numpy.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.state_density.Lambda[0], self.state_density.ln_det_Sigma[0]
        
    # TODO: Optimal initial state density