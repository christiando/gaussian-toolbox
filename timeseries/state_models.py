import numpy
import sys
sys.path.append('../src/')
import densities, conditionals


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
        
    def prediction(self, pre_filter_density: 'GaussianDensity') -> 'GaussianDensity':
        """ Here the prediction density is calculated.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1
        
        :param pre_filter_density: GaussianDensity
            Density p(z_t-1|x_{1:t-1})
            
        :return: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        """
        raise NotImplementedError('Prediction for state model not implemented.')
        
    def smoothing(self, cur_filter_density: 'GaussianDensity', 
                  post_smoothing_density: 'GaussianDensity') -> 'GaussianDensity':
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
                               two_step_smoothing_density: 'GaussianDensity'):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        raise NotImplementedError('Hyperparamers for state model not implemented.')
        
    def update_init_density(self, init_smooth_density: 'GaussianDensity') -> 'GaussianDensity':
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
        
    def prediction(self, pre_filter_density: 'GaussianDensity') -> 'GaussianDensity':
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
                  post_smoothing_density: 'GaussianDensity') -> 'GaussianDensity':
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
                               two_step_smoothing_density: 'GaussianDensity'):
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
        AEzzA = numpy.sum(numpy.einsum('abc,cd->abd', numpy.einsum('ab,cbd->cad', self.A, 
                                                                   Ezz[:-1]), self.A.T), axis=0)
        self.Qz = (numpy.sum(Ezz[1:], axis=0) + Az_b2 - mu_b - mu_b.T - AEzz_cross - AEzz_cross.T) / T
        
    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.ConditionalGaussianDensity(numpy.array([self.A]),
                                                                     numpy.array([self.b]),
                                                                     numpy.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.state_density.Lambda[0], self.state_density.ln_det_Sigma[0]
        
    def update_init_density(self, init_smooth_density: 'GaussianDensity') -> 'GaussianDensity':
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
        
        
        
        