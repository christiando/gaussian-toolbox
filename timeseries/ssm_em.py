#import numpy
import autograd.numpy as numpy
import observation_models, state_models
import sys
sys.path.append('../src/')
import densities

class StateSpaceEM:
    
    def __init__(self, X: numpy.ndarray, observation_model: observation_models.ObservationModel, 
                 state_model: state_models.StateModel, max_iter: int=100, conv_crit: float=1e-4):
        self.X = X
        self.T, self.Dx = self.X.shape
        self.Dz = state_model.Dz
        # observation model
        self.om = observation_model
        # state model
        self.sm = state_model
        self.max_iter = max_iter
        self.conv_crit = conv_crit
        self.iteration = 0
        self.llk_list = []
        # Setup densities
        self.prediction_density = self._setup_density()
        self.filter_density = self._setup_density()
        self.smoothing_density = self._setup_density()
        self.twostep_smoothing_density = self._setup_density(D= int(2*self.Dz))
        self.twostep_smoothing_density = self.twostep_smoothing_density.slice(range(self.T))
        
    def _setup_density(self, D: int=None) -> densities.GaussianDensity:
        """ Initializes a density object (with uniform densities).
        """
        if D is None:
            D = self.Dz
        Sigma = numpy.tile(numpy.eye(D)[None], (self.T+1,1,1))
        Lambda = numpy.tile(numpy.eye(D)[None], (self.T+1,1,1))
        mu = numpy.zeros((self.T + 1, D))
        ln_det_Sigma = D * numpy.log(numpy.ones(self.T+1))
        return densities.GaussianDensity(Sigma, mu, Lambda, ln_det_Sigma)
        
    def run(self):
        converged = False
        while self.iteration < self.max_iter and not converged:
            self.estep()
            self.llk_list.append(self.compute_log_likelihood())
            self.mstep()
            if self.iteration>1:
                conv = (self.llk_list[-1] - self.llk_list[-2]) / numpy.amax([1, 
                                                                             numpy.abs(self.llk_list[-1]), 
                                                                             numpy.abs(self.llk_list[-2])])
                converged = conv < self.conv_crit
            self.iteration += 1
            if self.iteration % 10 == 0:
                print('Iteration %d - llk=%.1f' %(self.iteration, self.llk_list[-1]))
        if not converged:
            print('EM reached the maximal number of iterations.')
        else:
            print('EM did converge.')
    
    
    def estep(self):
        """ Performs the expectation step, i.e. the forward-backward algorithm.
        """
        self.forward_path()
        self.backward_path()
        
    def mstep(self):
        """ Performs the maximization step, i.e. the updates of model parameters.
        """
        # Update parameters of state model
        self.sm.update_hyperparameters(self.smoothing_density, self.twostep_smoothing_density)
        # Update initial latent density.
        init_smooth_density = self.smoothing_density.slice([0])
        opt_init_density = self.sm.update_init_density(init_smooth_density)
        self.filter_density.update([0], opt_init_density)
        # Update parameters of observation model
        self.om.update_hyperparameters(self.smoothing_density, self.X)
        
    def forward_path(self):
        """ Iterates forward, alternately doing prediction and filtering step.
        """
        for t in range(1, self.T+1):
            pre_filter_density = self.filter_density.slice([t-1])
            cur_prediction_density = self.sm.prediction(pre_filter_density)
            self.prediction_density.update([t], cur_prediction_density)
            cur_filter_density = self.om.filtering(cur_prediction_density, self.X[t-1:t])
            self.filter_density.update([t], cur_filter_density)
        
    def backward_path(self):
        """ Iterates backward doing smoothing step.
        """
        last_filter_density = self.filter_density.slice([self.T])
        self.smoothing_density.update([self.T], last_filter_density)
        
        for t in numpy.arange(self.T-1,-1,-1):
            cur_filter_density = self.filter_density.slice([t])
            post_smoothing_density = self.smoothing_density.slice([t+1])
            cur_smoothing_density, cur_two_step_smoothing_density = self.sm.smoothing(cur_filter_density, 
                                                                                   post_smoothing_density)
            self.smoothing_density.update([t], cur_smoothing_density)
            self.twostep_smoothing_density.update([t], cur_two_step_smoothing_density)
            
    def compute_log_likelihood(self) -> float:
        """ Computes the log-likelihood of the model, given by
        
        $$
        \ell = \sum_t \ln p(x_t|x_{1:t-1}).
        $$
        """
        llk = 0
        px = self.om.emission_density.affine_marginal_transformation(self.prediction_density)
        for t in range(1,self.T+1):
            cur_px = px.slice([t])
            llk += cur_px.evaluate_ln(self.X[t-1:t])[0,0]
        return llk
    
    def compute_data_density(self) -> densities.GaussianDensity:
        px = self.om.emission_density.affine_marginal_transformation(self.prediction_density)
        return px.slice(numpy.arange(1, self.T+1))