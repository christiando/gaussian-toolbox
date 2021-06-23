##################################################################################################
# This file is part of the Gaussian Toolbox,                                                     #
#                                                                                                #
# It contains the class to fit state space models (SSMs) with the expectation-maximization       #
# algorithm.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

from autograd import numpy
import observation_models, state_models
import sys
sys.path.append('../src/')
import densities

class StateSpaceEM:
    
    def __init__(self, X: numpy.ndarray, observation_model: observation_models.ObservationModel, 
                 state_model: state_models.StateModel, max_iter: int=100, conv_crit: float=1e-4,
                 u_x: numpy.ndarray=None, u_z: numpy.ndarray=None):
        """ Class to fit a state space model with the expectation-maximization procedure.
        
        :param X: numpy.ndarray [T, Dx]
            Training data.
        :param observation_model: ObservationModel
            The observation model of the data.
        :param state_model: StateModel
            The state model for the latent variables.
        :param max_iter: int
            Maximal number of EM iteration performed. (Default=100)
        :param conv_crit: float
            Convergence criterion for the EM procedure.
        :param u_x: numpy.ndarray [T,...]
            Control variables for observation model. (Default=None)
        :param u_z: numpy.ndarray [T,...]
            Control variables for state model. (Default=None)   
        """
        self.X = X
        self.T, self.Dx = self.X.shape
        self.u_x, self.u_z = u_x, u_z
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
        self.prediction_density = self._setup_density(T=self.T + 1)
        self.filter_density = self._setup_density(T=self.T + 1)
        self.smoothing_density = self._setup_density(T=self.T + 1)
        self.twostep_smoothing_density = self._setup_density(D=int(2*self.Dz))
        #self.twostep_smoothing_density = self.twostep_smoothing_density.slice(range(self.T))
        
    def _setup_density(self, D: int=None, T: int=None) -> densities.GaussianDensity:
        """ Initializes a density object (with uniform densities).
        """
        if D is None:
            D = self.Dz
        if T is None:
            T = self.T
        Sigma = numpy.tile(numpy.eye(D)[None], (T,1,1))
        Lambda = numpy.tile(numpy.eye(D)[None], (T,1,1))
        mu = numpy.zeros((T, D))
        ln_det_Sigma = D * numpy.log(numpy.ones(T))
        return densities.GaussianDensity(Sigma, mu, Lambda, ln_det_Sigma)
        
    def run(self):
        """ Runs the expectation-maximization algorithm, until converged 
            or maximal number of iterations is reached.
        """
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
        self.sm.update_hyperparameters(self.smoothing_density, self.twostep_smoothing_density, u_z=self.u_z)
        # Update initial latent density.
        init_smooth_density = self.smoothing_density.slice([0])
        opt_init_density = self.sm.update_init_density(init_smooth_density)
        self.filter_density.update([0], opt_init_density)
        # Update parameters of observation model
        self.om.update_hyperparameters(self.smoothing_density, self.X, u_x=self.u_x)
        
    def forward_path(self):
        """ Iterates forward, alternately doing prediction and filtering step.
        """
        for t in range(1, self.T+1):
            pre_filter_density = self.filter_density.slice([t-1])
            if self.u_z is not None:
                uz_t = self.u_z[t-1:t]
            else:
                uz_t = None
            cur_prediction_density = self.sm.prediction(pre_filter_density, uz_t=uz_t)
            self.prediction_density.update([t], cur_prediction_density)
            if self.u_x is not None:
                ux_t = self.u_x[t-1:t]
            else:
                ux_t = None
            cur_filter_density = self.om.filtering(cur_prediction_density, self.X[t-1:t], ux_t=ux_t)
            self.filter_density.update([t], cur_filter_density)
        
    def backward_path(self):
        """ Iterates backward doing smoothing step.
        """
        last_filter_density = self.filter_density.slice([self.T])
        self.smoothing_density.update([self.T], last_filter_density)
        
        for t in numpy.arange(self.T-1,-1,-1):
            cur_filter_density = self.filter_density.slice([t])
            post_smoothing_density = self.smoothing_density.slice([t+1])
            if self.u_z is not None:
                uz_t = self.u_z[t-1:t]
            else:
                uz_t = None
            cur_smoothing_density, cur_two_step_smoothing_density = self.sm.smoothing(cur_filter_density,
                                                                                      post_smoothing_density,
                                                                                      uz_t=uz_t)
            self.smoothing_density.update([t], cur_smoothing_density)
            self.twostep_smoothing_density.update([t], cur_two_step_smoothing_density)
            
    def compute_log_likelihood(self) -> float:
        """ Computes the log-likelihood of the model, given by
    
        $$
        \ell = \sum_t \ln p(x_t|x_{1:t-1}).
        $$
        
        :return: float
            Data log likelihood.
        """
        p_z = self.prediction_density.slice(range(1,self.T+1))
        return self.om.evaluate_llk(p_z, self.X, u=self.u_x)
    
    def compute_predictive_log_likelihood(self, X: numpy.ndarray, p0: 'GaussianDensity'=None, 
                                          u_x: numpy.ndarray=None, u_z: numpy.ndarray=None):
        """ Computes the likelihood for given data X.
        
        :param X: numpy.ndarray [T, Dx]
            Data for which likelihood is computed.
        :param p0: GaussianDensity
            Density for the initial latent state. If None, the initial density 
            of the training data is taken. (Default=None)
        :param u_x: numpy.ndarray [T, ...]
            Control parameters for observation model. (Default=None)
        :param u_z: numpy.ndarray [T, ...]
            Control parameters for state model. (Default=None)
            
        :return: float
            Data log likelihood.
        """
        T = X.shape[0]
        if p0 is None:
            p0 = self.filter_density.slice([0])
        prediction_density = self._setup_density(T=T+1)
        filter_density = self._setup_density(T=T+1)
        filter_density.update([0], p0)
        for t in range(1, T+1):
            # Filter
            pre_filter_density = filter_density.slice([t-1])
            if u_z is not None:
                uz_t = u_z[t-1:t]
            else:
                uz_t = None
            cur_prediction_density = self.sm.prediction(pre_filter_density, uz_t=uz_t)
            prediction_density.update([t], cur_prediction_density)
            if u_x is not None:
                ux_t = u_x[t-1:t]
            else:
                ux_t = None
            cur_filter_density = self.om.filtering(cur_prediction_density, X[t-1:t], ux_t=ux_t)
            filter_density.update([t], cur_filter_density)
        p_z = prediction_density.slice(range(1,self.T+1))
        return self.om.evaluate_llk(p_z, X, self.u_x)
    
    def predict(self, X:numpy.ndarray, p0: 'GaussianDensity'=None, smoothed:bool=False, 
                u_x: numpy.ndarray=None, u_z: numpy.ndarray=None):
        """ Obtains predictions for data.
        
        :param X: numpy.ndarray [T, Dx]
            Data for which predictions are computed. Non observed values are NaN.
        :param p0: GaussianDensity
            Density for the initial latent state. If None, the initial density 
            of the training data is taken. (Default=None)
        :param smoothed: bool
            Uses the smoothed density for prediction. (Default=False)
        :param u_x: numpy.ndarray [T,...]
            Control variables for observation model. (Default=None)
        :param u_z: numpy.ndarray [T,...]
            Control variables for state model. (Default=None)
        
        :return: (GaussianDensity, numpy.ndarray [T, Dx], numpy.ndarray [T, Dx])
            Filter/smoothed density, mean, and standard deviation of predictions. Mean 
            is equal to the data and std equal to 0 for entries, where data is observed.
        """
        T = X.shape[0]
        if p0 is None:
            p0 = self.filter_density.slice([0])
        prediction_density = self._setup_density(T=T+1)
        filter_density = self._setup_density(T=T+1)
        filter_density.update([0], p0)
        mu_unobserved = numpy.copy(X)
        std_unobserved = numpy.zeros(X.shape)
        # Filtering
        for t in range(1, T+1):
            # Filter
            pre_filter_density = filter_density.slice([t-1])
            if u_z is not None:
                uz_t = u_z[t-1:t]
            else:
                uz_t = None
            cur_prediction_density = self.sm.prediction(pre_filter_density, uz_t=uz_t)
            prediction_density.update([t], cur_prediction_density)
            if u_x is not None:
                ux_t = u_x[t-1:t]
            else:
                ux_t = None
            cur_filter_density = self.om.gappy_filtering(cur_prediction_density, X[t-1:t], ux_t=ux_t)
            filter_density.update([t], cur_filter_density)
            # Get density of unobserved data
            if not smoothed:
                mu_ux, std_ux = self.om.gappy_data_density(cur_prediction_density, X[t-1:t], ux_t=ux_t)
                mu_unobserved[t-1, numpy.isnan(X[t-1])] = mu_ux
                std_unobserved[t-1, numpy.isnan(X[t-1])] = std_ux
        if not smoothed:
            return filter_density, mu_unobserved, std_unobserved
        # Smoothing
        else:
            # Initialize smoothing
            smoothing_density = self._setup_density(T=T+1)
            smoothing_density.update([T], filter_density.slice([T]))
            for t in numpy.arange(T-1,-1,-1):
                # Smoothing step
                cur_filter_density = filter_density.slice([t])
                post_smoothing_density = smoothing_density.slice([t+1])
                if u_z is not None:
                    uz_t = u_z[t-1:t]
                else:
                    uz_t = None
                cur_smoothing_density, cur_two_step_smoothing_density = self.sm.smoothing(cur_filter_density, 
                                                                                          post_smoothing_density,
                                                                                          uz_t=uz_t)
                smoothing_density.update([t], cur_smoothing_density)
                # Get density of unobserved data
                if u_x is not None:
                    ux_t = u_x[t-1:t]
                else:
                    ux_t = None
                mu_ux, std_ux = self.om.gappy_data_density(cur_smoothing_density, X[t:t+1], ux_t=ux_t)
                mu_unobserved[t, numpy.isnan(X[t])] = mu_ux
                std_unobserved[t, numpy.isnan(X[t])] = std_ux
            return smoothing_density, mu_unobserved, std_unobserved
        
                
            
    def compute_data_density(self) -> densities.GaussianDensity:
        """ Computes the data density for the training data, given the prediction density.
        
        :return: GaussianDensity
            Data density.
        """
        px = self.om.emission_density.affine_marginal_transformation(self.prediction_density)
        return px.slice(numpy.arange(1, self.T+1))