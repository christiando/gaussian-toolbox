##################################################################################################
# This file is part of the Gaussian Toolbox,                                                     #
#                                                                                                #
# It contains the class to fit state space models (SSMs) with the expectation-maximization       #
# algorithm.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
import sys

sys.path.append("../")
from jax import numpy as jnp
from jax import jit, lax
import objax

from timeseries_jax import observation_models, state_models
from src_jax import densities
import pickle
import os
import numpy as onp
import time
from typing import Union, Tuple


def load_model(model_name: str, path: str = "") -> "StateSpaceEM":
    """ Loads state space em model.
    
    :param model_name: str
        Name of the model, which is used as file name.
    :param path: str
        Path to which model is saved to. (Default='')
    :return: StateSpaceEM
        Loaded model.
    """
    return pickle.load(open("%s/%s.p" % (path, model_name), "rb"))


class StateSpaceEM(objax.Module):
    def __init__(
        self,
        X: jnp.ndarray,
        observation_model: observation_models.ObservationModel,
        state_model: state_models.StateModel,
        max_iter: int = 100,
        conv_crit: float = 1e-3,
        u_x: jnp.ndarray = None,
        u_z: jnp.ndarray = None,
        timeit: bool = False,
    ):
        """ Class to fit a state space model with the expectation-maximization procedure.
        
        :param X: jnp.ndarray [T, Dx]
            Training data.
        :param observation_model: ObservationModel
            The observation model of the data.
        :param state_model: StateModel
            The state model for the latent variables.
        :param max_iter: int
            Maximal number of EM iteration performed. (Default=100)
        :param conv_crit: float
            Convergence criterion for the EM procedure.
        :param u_x: jnp.ndarray [T,...]
            Control variables for observation model. (Default=None)
        :param u_z: jnp.ndarray [T,...]
            Control variables for state model. (Default=None)
        :param timeit: bool
            If true, prints the timings. (Default=False)
        """
        self.X = X
        self.T, self.Dx = self.X.shape
        if u_x is None:
            self.u_x = jnp.empty((self.T, 0))
        else:
            self.u_x = u_x

        if u_z is None:
            self.u_z = jnp.empty((self.T, 0))
        else:
            self.u_z = u_z

        self.Dz = state_model.Dz
        # observation model
        self.om = observation_model
        # state model
        self.sm = state_model
        self.max_iter = max_iter
        self.conv_crit = conv_crit
        self.timeit = timeit
        self.iteration = 0
        self.llk_list = []
        # Setup densities
        self.prediction_density = self._setup_density(T=self.T + 1)
        self.filter_density = self._setup_density(T=self.T + 1)
        self.smoothing_density = self._setup_density(T=self.T + 1)
        self.twostep_smoothing_density = self._setup_density(D=int(2 * self.Dz))
        # self.twostep_smoothing_density = self.twostep_smoothing_density.slice(range(self.T))

    def _setup_density(self, D: int = None, T: int = None) -> densities.GaussianDensity:
        """ Initializes a density object (with uniform densities).
        """
        if D is None:
            D = self.Dz
        if T is None:
            T = self.T
        Sigma = jnp.tile(jnp.eye(D)[None], (T, 1, 1))
        Lambda = jnp.tile(jnp.eye(D)[None], (T, 1, 1))
        mu = jnp.zeros((T, D))
        ln_det_Sigma = D * jnp.log(jnp.ones(T))
        return densities.GaussianDensity(Sigma, mu, Lambda, ln_det_Sigma)

    def run(self):
        """ Runs the expectation-maximization algorithm, until converged 
            or maximal number of iterations is reached.
        """
        converged = False
        while self.iteration < self.max_iter and not converged:
            time_start_total = time.perf_counter()
            self.estep()
            etime = time.perf_counter() - time_start_total
            time_start = time.perf_counter()
            self.llk_list.append(self.compute_log_likelihood())
            llk_time = time.perf_counter() - time_start
            time_start = time.perf_counter()
            self.mstep()
            mtime = time.perf_counter() - time_start
            if self.iteration > 3:
                conv = (self.llk_list[-1] - self.llk_list[-2]) / jnp.amax(
                    jnp.array(
                        [1, jnp.abs(self.llk_list[-1]), jnp.abs(self.llk_list[-2])]
                    )
                )
                converged = jnp.abs(conv) < self.conv_crit
            self.iteration += 1
            if self.iteration % 10 == 0:
                print("Iteration %d - llk=%.1f" % (self.iteration, self.llk_list[-1]))
            tot_time = time.perf_counter() - time_start_total
            if self.timeit:
                print(
                    "###################### \n"
                    + "E-step: Run Time %.1f \n" % etime
                    + "LLK: Run Time %.1f \n" % llk_time
                    + "M-step: Run Time %.1f \n" % mtime
                    + "Total: Run Time %.1f \n" % tot_time
                    + "###################### \n"
                )
        if not converged:
            print("EM reached the maximal number of iterations.")
        else:
            print("EM did converge.")

    def estep(self):
        """ Performs the expectation step, i.e. the forward-backward algorithm.
        """
        self.forward_path()
        self.backward_path()

    def mstep(self):
        """ Performs the maximization step, i.e. the updates of model parameters.
        """
        # Update parameters of state model
        self.sm.update_hyperparameters(
            self.smoothing_density,
            self.twostep_smoothing_density,
            u_z=self.u_z,
            iteration=self.iteration,
        )
        # Update initial latent density.
        init_smooth_density = self.smoothing_density.slice(jnp.array([0]))
        opt_init_density = self.sm.update_init_density(init_smooth_density)
        self.filter_density.update(jnp.array([0]), opt_init_density)
        # Update parameters of observation model
        self.om.update_hyperparameters(
            self.smoothing_density, self.X, u_x=self.u_x, iteration=self.iteration
        )

    def forward_step(self, carry, vars_t):
        X_t, uz_t, ux_t = vars_t
        Sigma, mu, Lambda, ln_det_Sigma = carry
        pre_filter_density = densities.GaussianDensity(
            Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma
        )
        cur_prediction_density = self.sm.prediction(pre_filter_density, u=uz_t)
        cur_filter_density = self.om.filtering(
            cur_prediction_density, X_t[None], ux_t=ux_t
        )
        carry = (
            cur_filter_density.Sigma,
            cur_filter_density.mu,
            cur_filter_density.Lambda,
            cur_filter_density.ln_det_Sigma,
        )
        result = (
            cur_prediction_density.Sigma[0],
            cur_prediction_density.mu[0],
            cur_prediction_density.Lambda[0],
            cur_prediction_density.ln_det_Sigma[0],
            cur_filter_density.Sigma[0],
            cur_filter_density.mu[0],
            cur_filter_density.Lambda[0],
            cur_filter_density.ln_det_Sigma[0],
        )
        return carry, result

    def forward_path(self):
        """ Iterates forward, alternately doing prediction and filtering step.
        """
        init = (
            self.filter_density.Sigma[:1],
            self.filter_density.mu[:1],
            self.filter_density.Lambda[:1],
            self.filter_density.ln_det_Sigma[:1],
        )
        forward_step = jit(lambda cf, vars_t: self.forward_step(cf, vars_t))
        _, result = lax.scan(
            forward_step, init, (self.X, self.u_z[:, None], self.u_x[:, None])
        )
        (
            Sigma_prediction,
            mu_prediction,
            Lambda_prediction,
            ln_det_Sigma_prediction,
            Sigma_filter,
            mu_filter,
            Lambda_filter,
            ln_det_Sigma_filter,
        ) = result
        new_prediction = densities.GaussianDensity(
            Sigma=Sigma_prediction,
            mu=mu_prediction,
            Lambda=Lambda_prediction,
            ln_det_Sigma=ln_det_Sigma_prediction,
        )
        t_range = jnp.arange(1, self.T + 1)
        self.prediction_density.update(t_range, new_prediction)
        new_filter = densities.GaussianDensity(
            Sigma=Sigma_filter,
            mu=mu_filter,
            Lambda=Lambda_filter,
            ln_det_Sigma=ln_det_Sigma_filter,
        )
        self.filter_density.update(t_range, new_filter)

    def backward_step(self, carry, vars_t: Tuple[int, jnp.array]):
        t, uz_t = vars_t
        cur_filter_density = self.filter_density.slice(jnp.array([t]))
        Sigma, mu, Lambda, ln_det_Sigma = carry
        post_smoothing_density = densities.GaussianDensity(
            Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma
        )
        cur_smoothing_density, cur_two_step_smoothing_density = self.sm.smoothing(
            cur_filter_density, post_smoothing_density, u=uz_t
        )
        carry = (
            cur_smoothing_density.Sigma,
            cur_smoothing_density.mu,
            cur_smoothing_density.Lambda,
            cur_smoothing_density.ln_det_Sigma,
        )
        result = (
            cur_smoothing_density.Sigma[0],
            cur_smoothing_density.mu[0],
            cur_smoothing_density.Lambda[0],
            cur_smoothing_density.ln_det_Sigma[0],
            cur_two_step_smoothing_density.Sigma[0],
            cur_two_step_smoothing_density.mu[0],
            cur_two_step_smoothing_density.Lambda[0],
            cur_two_step_smoothing_density.ln_det_Sigma[0],
        )
        return carry, result

    def backward_path(self):
        """ Iterates backward doing smoothing step.
        """
        last_filter_density = self.filter_density.slice(jnp.array([self.T]))
        cs_init = (
            last_filter_density.Sigma,
            last_filter_density.mu,
            last_filter_density.Lambda,
            last_filter_density.ln_det_Sigma,
        )
        self.smoothing_density.update(jnp.array([self.T]), last_filter_density)
        backward_step = jit(lambda cs, vars_t: self.backward_step(cs, vars_t))
        t_range = jnp.arange(self.T - 1, -1, -1)
        _, result = lax.scan(backward_step, cs_init, (t_range, self.u_z[:, None]))
        (
            Sigma_smooth,
            mu_smooth,
            Lambda_smooth,
            ln_det_Sigma_smooth,
            Sigma_two_step_smooth,
            mu_two_step_smooth,
            Lambda_two_step_smooth,
            ln_det_Sigma_two_step_smooth,
        ) = result
        new_smooth_density = densities.GaussianDensity(
            Sigma=Sigma_smooth,
            mu=mu_smooth,
            Lambda=Lambda_smooth,
            ln_det_Sigma=ln_det_Sigma_smooth,
        )
        self.smoothing_density.update(t_range, new_smooth_density)
        self.twostep_smoothing_density = densities.GaussianDensity(
            Sigma=Sigma_two_step_smooth,
            mu=mu_two_step_smooth,
            Lambda=Lambda_two_step_smooth,
            ln_det_Sigma=ln_det_Sigma_two_step_smooth,
        )

    def compute_log_likelihood(self) -> float:
        """ Computes the log-likelihood of the model, given by
    
        $$
        \ell = \sum_t \ln p(x_t|x_{1:t-1}).
        $$
        
        :return: float
            Data log likelihood.
        """
        p_z = self.prediction_density.slice(jnp.arange(1, self.T + 1))
        return self.om.evaluate_llk(p_z, self.X, u=self.u_x[:,None])

    def compute_predictive_log_likelihood(
        self,
        X: jnp.ndarray,
        p0: densities.GaussianDensity = None,
        u_x: jnp.ndarray = None,
        u_z: jnp.ndarray = None,
        ignore_init_samples: int = 0,
    ):
        """ Computes the likelihood for given data X.
        
        :param X: jnp.ndarray [T, Dx]
            Data for which likelihood is computed.
        :param p0: GaussianDensity
            Density for the initial latent state. If None, the initial density 
            of the training data is taken. (Default=None)
        :param u_x: jnp.ndarray [T, ...]
            Control parameters for observation model. (Default=None)
        :param u_z: jnp.ndarray [T, ...]
            Control parameters for state model. (Default=None)
        :param ignore_init_samples: int
            How many initial samples should be ignored. 
            
        :return: float
            Data log likelihood.
        """
        T = X.shape[0]
        if p0 is None:
            # p0 = self.filter_density.slice([0])
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
        init = (p0.Sigma[:1], p0.mu[:1], p0.Lambda[:1], p0.ln_det_Sigma[:1])
        t_range = jnp.arange(1, T + 1)
        forward_step = jit(lambda cf, t: self.forward_step(cf, t, self.X))
        _, result = lax.scan(forward_step, init, t_range)
        (
            Sigma_prediction,
            mu_prediction,
            Lambda_prediction,
            ln_det_Sigma_prediction,
            Sigma_filter,
            mu_filter,
            Lambda_filter,
            ln_det_Sigma_filter,
        ) = result
        prediction_density = densities.GaussianDensity(
            Sigma=Sigma_prediction,
            mu=mu_prediction,
            Lambda=Lambda_prediction,
            ln_det_Sigma=ln_det_Sigma_prediction,
        )
        p_z = prediction_density.slice(jnp.arange(ignore_init_samples, T))
        if u_x is None:
            u_x = jnp.empty((T, 0))
        u_x_tmp = u_x[ignore_init_samples:]
        return self.om.evaluate_llk(p_z, X[ignore_init_samples:], u_x=u_x_tmp)

    def gappy_forward_step(self, carry, vars_t):
        X_t, uz_t, ux_t = vars_t
        Sigma, mu, Lambda, ln_det_Sigma = carry
        pre_filter_density = densities.GaussianDensity(
            Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma
        )
        cur_prediction_density = self.sm.prediction(pre_filter_density, u=uz_t)
        cur_filter_density = self.om.gappy_filtering(
            cur_prediction_density, X_t[None], ux_t=ux_t
        )
        carry = (
            cur_filter_density.Sigma,
            cur_filter_density.mu,
            cur_filter_density.Lambda,
            cur_filter_density.ln_det_Sigma,
        )
        result = (
            cur_prediction_density.Sigma[0],
            cur_prediction_density.mu[0],
            cur_prediction_density.Lambda[0],
            cur_prediction_density.ln_det_Sigma[0],
            cur_filter_density.Sigma[0],
            cur_filter_density.mu[0],
            cur_filter_density.Lambda[0],
            cur_filter_density.ln_det_Sigma[0],
        )

        return carry, result

    def compute_predictive_density(
        self,
        X: jnp.ndarray,
        p0: densities.GaussianDensity = None,
        u_x: jnp.ndarray = None,
        u_z: jnp.ndarray = None,
    ):
        """ Computes the likelihood for given data X.
        
        :param X: jnp.ndarray [T, Dx]
            Data for which likelihood is computed.
        :param p0: GaussianDensity
            Density for the initial latent state. If None, the initial density 
            of the training data is taken. (Default=None)
        :param u_x: jnp.ndarray [T, ...]
            Control parameters for observation model. (Default=None)
        :param u_z: jnp.ndarray [T, ...]
            Control parameters for state model. (Default=None)
            
        :return: float
            Data log likelihood.
        """
        T = X.shape[0]
        if p0 is None:
            # p0 = self.filter_density.slice([0])
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
        filter_density = self._setup_density(T=T + 1)
        filter_density.update(jnp.array([0]), p0)

        init = (
            filter_density.Sigma[:1],
            filter_density.mu[:1],
            filter_density.Lambda[:1],
            filter_density.ln_det_Sigma[:1],
        )
        gappy_forward_step = jit(lambda cf, vars_t: self.gappy_forward_step(cf, vars_t))
        _, result = lax.scan(
            gappy_forward_step, init, (self.X, self.u_z[:, None], self.u_x[:, None])
        )

        (
            Sigma_prediction,
            mu_prediction,
            Lambda_prediction,
            ln_det_Sigma_prediction,
            Sigma_filter,
            mu_filter,
            Lambda_filter,
            ln_det_Sigma_filter,
        ) = result

        new_prediction_density = densities.GaussianDensity(
            Sigma=Sigma_prediction,
            mu=mu_prediction,
            Lambda=Lambda_prediction,
            ln_det_Sigma=ln_det_Sigma_prediction,
        )
        px = self.om.emission_density.affine_marginal_transformation(
            new_prediction_density
        )
        return px

    def predict(
        self,
        X: jnp.ndarray,
        p0: densities.GaussianDensity = None,
        smoothed: bool = False,
        u_x: jnp.ndarray = None,
        u_z: jnp.ndarray = None,
    ):
        """ Obtains predictions for data.
        
        :param X: jnp.ndarray [T, Dx]
            Data for which predictions are computed. Non observed values are NaN.
        :param p0: GaussianDensity
            Density for the initial latent state. If None, the initial density 
            of the training data is taken. (Default=None)
        :param smoothed: bool
            Uses the smoothed density for prediction. (Default=False)
        :param u_x: jnp.ndarray [T,...]
            Control variables for observation model. (Default=None)
        :param u_z: jnp.ndarray [T,...]
            Control variables for state model. (Default=None)
        
        :return: (GaussianDensity, jnp.ndarray [T, Dx], jnp.ndarray [T, Dx])
            Filter/smoothed density, mean, and standard deviation of predictions. Mean 
            is equal to the data and std equal to 0 for entries, where data is observed.
        """
        T = X.shape[0]
        if p0 is None:
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )

        if u_z == None:
            u_z = jnp.empty((T, 0))
        if u_x == None:
            u_x = jnp.empty((T, 0))
        filter_density = self._setup_density(T=T + 1)
        filter_density.update(jnp.array([0]), p0)

        Sigma_f, mu_f, Lambda_f, ln_det_Sigma_f = (
            onp.array(filter_density.Sigma),
            onp.array(filter_density.mu),
            onp.array(filter_density.Lambda),
            onp.array(filter_density.ln_det_Sigma),
        )

        mu_unobserved = onp.copy(X)
        std_unobserved = onp.zeros(X.shape)
        cur_filter_density = filter_density.slice(jnp.array([0]))
        # Filtering
        for t in range(1, T + 1):
            # Filter
            uz_t = u_z[t - 1]
            cur_prediction_density = self.sm.prediction(cur_filter_density, u=uz_t)
            # prediction_density.update([t], cur_prediction_density)
            ux_t = u_x[t - 1]
            cur_filter_density = self.om.gappy_filtering(
                cur_prediction_density, X[t - 1].reshape((1, -1)), ux_t=ux_t
            )
            Sigma_f[t] = cur_filter_density.Sigma
            mu_f[t] = cur_filter_density.mu
            Lambda_f[t] = cur_filter_density.Lambda
            ln_det_Sigma_f[t] = cur_filter_density.ln_det_Sigma
            # Get density of unobserved data
            if not smoothed:
                mu_ux, std_ux = self.om.gappy_data_density(
                    cur_prediction_density, X[t - 1].reshape((1, -1)), ux_t=ux_t
                )
                mu_unobserved[t - 1, jnp.isnan(X[t - 1])] = mu_ux
                std_unobserved[t - 1, jnp.isnan(X[t - 1])] = std_ux
            filter_density = densities.GaussianDensity(
                jnp.asarray(Sigma_f),
                jnp.asarray(mu_f),
                jnp.asarray(Lambda_f),
                jnp.asarray(ln_det_Sigma_f),
            )
        if not smoothed:
            return filter_density, mu_unobserved, std_unobserved
        # Smoothing
        else:
            # Initialize smoothing
            smoothing_density = self._setup_density(T=T + 1)
            cur_smoothing_density = filter_density.slice(jnp.array([T]))
            smoothing_density.update(jnp.array([T]), cur_smoothing_density)
            Sigma_s, mu_s, Lambda_s, ln_det_Sigma_s = (
                onp.array(smoothing_density.Sigma),
                onp.array(smoothing_density.mu),
                onp.array(smoothing_density.Lambda),
                onp.array(smoothing_density.ln_det_Sigma),
            )

            for t in jnp.arange(T - 1, -1, -1):
                # Smoothing step
                cur_filter_density = filter_density.slice(jnp.array([t]))
                uz_t = u_z[t - 1]

                cur_smoothing_density = self.sm.smoothing(
                    cur_filter_density, cur_smoothing_density, u=uz_t
                )[0]
                Sigma_s[t] = cur_smoothing_density.Sigma
                mu_s[t] = cur_smoothing_density.mu
                Lambda_s[t] = cur_smoothing_density.Lambda
                ln_det_Sigma_s[t] = cur_smoothing_density.ln_det_Sigma
                # Get density of unobserved data
                ux_t = u_x[t - 1]
                mu_ux, std_ux = self.om.gappy_data_density(
                    cur_smoothing_density, X[t].reshape((1, -1)), ux_t=ux_t
                )
                mu_unobserved[t, jnp.isnan(X[t])] = mu_ux
                std_unobserved[t, jnp.isnan(X[t])] = std_ux
            smoothing_density = densities.GaussianDensity(
                Sigma=jnp.asarray(Sigma_s),
                mu=jnp.asarray(mu_s),
                Lambda=jnp.asarray(Lambda_s),
                ln_det_Sigma=jnp.asarray(ln_det_Sigma_s),
            )
            return smoothing_density, mu_unobserved, std_unobserved

    def gappy_forward_step_static(
        self,
        carry,
        vars_t,
        observed_dims: jnp.ndarray = None,
        nonobserved_dims: jnp.ndarray = None,
    ) -> Union[dict, jnp.ndarray, jnp.ndarray]:
        X_t, uz_t, ux_t = vars_t
        Sigma, mu, Lambda, ln_det_Sigma = carry
        pre_filter_density = densities.GaussianDensity(
            Sigma=Sigma, mu=mu, Lambda=Lambda, ln_det_Sigma=ln_det_Sigma
        )
        cur_prediction_density = self.sm.prediction(pre_filter_density, u=uz_t)
        cur_filter_density = self.om.gappy_filtering_static(
            cur_prediction_density, X_t[None], observed_dims, ux_t=ux_t
        )
        mu_x_t, std_x_t = self.om.gappy_data_density_static(
            cur_filter_density, X_t[None], observed_dims, nonobserved_dims
        )

        carry = (
            cur_filter_density.Sigma,
            cur_filter_density.mu,
            cur_filter_density.Lambda,
            cur_filter_density.ln_det_Sigma,
        )
        result = (
            cur_filter_density.Sigma[0],
            cur_filter_density.mu[0],
            cur_filter_density.Lambda[0],
            cur_filter_density.ln_det_Sigma[0],
            mu_x_t,
            std_x_t,
        )
        return carry, result

    def predict_static(
        self,
        X: jnp.ndarray,
        p0: densities.GaussianDensity = None,
        observed_dims: jnp.ndarray = None,
        u_z: jnp.ndarray = None,
        u_x: jnp.ndarray = None,
    ) -> Union[densities.GaussianDensity, jnp.array, jnp.array]:
        """Predicts data with fixed condition dimensions. Faster, but more rigid than predict().
        
        TODO: Implement also smoothing.

        :param X: Data array containing the variabels to condition on, and indicating how long we wish to sample.
        :type X: jnp.ndarray [T, Dx]
        :param observed_dims: Dimension that are observed. If none no dimension is observed, defaults to None
        :type observed_dims: jnp.ndarray, optional [num_observed_dimensions]
        :param p0: Initial state density. If none, standard normal., defaults to None
        :type p0: densities.GaussianDensity, optional
        :return: Filter density of latent vairbale, mean and standard deviation of unobserved data.
        :rtype: Union[densities.GaussianDensity, jnp.array, jnp.array]
        """

        T = X.shape[0]
        if u_z == None:
            u_z = jnp.empty((T, 0))
        if u_x == None:
            u_x = jnp.empty((T, 0))

        if p0 is None:
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
        filter_density = self._setup_density(T=T + 1)
        filter_density.update(jnp.array([0]), p0)

        if observed_dims == None:
            nonobserved_dims = None
        else:
            nonobserved_dims = jnp.setxor1d(jnp.arange(self.Dx), observed_dims)

        gappy_forward_step_static = jit(
            lambda cf, vars_t: self.gappy_forward_step_static(
                cf, vars_t, observed_dims, nonobserved_dims
            )
        )
        init = (p0.Sigma, p0.mu, p0.Lambda, p0.ln_det_Sigma)
        _, result = lax.scan(
            gappy_forward_step_static, init, (X, u_z[:, None], u_x[:, None])
        )
        Sigma_f, mu_f, Lambda_f, ln_det_Sigma_f, mu_x, std_x = result
        t_range = jnp.arange(1, T + 1)
        filter_density_new = densities.GaussianDensity(
            Sigma=Sigma_f, mu=mu_f, Lambda=Lambda_f, ln_det_Sigma=ln_det_Sigma_f
        )
        filter_density.update(t_range, filter_density_new)
        return filter_density, mu_x, std_x

    def sample_step_static(
        self,
        z_old: jnp.array,
        vars_t: Tuple,
        observed_dims: jnp.ndarray,
        unobserved_dims: jnp.ndarray,
    ) -> Union[jnp.ndarray, jnp.ndarray]:
        """One time step sample for fixed observed data dimensions.

        :param z_old: Sample of latent variable in previous time step.
        :type z_old: jnp.ndarray [num_samples, Dz]
        :param rand_nums_z: Random numbers for sampling latent dimensions.
        :type rand_nums_z: jnp.ndarray [num_samples, Dz]
        :param x: Data vector for current time step.
        :type x: jnp.ndarray [1, Dx]
        :param rand_nums_x: Random numbers for sampling x.
        :type rand_nums_x: jnp.ndarray [num_samples, num_unobserved_dims]
        :param observed_dims: Observed dimensions.
        :type observed_dims: jnp.ndarray [num_observed_dims]
        :param unobserved_dims: Unobserved dimensions.
        :type unobserved_dims: jnp.ndarray [num_unobserved_dims]
        :return: Latent variable and data sample (only unobserved) for current time step.
        :rtype: Union[jnp.ndarray, jnp.ndarray] [num_samples, Dz] [num_samples, num_unobserved]
        """

        rand_nums_z_t, x_t, rand_nums_x_t, uz_t, ux_t = vars_t
        p_z = self.sm.condition_on_past(z_old, u=uz_t)
        L = jnp.linalg.cholesky(p_z.Sigma)
        z_sample = p_z.mu + jnp.einsum("abc,ac->ab", L, rand_nums_z_t)
        p_x = self.om.condition_on_z_and_observations(
            z_sample, x_t, observed_dims, unobserved_dims, ux_t=ux_t
        )
        L = jnp.linalg.cholesky(p_x.Sigma)
        x_sample = p_x.mu + jnp.einsum("abc,ac->ab", L, rand_nums_x_t)
        result = z_sample, x_sample
        return z_sample, result

    def sample_trajectory_static(
        self,
        X: jnp.ndarray,
        observed_dims: jnp.ndarray = None,
        p0: densities.GaussianDensity = None,
        num_samples: int = 1,
        u_z: jnp.ndarray = None,
        u_x: jnp.ndarray = None,
    ) -> Union[jnp.ndarray, jnp.ndarray]:
        """Samples a trajectories, with fixed observed data dimensions.

        :param X: Data array containing the variabels to condition on, and indicating how long we wish to sample.
        :type X: jnp.ndarray [T, Dx]
        :param observed_dims: Dimension that are observed. If none no dimension is observed, defaults to None
        :type observed_dims: jnp.ndarray, optional [num_observed_dimensions]
        :param p0: Initial state density. If none, standard normal., defaults to None
        :type p0: densities.GaussianDensity, optional
        :param num_samples: How many trajectories should be sampled, defaults to 1
        :type num_samples: int, optional
        :return: Samples of the latent variables, and the unobserved data dimensions.
        :rtype: Union[jnp.ndarray, jnp.ndarray] [T+1, nums_samples, Dz] [T, nums_samples, num_unobserved_dims]
        """
        T = X.shape[0]
        if u_z == None:
            u_z = jnp.empty((T, 0))
        if u_x == None:
            u_x = jnp.empty((T, 0))
        if p0 is None:
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
        if observed_dims == None:
            unobserved_dims = jnp.arange(self.Dx)
            num_unobserved_dims = X.shape[1]
        else:
            unobserved_dims = jnp.setxor1d(jnp.arange(self.Dx), observed_dims)
            num_unobserved_dims = len(unobserved_dims)

        init = jnp.asarray(p0.sample(num_samples)[:, 0])
        sample_step = jit(
            lambda z_old, vars_t: self.sample_step_static(
                z_old, vars_t, observed_dims, unobserved_dims
            )
        )
        rand_nums_z = objax.random.normal((T, num_samples, self.Dz))
        rand_nums_x = objax.random.normal((T, num_samples, num_unobserved_dims))

        _, result = lax.scan(
            sample_step, init, (rand_nums_z, X, rand_nums_x, u_z[:, None], u_x[:, None])
        )
        z_sample, X_sample = result

        return z_sample, X_sample

    def compute_data_density(self) -> densities.GaussianDensity:
        """ Computes the data density for the training data, given the prediction density.
        
        :return: GaussianDensity
            Data density.
        """
        px = self.om.emission_density.affine_marginal_transformation(
            self.prediction_density.slice(jnp.arange(1, self.T + 1))
        )
        return px

    def save(self, model_name: str, path: str = "", overwrite: bool = False):
        """ Saves the model.
        
        :param model_name: str
            Name of the model, which is used as file name.
        :param path: str
            Path to which model is saved to. (Default='')
        """
        if os.path.isfile(path) and not overwrite:
            raise RuntimeException(
                "File already exists. Pick another name or indicate overwrite."
            )
        else:
            pickle.dump(self, open("%s/%s.p" % (path, model_name), "wb"))
