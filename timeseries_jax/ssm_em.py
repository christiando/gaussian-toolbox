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
from jax import jit, random

# import numpy as np
from timeseries_jax import observation_models, state_models
from src_jax import densities
import pickle
import os
import numpy as np
import time
from typing import Union


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


class StateSpaceEM:
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
        self.u_x, self.u_z = u_x, u_z
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
            if self.iteration % 1 == 0:
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

    def forward_step(self, t: int, pf_dict, X: jnp.array):
        pre_filter_density = densities.GaussianDensity(**pf_dict)
        if self.u_z is not None:
            uz_t = self.u_z[t - 1].reshape((1, -1))
        else:
            uz_t = None
        cur_prediction_density = self.sm.prediction(pre_filter_density, uz_t=uz_t)
        if self.u_x is not None:
            ux_t = self.u_x[t - 1].reshape((1, -1))
        else:
            ux_t = None
        cur_filter_density = self.om.filtering(
            cur_prediction_density, X[t - 1][None], ux_t=ux_t
        )

        cp_dict = {
            "Sigma": cur_prediction_density.Sigma,
            "mu": cur_prediction_density.mu,
            "Lambda": cur_prediction_density.Lambda,
            "ln_det_Sigma": cur_prediction_density.ln_det_Sigma,
        }
        cf_dict = {
            "Sigma": cur_filter_density.Sigma,
            "mu": cur_filter_density.mu,
            "Lambda": cur_filter_density.Lambda,
            "ln_det_Sigma": cur_filter_density.ln_det_Sigma,
        }
        return cf_dict, cp_dict

    def forward_path(self):
        """ Iterates forward, alternately doing prediction and filtering step.
        """
        cf_dict = {
            "Sigma": self.filter_density.Sigma[:1],
            "mu": self.filter_density.mu[:1],
            "Lambda": self.filter_density.Lambda[:1],
            "ln_det_Sigma": self.filter_density.ln_det_Sigma[:1],
        }
        forward_step = jit(lambda t, cf: self.forward_step(t, cf, self.X))
        Sigma_p, mu_p, Lambda_p, ln_det_Sigma_p = (
            np.array(self.prediction_density.Sigma),
            np.array(self.prediction_density.mu),
            np.array(self.prediction_density.Lambda),
            np.array(self.prediction_density.ln_det_Sigma),
        )
        Sigma_f, mu_f, Lambda_f, ln_det_Sigma_f = (
            np.array(self.filter_density.Sigma),
            np.array(self.filter_density.mu),
            np.array(self.filter_density.Lambda),
            np.array(self.filter_density.ln_det_Sigma),
        )
        for t in range(1, self.T + 1):
            cf_dict, cp_dict = forward_step(t, cf_dict)
            Sigma_p[t] = cp_dict["Sigma"]
            mu_p[t] = cp_dict["mu"]
            Lambda_p[t] = cp_dict["Lambda"]
            ln_det_Sigma_p[t] = cp_dict["ln_det_Sigma"]
            Sigma_f[t] = cf_dict["Sigma"]
            mu_f[t] = cf_dict["mu"]
            Lambda_f[t] = cf_dict["Lambda"]
            ln_det_Sigma_f[t] = cf_dict["ln_det_Sigma"]
        self.prediction_density = densities.GaussianDensity(
            jnp.asarray(Sigma_p),
            jnp.asarray(mu_p),
            jnp.asarray(Lambda_p),
            jnp.asarray(ln_det_Sigma_p),
        )
        self.filter_density = densities.GaussianDensity(
            jnp.asarray(Sigma_f),
            jnp.asarray(mu_f),
            jnp.asarray(Lambda_f),
            jnp.asarray(ln_det_Sigma_f),
        )

    def backward_step(self, t: int, ps_dict: dict, X: jnp.array):
        cur_filter_density = self.filter_density.slice(jnp.array([t]))
        post_smoothing_density = densities.GaussianDensity(**ps_dict)
        if self.u_z is not None:
            uz_t = self.u_z[t - 1].reshape((1, -1))
        else:
            uz_t = None
        cur_smoothing_density, cur_two_step_smoothing_density = self.sm.smoothing(
            cur_filter_density, post_smoothing_density, uz_t=uz_t
        )
        cs_dict = {
            "Sigma": cur_smoothing_density.Sigma,
            "mu": cur_smoothing_density.mu,
            "Lambda": cur_smoothing_density.Lambda,
            "ln_det_Sigma": cur_smoothing_density.ln_det_Sigma,
        }
        ctss_dict = {
            "Sigma": cur_two_step_smoothing_density.Sigma,
            "mu": cur_two_step_smoothing_density.mu,
            "Lambda": cur_two_step_smoothing_density.Lambda,
            "ln_det_Sigma": cur_two_step_smoothing_density.ln_det_Sigma,
        }
        return cs_dict, ctss_dict

    def backward_path(self):
        """ Iterates backward doing smoothing step.
        """
        last_filter_density = self.filter_density.slice(jnp.array([self.T]))
        cs_dict = {
            "Sigma": last_filter_density.Sigma,
            "mu": last_filter_density.mu,
            "Lambda": last_filter_density.Lambda,
            "ln_det_Sigma": last_filter_density.ln_det_Sigma,
        }
        self.smoothing_density.update(jnp.array([self.T]), last_filter_density)

        Sigma_s, mu_s, Lambda_s, ln_det_Sigma_s = (
            np.array(self.smoothing_density.Sigma),
            np.array(self.smoothing_density.mu),
            np.array(self.smoothing_density.Lambda),
            np.array(self.smoothing_density.ln_det_Sigma),
        )
        Sigma_tss, mu_tss, Lambda_tss, ln_det_Sigma_tss = (
            np.array(self.twostep_smoothing_density.Sigma),
            np.array(self.twostep_smoothing_density.mu),
            np.array(self.twostep_smoothing_density.Lambda),
            np.array(self.twostep_smoothing_density.ln_det_Sigma),
        )

        backward_step = jit(lambda t, cs: self.backward_step(t, cs, self.X))
        for t in jnp.arange(self.T - 1, -1, -1):
            cs_dict, ctss_dict = backward_step(t, cs_dict)
            Sigma_s[t] = cs_dict["Sigma"]
            mu_s[t] = cs_dict["mu"]
            Lambda_s[t] = cs_dict["Lambda"]
            ln_det_Sigma_s[t] = cs_dict["ln_det_Sigma"]
            Sigma_tss[t] = ctss_dict["Sigma"]
            mu_tss[t] = ctss_dict["mu"]
            Lambda_tss[t] = ctss_dict["Lambda"]
            ln_det_Sigma_tss[t] = ctss_dict["ln_det_Sigma"]
        self.smoothing_density = densities.GaussianDensity(
            Sigma=jnp.asarray(Sigma_s),
            mu=jnp.asarray(mu_s),
            Lambda=jnp.asarray(Lambda_s),
            ln_det_Sigma=jnp.asarray(ln_det_Sigma_s),
        )
        self.twostep_smoothing_density = densities.GaussianDensity(
            Sigma=jnp.asarray(Sigma_tss),
            mu=jnp.asarray(mu_tss),
            Lambda=jnp.asarray(Lambda_tss),
            ln_det_Sigma=jnp.asarray(ln_det_Sigma_tss), 
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
        return self.om.evaluate_llk(p_z, self.X, u=self.u_x)

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
        prediction_density = self._setup_density(T=T + 1)
        filter_density = self._setup_density(T=T + 1)
        filter_density.update(jnp.array([0]), p0)

        cf_dict = {
            "Sigma": filter_density.Sigma[:1],
            "mu": filter_density.mu[:1],
            "Lambda": filter_density.Lambda[:1],
            "ln_det_Sigma": filter_density.ln_det_Sigma[:1],
        }
        forward_step = jit(lambda t, cf: self.forward_step(t, cf, X))

        Sigma_p, mu_p, Lambda_p, ln_det_Sigma_p = (
            np.array(prediction_density.Sigma),
            np.array(prediction_density.mu),
            np.array(prediction_density.Lambda),
            np.array(prediction_density.ln_det_Sigma),
        )

        for t in range(1, T + 1):
            cf_dict, cp_dict = forward_step(t, cf_dict)
            Sigma_p[t] = cp_dict["Sigma"]
            mu_p[t] = cp_dict["mu"]
            Lambda_p[t] = cp_dict["Lambda"]
            ln_det_Sigma_p[t] = cp_dict["ln_det_Sigma"]

        prediction_density = densities.GaussianDensity(
            jnp.asarray(Sigma_p),
            jnp.asarray(mu_p),
            jnp.asarray(Lambda_p),
            jnp.asarray(ln_det_Sigma_p),
        )
        p_z = prediction_density.slice(jnp.arange(1 + ignore_init_samples, T + 1))
        if u_x is not None:
            u_x_tmp = u_x[ignore_init_samples:]
        else:
            u_x_tmp = u_x
        return self.om.evaluate_llk(p_z, X[ignore_init_samples:], u_x=u_x_tmp)

    def gappy_forward_step(self, t: int, pf_dict, X):
        pre_filter_density = densities.GaussianDensity(**pf_dict)
        if self.u_z is not None:
            uz_t = self.u_z[t - 1].reshape((1, -1))
        else:
            uz_t = None
        cur_prediction_density = self.sm.prediction(pre_filter_density, uz_t=uz_t)
        if self.u_x is not None:
            ux_t = self.u_x[t - 1].reshape((1, -1))
        else:
            ux_t = None
        cur_filter_density = self.om.gappy_filtering(
            cur_prediction_density, X[t - 1][None], ux_t=ux_t
        )

        cp_dict = {
            "Sigma": cur_prediction_density.Sigma,
            "mu": cur_prediction_density.mu,
            "Lambda": cur_prediction_density.Lambda,
            "ln_det_Sigma": cur_prediction_density.ln_det_Sigma,
        }
        cf_dict = {
            "Sigma": cur_filter_density.Sigma,
            "mu": cur_filter_density.mu,
            "Lambda": cur_filter_density.Lambda,
            "ln_det_Sigma": cur_filter_density.ln_det_Sigma,
        }
        return cf_dict, cp_dict

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

        Sigma_p, mu_p, Lambda_p, ln_det_Sigma_p = (
            np.array(filter_density.Sigma),
            np.array(filter_density.mu),
            np.array(filter_density.Lambda),
            np.array(filter_density.ln_det_Sigma),
        )

        cf_dict = {
            "Sigma": filter_density.Sigma[:1],
            "mu": filter_density.mu[:1],
            "Lambda": filter_density.Lambda[:1],
            "ln_det_Sigma": filter_density.ln_det_Sigma[:1],
        }

        for t in range(1, T + 1):
            cf_dict, cp_dict = self.gappy_forward_step(t, cf_dict, X)
            Sigma_p[t] = cp_dict["Sigma"]
            mu_p[t] = cp_dict["mu"]
            Lambda_p[t] = cp_dict["Lambda"]
            ln_det_Sigma_p[t] = cp_dict["ln_det_Sigma"]

        prediction_density = densities.GaussianDensity(
            jnp.asarray(Sigma_p),
            jnp.asarray(mu_p),
            jnp.asarray(Lambda_p),
            jnp.asarray(ln_det_Sigma_p),
        )
        px = self.om.emission_density.affine_marginal_transformation(
            prediction_density.slice(jnp.arange(1, T + 1))
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
        filter_density = self._setup_density(T=T + 1)
        filter_density.update(jnp.array([0]), p0)

        Sigma_f, mu_f, Lambda_f, ln_det_Sigma_f = (
            np.array(filter_density.Sigma),
            np.array(filter_density.mu),
            np.array(filter_density.Lambda),
            np.array(filter_density.ln_det_Sigma),
        )

        mu_unobserved = np.copy(X)
        std_unobserved = np.zeros(X.shape)
        cur_filter_density = filter_density.slice(jnp.array([0]))
        # Filtering
        for t in range(1, T + 1):
            # Filter
            if u_z is not None:
                uz_t = u_z[t - 1].reshape((-1, 1))
            else:
                uz_t = None
            cur_prediction_density = self.sm.prediction(cur_filter_density, uz_t=uz_t)
            # prediction_density.update([t], cur_prediction_density)
            if u_x is not None:
                ux_t = u_x[t - 1].reshape((-1, 1))
            else:
                ux_t = None
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
                np.array(smoothing_density.Sigma),
                np.array(smoothing_density.mu),
                np.array(smoothing_density.Lambda),
                np.array(smoothing_density.ln_det_Sigma),
            )

            for t in jnp.arange(T - 1, -1, -1):
                # Smoothing step
                cur_filter_density = filter_density.slice(jnp.array([t]))
                if u_z is not None:
                    uz_t = u_z[t - 1].reshape((1, -1))
                else:
                    uz_t = None
                cur_smoothing_density = self.sm.smoothing(
                    cur_filter_density, cur_smoothing_density, uz_t=uz_t
                )[0]
                Sigma_s[t] = cur_smoothing_density.Sigma
                mu_s[t] = cur_smoothing_density.mu
                Lambda_s[t] = cur_smoothing_density.Lambda
                ln_det_Sigma_s[t] = cur_smoothing_density.ln_det_Sigma
                # Get density of unobserved data
                if u_x is not None:
                    ux_t = u_x[t - 1].reshape((1, -1))
                else:
                    ux_t = None
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
        
    def gappy_forward_step_static(self, t: int, pf_dict: dict, X: jnp.ndarray, observed_dims: jnp.ndarray = None, nonobserved_dims: jnp.ndarray=None) -> Union[dict, jnp.ndarray, jnp.ndarray]:
        pre_filter_density = densities.GaussianDensity(**pf_dict)
        cur_prediction_density = self.sm.prediction(pre_filter_density, uz_t=None)
        cur_filter_density = self.om.gappy_filtering_static(
            cur_prediction_density, X[t - 1][None], observed_dims, ux_t=None
        )
        mu_x_t, std_x_t = self.om.gappy_data_density_static(cur_filter_density, X[t - 1][None], observed_dims, nonobserved_dims)

        cf_dict = {
            "Sigma": cur_filter_density.Sigma,
            "mu": cur_filter_density.mu,
            "Lambda": cur_filter_density.Lambda,
            "ln_det_Sigma": cur_filter_density.ln_det_Sigma,
        }
        return cf_dict, mu_x_t, std_x_t
        
    def predict_static(self,
        X: jnp.ndarray,
        p0: densities.GaussianDensity = None,
        observed_dims: jnp.ndarray = None,
        u_x: jnp.ndarray = None,
        u_z: jnp.ndarray = None) -> Union[densities.GaussianDensity, jnp.array, jnp.array]:
        
        T = X.shape[0]
        if p0 is None:
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
            
        cf_dict = {
            "Sigma": p0.Sigma,
            "mu": p0.mu,
            "Lambda": p0.Lambda,
            "ln_det_Sigma": p0.ln_det_Sigma,
        }
        filter_density = self._setup_density(T=T + 1)
        filter_density.update(jnp.array([0]), p0)

        Sigma_f, mu_f, Lambda_f, ln_det_Sigma_f = (
            np.array(filter_density.Sigma),
            np.array(filter_density.mu),
            np.array(filter_density.Lambda),
            np.array(filter_density.ln_det_Sigma),
        )
        
        if observed_dims == None:
            num_nonobserved_dims = X.shape[1]
            nonobserved_dims = None
        else:
            nonobserved_dims = jnp.setxor1d(jnp.arange(self.Dx), observed_dims)
            num_nonobserved_dims = len(nonobserved_dims)
        mu_x, std_x = np.empty([T, num_nonobserved_dims]), np.empty([T, num_nonobserved_dims])
        gappy_forward_step_static = jit(lambda t, cf: self.gappy_forward_step_static(t, cf, X, observed_dims, nonobserved_dims))
        
        for t in range(1, T + 1):
            cf_dict, mu_x_t, std_x_t = gappy_forward_step_static(t, cf_dict)
            Sigma_f[t] = cf_dict["Sigma"]
            mu_f[t] = cf_dict["mu"]
            Lambda_f[t] = cf_dict["Lambda"]
            ln_det_Sigma_f[t] = cf_dict["ln_det_Sigma"]
            mu_x[t-1] = mu_x_t
            std_x[t-1] = std_x_t
        filter_density = densities.GaussianDensity(Sigma=jnp.asarray(Sigma_f),
                mu=jnp.asarray(mu_f),
                Lambda=jnp.asarray(Lambda_f),
                ln_det_Sigma=jnp.asarray(ln_det_Sigma_f),
                )
        mu_x, std_x = jnp.asarray(mu_x), jnp.asarray(std_x)
        return filter_density, mu_x, std_x
        

    def sample_trajectory(
        self,
        X: jnp.ndarray,
        obs_indices: jnp.ndarray = None,
        num_samples: int = 1,
        p0: densities.GaussianDensity = None,
    ) -> jnp.ndarray:
        T = X.shape[0]
        if p0 is None:
            p0 = densities.GaussianDensity(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
        z_sample = np.empty((T + 1, num_samples, self.Dz))
        z_sample[0] = np.asarray(p0.sample(num_samples)[:, 0])
        X_sample = np.empty((T, num_samples, self.Dx))
        if obs_indices != None:
            all_indices = jnp.arange(self.Dx)
            unobs_indices = all_indices[
                jnp.logical_not(jnp.isin(all_indices, obs_indices))
            ]

        for t in range(1, T + 1):
            z_sample[t] = np.asarray(
                self.sm.state_density.condition_on_x(z_sample[t - 1]).sample(1)[0]
            )
            if obs_indices == None:
                px = self.om.emission_density.condition_on_x(z_sample[t])
                X_sample[t - 1] = np.asarray(px.sample(1)[0])
            else:
                px = (
                    self.om.emission_density.condition_on_x(z_sample[t])
                    .condition_on(obs_indices)
                    .condition_on_x(X[t - 1, obs_indices])
                )
                X_sample[t - 1, :, unobs_indices] = np.asarray(px.sample(1)[0])
                X_sample[t - 1, :, obs_indices] = np.asarray(X[t - 1, obs_indices])

        return jnp.asarray(z_sample), jnp.asarray(X_sample)

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
