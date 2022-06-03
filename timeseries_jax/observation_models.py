##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the class to fit observation models that can be incroporated in the SSM-framwork.  #
#                                                                                                #
# Implemented so far:                                                                            #
#       + LinearObservationModel (Gaussian Emission)                                             #
#       + HCCovObservationModel (Gaussian Emission with state dependent covariance)              #
#       + LSEMObservationModel (Gaussian Emission with non linear mean)                          #
# Yet to be implemented:                                                                         #
#       - HCCovLSEMObservationModel (Gaussian Emission with non linear mean and state dependent  #
#                                    covariance)                                                 #
#       - BernoulliObservationModel (Emissions for binary data)                                  #
#       - PoissonObservationModel (Emissions for count data)                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
import sys

sys.path.append("../")
import scipy
from scipy.optimize import minimize_scalar
from jax import numpy as jnp
from jax import scipy as jsc
import numpy as np
from jax import lax
from jax import jit, value_and_grad, vmap
import objax
from functools import partial

from src_jax import densities, conditionals, factors

# from pathos.multiprocessing import ProcessingPool as Pool


def recommend_dims(X, smooth_window=20, cut_off=0.99):
    X_mean = jnp.mean(X, axis=0)
    T = X.shape[0]
    X_smoothed = jnp.empty(X.shape)
    for i in range(X.shape[1]):
        X_smoothed.at[:, i].set(
            jnp.convolve(X[:, i], jnp.ones(smooth_window) / smooth_window, mode="same")
        )
    eig_vals_X, eig_vecs_X = scipy.linalg.eigh(
        jnp.dot((X_smoothed - X_mean[None]).T, X_smoothed - X_mean[None])
    )
    Dz = (
        jnp.searchsorted(jnp.cumsum(eig_vals_X[::-1]) / jnp.sum(eig_vals_X), cut_off)
        + 1
    )
    C = eig_vecs_X[:, -Dz:] * eig_vals_X[-Dz:] / T
    z_hat = jnp.dot(jnp.linalg.pinv(C), (X_smoothed - X_mean).T).T
    delta_X = X - jnp.dot(z_hat, C.T) - X_mean
    cov = jnp.dot(delta_X.T, delta_X)
    eig_vals_deltaX, eig_vecs_deltaX = scipy.linalg.eigh(cov)
    Du = jnp.searchsorted(
        jnp.cumsum(eig_vals_deltaX[::-1]) / jnp.sum(eig_vals_deltaX), cut_off
    )
    return Dz, Du


def augment_taken(X: jnp.ndim, delta: int = 1, num_delays: int = 1) -> jnp.ndarray:
    T, Dx = X.shape
    T_new, Dx_new = T - delta * (num_delays - 1), Dx * num_delays
    X_new = np.empty((T_new, Dx_new))
    for idelay in range(num_delays):
        X_new[:, idelay * Dx : (idelay + 1) * Dx] = X[
            idelay * delta : T_new + delta * idelay
        ]
    return jnp.asarray(X_new)


def logcosh(x):
    # s always has real part >= 0
    s = jnp.sign(x) * x
    p = jnp.exp(-2 * s)
    return s + jnp.log1p(p) - jnp.log(2)


class ObservationModel(objax.Module):
    def __init__(self):
        """ This is the template class for observation models in state space models. 
        Basically these classes should contain all functionality for the mapping between 
        the latent variables z, and observations x, i.e. p(x_t|z_t). The object should 
        have an attribute `emission_density`, which is be a `ConditionalDensity`. 
        Furthermore, it should be possible to optimize hyperparameters, when provided 
        with a density over the latent space.
        """
        self.emission_density = None

    def filtering(
        self, prediction_density: densities.GaussianDensity, x_t: jnp.ndarray, **kwargs
    ) -> densities.GaussianDensity:
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
        
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        raise NotImplementedError("Filtering for observation model not implemented.")

    def update_hyperparameters(
        self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray, **kwargs
    ):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """
        raise NotImplementedError(
            "Hyperparameter updates for observation model not implemented."
        )

    def evalutate_llk(
        self, p_z: densities.GaussianDensity, X: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """ Computes the log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: jnp.ndarray [T, Dx]
            Observations.
        """
        raise NotImplementedError(
            "Log likelihood not implemented for observation model."
        )


class LinearObservationModel(ObservationModel):
    def __init__(self, Dx: int, Dz: int, noise_x: float = 1.0):
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
        self.emission_density = conditionals.ConditionalGaussianDensity(
            jnp.array([self.C]), jnp.array([self.d]), jnp.array([self.Qx])
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.emission_density.Lambda[0],
            self.emission_density.ln_det_Sigma[0],
        )

    def pca_init(self, X: jnp.ndarray, smooth_window: int = 10):
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
            X_smoothed[:, i] = jnp.convolve(
                X[:, i], jnp.ones(smooth_window) / smooth_window, mode="same"
            )
        eig_vals, eig_vecs = scipy.linalg.eigh(
            jnp.dot((X_smoothed - self.d[None]).T, X_smoothed - self.d[None]),
            eigvals=(self.Dx - jnp.amin([self.Dz, self.Dx]), self.Dx - 1),
        )
        self.C[:, : jnp.amin([self.Dz, self.Dx])] = eig_vecs * eig_vals / T
        z_hat = jnp.dot(jnp.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - jnp.dot(z_hat, self.C.T) - self.d
        self.Qx = jnp.dot(delta_X.T, delta_X)
        self.emission_density = conditionals.ConditionalGaussianDensity(
            jnp.array([self.C]), jnp.array([self.d]), jnp.array([self.Qx])
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.emission_density.Lambda[0],
            self.emission_density.ln_det_Sigma[0],
        )

    def filtering(
        self, prediction_density: densities.GaussianDensity, x_t: jnp.ndarray, **kwargs
    ) -> densities.GaussianDensity:
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
        p_z_given_x = self.emission_density.affine_conditional_transformation(
            prediction_density
        )
        # Condition on x_t
        cur_filter_density = p_z_given_x.condition_on_x(x_t)
        return cur_filter_density

    def gappy_filtering(
        self, prediction_density: densities.GaussianDensity, x_t: jnp.ndarray, **kwargs
    ) -> densities.GaussianDensity:
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
            marginal_dims = jnp.concatenate(
                [jnp.arange(self.Dz), self.Dz + observed_dims]
            )
            p_zx_observed = p_zx.get_marginal(marginal_dims)
            # p(z_t | x_t (observed), x_{1:t-1})
            conditional_dims = jnp.arange(self.Dz, self.Dz + len(observed_dims))
            p_z_given_x_observed = p_zx_observed.condition_on(conditional_dims)
            cur_filter_density = p_z_given_x_observed.condition_on_x(
                x_t[:, observed_dims]
            )
            return cur_filter_density

    def gappy_filtering_static(
        self,
        prediction_density: densities.GaussianDensity,
        x_t: jnp.ndarray,
        observed_dims: jnp.ndarray = None,
        **kwargs
    ) -> densities.GaussianDensity:
        # In case all data are unobserved
        if observed_dims == None:
            return prediction_density
        # In case all data are observed
        elif len(observed_dims) == self.Dx:
            cur_filter_density = self.filtering(prediction_density, x_t)
            return cur_filter_density
        # In case we have only partial observations
        else:
            # p(z_t, x_t| x_{1:t-1})
            p_zx = self.emission_density.affine_joint_transformation(prediction_density)
            # p(z_t, x_t (observed) | x_{1:t-1})
            marginal_dims = jnp.concatenate(
                [jnp.arange(self.Dz), self.Dz + observed_dims]
            )
            p_zx_observed = p_zx.get_marginal(marginal_dims)
            # p(z_t | x_t (observed), x_{1:t-1})
            conditional_dims = jnp.arange(self.Dz, self.Dz + len(observed_dims))
            nonconditional_dims = jnp.arange(0, self.Dz)
            p_z_given_x_observed = p_zx_observed.condition_on_explicit(
                conditional_dims, nonconditional_dims
            )
            cur_filter_density = p_z_given_x_observed.condition_on_x(
                x_t[:, observed_dims]
            )

            return cur_filter_density

    def gappy_data_density(
        self, p_z: densities.GaussianDensity, x_t: jnp.ndarray, **kwargs
    ):
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
            p_ux = p_ux_given_ox.condition_on_x(x_t[:, observed_dims])
            return p_ux.mu[0], jnp.sqrt(p_ux.Sigma.diagonal(axis1=-1, axis2=-2))

    def gappy_data_density_static(
        self,
        p_z: densities.GaussianDensity,
        x_t: jnp.ndarray,
        observed_dims: jnp.ndarray = None,
        nonobserved_dims: jnp.ndarray = None,
        **kwargs
    ):
        """ Here the data density is calculated for incomplete data. Not observed values should be nans.
        
         p(x_t) = p(x_t|z_t)p(z_t) dz_t
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: jnp.ndarray [1, Dx]
            Observation, where unobserved dimensions are filled with NANs.
        :return: (jnp.ndarray, jnp.ndarray)
            Mean and variance of unobserved entries.
        """
        if observed_dims == None:
            p_x = self.emission_density.affine_marginal_transformation(p_z)
            return p_x.mu[0], jnp.sqrt(p_x.Sigma[0].diagonal(axis1=-1, axis2=-2))
        # In case all data are observed
        elif len(observed_dims) == self.Dx:
            return jnp.array([]), jnp.array([])
        else:
            # In case we have only partial observations
            # Density over unobserved variables
            p_x = self.emission_density.affine_marginal_transformation(p_z)
            p_ux_given_ox = p_x.condition_on_explicit(observed_dims, nonobserved_dims)
            p_ux = p_ux_given_ox.condition_on_x(x_t[:, observed_dims])
            return p_ux.mu[0], jnp.sqrt(p_ux.Sigma.diagonal(axis1=-1, axis2=-2))[0]

    def compute_Q_function(
        self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray, **kwargs
    ) -> float:
        return jnp.sum(
            self.emission_density.integrate_log_conditional_y(smoothing_density)(X)
        )

    def update_hyperparameters(
        self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray, **kwargs
    ):
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

    def update_Qx(self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray):
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

        Exx = jnp.sum(
            smoothing_density.integrate(
                "Ax_aBx_b_outer", A_mat=A, a_vec=a_t, B_mat=A, b_vec=a_t
            )[1:],
            axis=0,
        )
        # for t in range(1, T+1):
        #     cur_smooth_density = smoothing_density.slice(jnp.array([t]))
        #     Exx += cur_smooth_density.integrate('Ax_aBx_b_outer', A_mat=A,
        #                                         a_vec=a_t[t-1], B_mat=A,
        #                                         b_vec=a_t[t-1])[0]
        self.Qx = 0.5 * (Exx + Exx.T) / T

    def update_C(self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray):
        """ This procedure updates the transition matrix of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """
        Ezz = jnp.sum(smoothing_density.integrate("xx")[1:], axis=0)
        Ez = smoothing_density.integrate("x")[1:]
        zx = jnp.sum(Ez[:, :, None] * (X[:, None] - self.d[None, None]), axis=0)
        self.C = jnp.linalg.solve(Ezz, zx).T

    def update_d(self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray):
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
        self.emission_density = conditionals.ConditionalGaussianDensity(
            jnp.array([self.C]), jnp.array([self.d]), jnp.array([self.Qx])
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.emission_density.Lambda[0],
            self.emission_density.ln_det_Sigma[0],
        )

    def evaluate_llk(
        self, p_z: densities.GaussianDensity, X: jnp.ndarray, **kwargs
    ) -> float:
        """ Computes the log likelihood of data given distribution over latent variables.
        
        :param p_z: GaussianDensity
            Density over latent variables.
        :param X: jnp.ndarray [T, Dx]
            Observations.
            
        :return: float
            Log likelihood.
        """
        p_x = self.emission_density.affine_marginal_transformation(p_z)
        llk = jnp.sum(p_x.evaluate_ln(X, element_wise=True))
        return llk

    def condition_on_z_and_observations(
        self,
        z_sample: jnp.ndarray,
        x_t: jnp.ndarray,
        observed_dims: jnp.ndarray,
        unobserved_dims: jnp.ndarray,
        **kwargs
    ) -> densities.GaussianDensity:
        """ Returns the density p(x_unobserved|X_observed=x, Z=z).

        :param z_sample: Values of latent variable
        :type z_sample: jnp.ndarray
        :param x_t: Data.
        :type x_t: jnp.ndarray
        :param observed_dims: Observed dimension
        :type observed_dims: jnp.ndarray
        :param unobserved_dims: Unobserved dimensions.
        :type unobserved_dims: jnp.ndarray
        :return: The density over unobserved dimensions.
        :rtype: densities.GaussianDensity
        """
        p_x = self.emission_density.condition_on_x(z_sample)
        if observed_dims != None:
            p_x = p_x.condition_on_explicit(observed_dims, unobserved_dims)
            p_x = p_x.condition_on_x(x_t[observed_dims][None])
        return p_x


class LSEMObservationModel(LinearObservationModel, objax.Module):
    def __init__(
        self, Dx: int, Dz: int, Dk: int, noise_x: float = 1.0, lr: float = 1e-3,
    ):
        """
        This implements a linear+squared exponential mean (LSEM) observation model
        
            x_t = C phi(z_{t}) + d + xi_t     with      xi_t ~ N(0,Qx).
            
            The feature function is 
            
            phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).
            
            The kernel and linear activation function are given by
            
            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.
            

        :param Dx: Dimensions of observations.
        :type Dx: int
        :param Dz: Dimensions of latent space.
        :type Dz: int
        :param Dk: Number of kernels.
        :type Dk: int
        :param noise_x: Initial observation noise, defaults to 1.0
        :type noise_x: float, optional
        :param lr: Learnig rate for learning W, defaults to 1e-3
        :type lr: float, optional
        """
        self.Dx, self.Dz, self.Dk = Dx, Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qx = noise_x ** 2 * jnp.eye(self.Dx)
        self.C = jnp.array(np.random.randn(self.Dx, self.Dphi))
        if self.Dx == self.Dz:
            self.C = self.C.at[:, : self.Dz].set(jnp.eye(self.Dx))
        else:
            self.C = self.C.at[:, : self.Dz].set(
                objax.random.normal((self.Dx, self.Dz))
            )
        self.d = jnp.zeros((self.Dx,))
        self.W = objax.TrainVar(jnp.array(np.random.randn(self.Dk, self.Dz + 1)))
        self.emission_density = conditionals.LSEMGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            W=self.W,
            Sigma=jnp.array([self.Qx]),
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.emission_density.Lambda[0],
            self.emission_density.ln_det_Sigma[0],
        )
        self.lr = lr

    def update_hyperparameters(
        self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray, **kwargs
    ):
        """Update the hyperparameters C,d,Qx,W.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: densities.GaussianDensity
        :param X: Observations.
        :type X: jnp.ndarray
        """
        phi = smoothing_density.slice(jnp.arange(1, smoothing_density.R))
        self.update_Qx(phi, X)
        self.update_Cd(phi, X)
        self.update_emission_density()
        self.update_W(phi, X)
        self.update_emission_density()

    def update_emission_density(self):
        """Create new emission density with current parameters.
        """
        self.emission_density = conditionals.LSEMGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            W=self.W,
            Sigma=jnp.array([self.Qx]),
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.emission_density.Lambda[0],
            self.emission_density.ln_det_Sigma[0],
        )

    def update_Qx(self, smoothing_density: densities.GaussianDensity, X: jnp.array):
        """Update observation covariance matrix Qx.
        
        Qx* = E[(X-C phi(z) - d)(X-C phi(z) - d)']

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: densities.GaussianDensity
        :param X: Observations.
        :type X: jnp.ndarray
        """
        T = X.shape[0]
        mu_x, Sigma_x = self.emission_density.get_expected_moments(smoothing_density)
        sum_mu_x2 = jnp.sum(
            Sigma_x - self.emission_density.Sigma + mu_x[:, None] * mu_x[:, :, None],
            axis=0,
        )
        sum_X_mu = jnp.sum(X[:, None] * mu_x[:, :, None], axis=0)
        self.Qx = (
            jnp.sum(X[:, None] * X[:, :, None], axis=0)
            - sum_X_mu
            - sum_X_mu.T
            + sum_mu_x2
        ) / T

    def update_Cd(self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray):
        """Update observation observation matrix C and vector d.
        
        C* = E[(X - d)phi(z)']E[phi(z)phi(z)']^{-1}
        d* = E[(X - C phi(x))]

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: densities.GaussianDensity
        :param X: Observations.
        :type X: jnp.ndarray
        """
        #### E[f(x)] ####
        # E[x] [R, Dx]
        T = X.shape[0]
        Ex = smoothing_density.integrate("x")
        # E[k(x)] [R, Dphi - Dx]
        p_k = smoothing_density.multiply(self.emission_density.k_func, update_full=True)
        Ekx = p_k.integrate().reshape((smoothing_density.R, self.Dphi - self.Dz))
        # E[f(x)]
        Ef = jnp.concatenate([Ex, Ekx], axis=1)
        B = jnp.einsum("ab,ac->bc", X - self.d[None], Ef)

        #### E[f(x)f(x)'] ####
        # Eff = jnp.empty([p_x.R, self.Dphi, self.Dphi])
        # Linear terms E[xx']
        Exx = jnp.sum(smoothing_density.integrate("xx"), axis=0)
        # Eff[:,:self.Dx,:self.Dx] =
        # Cross terms E[x k(x)']
        Ekx = jnp.sum(
            p_k.integrate("x").reshape((smoothing_density.R, self.Dk, self.Dz)), axis=0
        )
        # Eff[:,:self.Dx,self.Dx:] = jnp.swapaxes(Ekx, axis1=1, axis2=2)
        # Eff[:,self.Dx:,:self.Dx] = Ekx
        # kernel terms E[k(x)k(x)']
        Ekk = jnp.sum(
            p_k.multiply(self.emission_density.k_func, update_full=True)
            .integrate()
            .reshape((smoothing_density.R, self.Dk, self.Dk)),
            axis=0,
        )
        # Eff[:,self.Dx:,self.Dx:] = Ekk
        A = jnp.block([[Exx, Ekx.T], [Ekx, Ekk]])
        self.C = jnp.linalg.solve(A / T, B.T / T).T
        self.d = jnp.mean(X, axis=0) - jnp.dot(self.C, jnp.mean(Ef, axis=0))

    def update_W(self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray):
        """Updates the kernel weights.
        
        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: densities.GaussianDensity
        :param X: Observations.
        :type X: jnp.ndarray
        """
        opt = objax.optimizer.SGD(self.vars())
        T = X.shape[0]

        @objax.Function.with_vars(self.vars())
        def loss():
            self.emission_density.W = self.W
            return -self.compute_Q_function(smoothing_density, X) / T

        gv = objax.GradValues(loss, self.vars())

        @objax.Function.with_vars(self.vars() + opt.vars())
        def train_op():
            g, v = gv()  # returns gradients g and loss v
            opt(self.lr, g)  # update weights
            return v

        train_op = objax.Jit(train_op)

        for i in range(1000):
            v = train_op()


class HCCovObservationModel(LinearObservationModel):
    def __init__(self, Dx: int, Dz: int, Du: int, noise_x: float = 1.0):
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
        rand_mat = np.random.rand(self.Dx, self.Dx) - 0.5
        Q, R = np.linalg.qr(rand_mat)
        self.U = jnp.array(Q[:, : self.Du])
        # self.U = jnp.eye(Dx)[:, :Du]
        W = 1e-2 * np.random.randn(self.Du, self.Dz + 1)
        W[:, 0] = 0
        self.W = jnp.array(W)
        self.beta = noise_x ** 2 * jnp.ones(self.Du)
        self.sigma_x = jnp.array([noise_x])
        self.emission_density = conditionals.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            sigma_x=self.sigma_x,
            U=self.U,
            W=self.W,
            beta=self.beta,
        )
        self.get_omegas = jit(
            vmap(
                HCCovObservationModel.get_omegas_i,
                in_axes=(None, None, 0, 1, 0, None, None, None),
                out_axes=(0, 0, 0),
                axis_name="i",
            )
        )
        self.func1 = jit(HCCovObservationModel.Qfunc, static_argnums=(6, 7, 8))
        self.grad_func1 = jit(
            value_and_grad(HCCovObservationModel.Qfunc), static_argnums=(6, 7, 8)
        )
        self.func2 = jit(HCCovObservationModel.Qfunc_U_ls, static_argnums=(5, 6, 7))
        self.grad_func2 = jit(
            value_and_grad(HCCovObservationModel.Qfunc_U_ls), static_argnums=(5, 6, 7)
        )

    def lin_om_init(self, lin_om: LinearObservationModel):
        # self.sigma_x = jnp.array([jnp.sqrt(jnp.amin(lin_om.Qx.diagonal()))])
        beta, U = np.linalg.eig(lin_om.Qx)
        beta, U = jnp.array(beta), jnp.array(U)
        if self.Dx != 1:
            self.U = jnp.real(U[:, : self.Du])
            # self.beta = jnp.abs(jnp.real(beta[:self.Du]))
            # self.beta.at[(self.beta / self.sigma_x ** 2) < .5].set(.5 / self.sigma_x ** 2)
        # self.C = lin_om.C
        # self.d = lin_om.d
        self.update_emission_density()

    def pca_init(self, X: jnp.ndarray, smooth_window: int = 10):
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
            X_smoothed[:, i] = np.convolve(
                X[:, i], np.ones(smooth_window) / smooth_window, mode="same"
            )
        eig_vals, eig_vecs = scipy.linalg.eigh(
            jnp.dot((X_smoothed - self.d[None]).T, X_smoothed - self.d[None]),
            eigvals=(self.Dx - np.amin([self.Dz, self.Dx]), self.Dx - 1),
        )
        C = np.array(self.C)
        C[:, : np.amin([self.Dz, self.Dx])] = eig_vecs * eig_vals / T
        self.C = jnp.array(self.C)
        z_hat = jnp.dot(jnp.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - jnp.dot(z_hat, self.C.T) - self.d
        cov = jnp.dot(delta_X.T, delta_X)
        self.U = jnp.array(
            scipy.linalg.eigh(cov, eigvals=(self.Dx - self.Du, self.Dx - 1))[1]
        )
        self.emission_density = conditionals.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            sigma_x=self.sigma_x,
            U=self.U,
            W=self.W,
            beta=self.beta,
        )

    def compute_Q_function(
        self, smoothing_density: densities.GaussianDensity, X: jnp.ndarray,
    ):
        T = X.shape[0]
        phi_dict = smoothing_density.slice(jnp.arange(1, smoothing_density.R)).to_dict()
        params = conditionals.HCCovGaussianConditional.params_to_vector(
            self.C, self.d, self.sigma_x, self.beta, self.W
        )
        omega_dagger, omega_star, not_converged = self.get_omegas(
            phi_dict, X, self.W, self.U, self.beta, self.C, self.d, self.sigma_x
        )
        Q_val = (
            conditionals.HCCovGaussianConditional.Qfunc(
                params,
                phi_dict,
                X,
                self.U,
                omega_dagger,
                omega_star,
                self.Dx,
                self.Dz,
                self.Du,
            )
            * T
        )

    def update_hyperparameters(
        self,
        smoothing_density: densities.GaussianDensity,
        X: jnp.ndarray,
        iteration: int,
        **kwargs
    ):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: jnp.ndarray [T, Dx]
            The observations.
        """
        phi_dict = smoothing_density.slice(jnp.arange(1, smoothing_density.R)).to_dict()
        val_old = -np.inf
        converged = False
        num_iter = 0

        while not converged and num_iter < 1000:
            val = self.step(phi_dict, X, num_iter)
            converged = jnp.abs((val - val_old) / val_old) < 1e-5
            val_old = val
            num_iter += 1
        print(num_iter)
        self.update_emission_density()

    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.emission_density = conditionals.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            sigma_x=self.sigma_x,
            U=self.U,
            W=self.W,
            beta=self.beta,
        )

    def step(self, phi_dict, X, iteration, step_size: float = 0.001):
        omega_dagger, omega_star, not_converged = self.get_omegas(
            phi_dict, X, self.W, self.U, self.beta, self.C, self.d, self.sigma_x
        )
        # print(num_iter)
        T = X.shape[0]
        params = HCCovObservationModel.params_to_vector(
            self.C, self.d, self.sigma_x, self.beta, self.W
        )
        val, euclid_grad = self.grad_func1(
            params,
            phi_dict,
            X,
            self.U,
            omega_dagger,
            omega_star,
            self.Dx,
            self.Dz,
            self.Du,
        )
        params_new = self.euclid_step(params, euclid_grad, phi_dict, X)
        C, d, sigma_x, beta, W = HCCovObservationModel.vector_to_params(
            params_new, self.Dx, self.Dz, self.Du
        )
        self.C = C
        self.d = d
        self.sigma_x = sigma_x
        self.beta = beta
        self.W = W
        if self.Dx > 1:
            omega_dagger, omega_star, num_iter = self.get_omegas(
                phi_dict, X, self.W, self.U, self.beta, self.C, self.d, self.sigma_x
            )
            val, euclid_grad_U = self.grad_func2(
                self.U,
                phi_dict,
                X,
                omega_dagger,
                omega_star,
                self.Dx,
                self.Dz,
                self.Du,
                self.C,
                self.d,
                self.sigma_x,
                self.beta,
                self.W,
            )
            U = self.stiefel_step(euclid_grad_U, phi_dict, X, step_size)
            self.U = U
        return val

    def euclid_step(self, params, param_grad: jnp.array, phi_dict, X):
        def objective(t):
            params_new = params + t * param_grad
            C, d, sigma_x, beta, W = HCCovObservationModel.vector_to_params(
                params_new, self.Dx, self.Dz, self.Du
            )
            omega_dagger, omega_star, not_converged = self.get_omegas(
                phi_dict, X, W, self.U, beta, C, d, sigma_x
            )
            val = -self.func1(
                params_new,
                phi_dict,
                X,
                self.U,
                omega_dagger,
                omega_star,
                self.Dx,
                self.Dz,
                self.Du,
            )
            return val

        min_res = minimize_scalar(objective, tol=1e-4)
        opt_t = min_res.x
        params_new = params + opt_t * param_grad
        return params_new

    def stiefel_step(self, euclid_dU: jnp.ndarray, phi_dict, X, step_size):
        grad_stiefel = euclid_dU - jnp.dot(jnp.dot(self.U, euclid_dU.T), self.U)
        tangent = jnp.dot(self.U.T, euclid_dU) - jnp.dot(euclid_dU.T, self.U)
        # tangent = .5 * (tangent - tangent.T)
        geodesic = lambda t: jnp.dot(self.U, jsc.linalg.expm(t * tangent))

        def objective(t):
            U = geodesic(t)
            omega_dagger, omega_star, not_converged = self.get_omegas(
                phi_dict, X, self.W, U, self.beta, self.C, self.d, self.sigma_x
            )
            val = -self.func2(
                U,
                phi_dict,
                X,
                omega_dagger,
                omega_star,
                self.Dx,
                self.Dz,
                self.Du,
                self.C,
                self.d,
                self.sigma_x,
                self.beta,
                self.W,
            )
            return val

        min_res = minimize_scalar(objective, tol=1e-4)
        opt_t = min_res.x
        U_new = geodesic(opt_t)
        return U_new

    def apply_step(
        self, params, euclid_grad, step_size: float, phi_dict, X, get_omegas
    ):
        params = params + step_size * euclid_grad
        if self.Dx > 1:
            num_params = (self.Dz + 1) * self.Dx
            cur_params = self.Dx * self.Du
            euclid_dU = euclid_grad[
                jnp.arange(num_params, num_params + cur_params)
            ].reshape((self.Dx, self.Du))
            U_new = self.stiefel_step(euclid_dU, phi_dict, X, get_omegas, step_size)
            params = params.at[jnp.arange(num_params, num_params + cur_params)].set(
                U_new.flatten()
            )
        else:
            num_params = (self.Dz + 1) * self.Dx
            cur_params = self.Dx * self.Du
            params = params.at[jnp.arange(num_params, num_params + cur_params)].set(1)
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
        sigma_x = jnp.exp(0.5 * params[jnp.arange(num_params, num_params + cur_params)])
        num_params += cur_params
        cur_params = Du
        y = jnp.exp(params[jnp.arange(num_params, num_params + cur_params)])
        beta = 0.25 * sigma_x ** 2 + y
        num_params += cur_params
        cur_params = Du * (Dz + 1)
        W = params[jnp.arange(num_params, num_params + cur_params)].reshape(
            (Du, Dz + 1)
        )
        return C, d, sigma_x, beta, W

    @staticmethod
    def params_to_vector(
        C: jnp.ndarray,
        d: jnp.ndarray,
        sigma_x: jnp.ndarray,
        beta: jnp.ndarray,
        W: jnp.ndarray,
    ):
        C_flattened = C.flatten()
        ln_sigma2_x = 2.0 * jnp.log(sigma_x)
        ln_y = jnp.log(beta - 0.25 * sigma_x ** 2)
        W_flattened = W.flatten()
        params = jnp.concatenate([C_flattened, d, ln_sigma2_x, ln_y, W_flattened])
        return params

    @staticmethod
    def Qfunc(
        params: jnp.ndarray,
        phi_dict: dict,
        X: jnp.ndarray,
        U: jnp.array,
        omega_dagger: jnp.array,
        omega_star: jnp.array,
        Dx: int,
        Dz: int,
        Du: int,
    ):
        phi = densities.GaussianDensity(**phi_dict)
        C, d, sigma_x, beta, W = HCCovObservationModel.vector_to_params(
            params, Dx, Dz, Du
        )
        T = X.shape[0]
        vec = X - d
        E_epsilon2 = jnp.sum(
            phi.integrate("Ax_aBx_b_inner", A_mat=-C, a_vec=vec, B_mat=-C, b_vec=vec),
            axis=0,
        )
        uRu, log_lb_sum = HCCovObservationModel.get_lb_i(
            phi, X, W, U, beta, omega_dagger, omega_star, C, d, sigma_x
        )
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -0.5 * (E_epsilon2 - E_D_inv_epsilon2) / sigma_x ** 2
        # determinant part
        Qm = Qm - 0.5 * E_ln_sigma2_f + 0.5 * T * (Du - Dx) * jnp.log(sigma_x ** 2)
        return jnp.squeeze(Qm) / T

    @staticmethod
    def Qfunc_U_ls(
        U,
        phi_dict: dict,
        X: jnp.ndarray,
        omega_dagger: jnp.array,
        omega_star: jnp.array,
        Dx: int,
        Dz: int,
        Du: int,
        C,
        d,
        sigma_x,
        beta,
        W,
    ):
        T = X.shape[0]
        phi = densities.GaussianDensity(**phi_dict)
        vec = X - d
        E_epsilon2 = jnp.sum(
            phi.integrate("Ax_aBx_b_inner", A_mat=-C, a_vec=vec, B_mat=-C, b_vec=vec),
            axis=0,
        )
        uRu, log_lb_sum = HCCovObservationModel.get_lb_i(
            phi, X, W, U, beta, omega_dagger, omega_star, C, d, sigma_x
        )
        E_D_inv_epsilon2 = jnp.sum(uRu, axis=0)
        E_ln_sigma2_f = jnp.sum(log_lb_sum, axis=0)
        Qm = -0.5 * (E_epsilon2 - E_D_inv_epsilon2) / sigma_x ** 2
        # determinant part
        Qm = Qm - 0.5 * E_ln_sigma2_f + 0.5 * T * (Du - Dx) * jnp.log(sigma_x ** 2)
        return jnp.squeeze(Qm) / T

    ###################### Lower bound functions #######################################################################
    @staticmethod
    def get_omegas_i(
        phi_dict: dict,
        X: jnp.ndarray,
        W_i,
        u_i,
        beta,
        C,
        d,
        sigma_x,
        conv_crit: float = 1e-3,
    ):
        T = X.shape[0]
        phi = densities.GaussianDensity(**phi_dict)
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        u_i = u_i.reshape((-1, 1))
        uC = jnp.dot(u_i.T, -C)
        ux_d = jnp.dot(u_i.T, (X - d).T)
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = jnp.sqrt(
            phi.integrate("Ax_aBx_b_inner", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        )
        omega_star = 1e-15 * jnp.ones(T)
        # omega_star = omega_star_init
        omega_old = 10 * jnp.ones(T)

        def body_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # From the lower bound term
            g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
            nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
            nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
            ln_beta = (
                -jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta))
                - 0.5 * g_omega * (b_i ** 2 - omega_star ** 2)
                + jnp.log(beta)
            )
            ln_beta_plus = ln_beta + b_i
            ln_beta_minus = ln_beta - b_i
            # Create OneRankFactors
            exp_factor_plus = factors.OneRankFactor(
                v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
            )
            exp_factor_minus = factors.OneRankFactor(
                v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
            )
            # Create the two measures
            exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
            exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
            # Fourth order integrals E[h^2 (x-Cz-d)^2]
            quart_int_plus = exp_phi_plus.integrate(
                "Ax_aBx_bCx_cDx_d_inner",
                A_mat=uC,
                a_vec=ux_d.T,
                B_mat=uC,
                b_vec=ux_d.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int_minus = exp_phi_minus.integrate(
                "Ax_aBx_bCx_cDx_d_inner",
                A_mat=uC,
                a_vec=ux_d.T,
                B_mat=uC,
                b_vec=ux_d.T,
                C_mat=w_i,
                c_vec=b_i,
                D_mat=w_i,
                d_vec=b_i,
            )
            quart_int = quart_int_plus + quart_int_minus
            # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate(
                "Ax_aBx_b_inner", A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T
            )
            quad_int_minus = exp_phi_minus.integrate(
                "Ax_aBx_b_inner", A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T
            )
            quad_int = quad_int_plus + quad_int_minus
            omega_old = omega_star
            omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))
            num_iter = num_iter + 1
            return omega_star, omega_old, num_iter

        def cond_fun(omegas):
            omega_star, omega_old, num_iter = omegas
            # return lax.pmax(jnp.amax(jnp.abs(omega_star - omega_old)), 'i') > conv_crit
            return jnp.logical_and(
                jnp.amax(jnp.amax(jnp.abs(omega_star - omega_old) / omega_star))
                > conv_crit,
                num_iter < 100,
            )

        num_iter = 0
        init_val = (omega_star, omega_old, num_iter)
        omega_star, omega_old, num_iter = lax.while_loop(cond_fun, body_fun, init_val)
        indices_non_converged = (
            jnp.abs(omega_star - omega_old) / omega_star
        ) > conv_crit

        return omega_dagger, omega_star, indices_non_converged

    @staticmethod
    @partial(
        vmap, in_axes=(None, None, 0, 1, 0, 0, 0, None, None, None), out_axes=(0, 0)
    )
    def get_lb_i(
        phi: densities.GaussianDensity,
        X: jnp.ndarray,
        W_i,
        u_i,
        beta,
        omega_dagger,
        omega_star,
        C,
        d,
        sigma_x,
    ):
        # phi = densities.GaussianDensity(**phi_dict)
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

        omega_dagger = jnp.sqrt(
            phi.integrate("Ax_aBx_b_inner", A_mat=w_i, a_vec=b_i, B_mat=w_i, b_vec=b_i)
        )
        g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
        nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
        ln_beta = (
            -jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta))
            - 0.5 * g_omega * (b_i ** 2 - omega_star ** 2)
            + jnp.log(beta)
        )
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factors.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factors.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        # Create the two measures
        exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
        # Fourth order integrals E[h^2 (x-Cz-d)^2]
        quart_int_plus = exp_phi_plus.integrate(
            "Ax_aBx_bCx_cDx_d_inner",
            A_mat=uC,
            a_vec=ux_d.T,
            B_mat=uC,
            b_vec=ux_d.T,
            C_mat=w_i,
            c_vec=b_i,
            D_mat=w_i,
            d_vec=b_i,
        )
        quart_int_minus = exp_phi_minus.integrate(
            "Ax_aBx_bCx_cDx_d_inner",
            A_mat=uC,
            a_vec=ux_d.T,
            B_mat=uC,
            b_vec=ux_d.T,
            C_mat=w_i,
            c_vec=b_i,
            D_mat=w_i,
            d_vec=b_i,
        )
        quart_int = quart_int_plus + quart_int_minus
        # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
        quad_int_plus = exp_phi_plus.integrate(
            "Ax_aBx_b_inner", A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T
        )
        quad_int_minus = exp_phi_minus.integrate(
            "Ax_aBx_b_inner", A_mat=uC, a_vec=ux_d.T, B_mat=uC, b_vec=ux_d.T
        )
        quad_int = quad_int_plus + quad_int_minus
        omega_star = jnp.sqrt(jnp.abs(quart_int / quad_int))
        f_omega_dagger = HCCovObservationModel.f(omega_dagger, beta)
        log_lb = jnp.log(sigma_x ** 2 + f_omega_dagger)
        g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
        nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
        ln_beta = (
            -jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta))
            - 0.5 * g_omega * (b_i ** 2 - omega_star ** 2)
            + jnp.log(beta)
        )
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factors.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factors.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        # Create the two measures
        exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
        mat1 = -C
        vec1 = X - d
        R_plus = exp_phi_plus.integrate(
            "Ax_aBx_b_outer", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R_minus = exp_phi_minus.integrate(
            "Ax_aBx_b_outer", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R = R_plus + R_minus
        R = jnp.sum(R, axis=0)
        # R = .5 * (R + R.T)
        uRu = jnp.sum(u_i * jnp.dot(R, u_i))
        log_lb_sum = jnp.sum(log_lb)
        return uRu, log_lb_sum

    @staticmethod
    @partial(
        vmap, in_axes=(None, None, 0, 1, 0, 0, 0, None, None, None), out_axes=(0, 0)
    )
    def get_R(
        phi: densities.GaussianDensity,
        X: jnp.ndarray,
        W_i,
        beta,
        u_i,
        omega_star,
        C,
        d,
        sigma_x,
    ):

        # Lower bound for E[ln (sigma_x^2 + f(h))]
        T = X.shape[0]
        w_i = W_i[1:].reshape((1, -1))
        v = jnp.tile(w_i, (T, 1))
        b_i = W_i[:1]
        g_omega = HCCovObservationModel.g(omega_star, beta, sigma_x)
        nu_plus = (1.0 - g_omega[:, None] * b_i) * w_i
        nu_minus = (-1.0 - g_omega[:, None] * b_i) * w_i
        ln_beta = (
            -jnp.log(sigma_x ** 2 + HCCovObservationModel.f(omega_star, beta))
            - 0.5 * g_omega * (b_i ** 2 - omega_star ** 2)
            + jnp.log(beta)
        )
        ln_beta_plus = ln_beta + b_i
        ln_beta_minus = ln_beta - b_i
        # Create OneRankFactors
        exp_factor_plus = factors.OneRankFactor(
            v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus
        )
        exp_factor_minus = factors.OneRankFactor(
            v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus
        )
        # Create the two measures
        exp_phi_plus = phi.hadamard(exp_factor_plus, update_full=True)
        exp_phi_minus = phi.hadamard(exp_factor_minus, update_full=True)
        mat1 = -C
        vec1 = X - d
        R_plus = exp_phi_plus.integrate(
            "Ax_aBx_b_outer", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R_minus = exp_phi_minus.integrate(
            "Ax_aBx_b_outer", A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1
        )
        R = R_plus + R_minus
        R = jnp.mean(R, axis=0)
        # R = R / beta
        R = 0.5 * (R + R.T)
        return R

    ####################### Functions for bounds of non tractable terms in the Q-function ##############################
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
        return (
            HCCovObservationModel.f_prime(omega, beta)
            / (sigma_x ** 2 + HCCovObservationModel.f(omega, beta))
            / jnp.abs(omega)
        )

    @staticmethod
    def g_prime(omega, beta, sigma_x):
        """ Computes the function
        
            g(omega) = f'(omega) / (sigma_x^2 + f(omega)) / |omega|
            
            for the variational boind
            
        :param omega: jnp.ndarray
            Free variational parameter.
        :param beta: jnp.ndarray
            Scaling factor.
        """
        f = HCCovObservationModel.f(omega, beta)
        denominator = (sigma_x ** 2 + f) * jnp.abs(omega)

        g_p = f / denominator
        f_p = HCCovObservationModel.f_prime(omega, beta)
        g_p -= f_p * (f_p * jnp.abs(omega) + sigma_x ** 2 + f) / denominator ** 2
        return g_p


class BernoulliObservationModel(ObservationModel):
    def __init__(self, Dx: int, Dz: int, Dphi_u: int = 0):
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

    def compute_feature_vector(
        self, z: jnp.ndarray, ux: jnp.ndarray = None
    ) -> jnp.ndarray:
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
        phi[:, 0] = 1
        phi[:, 1 : self.Dz + 1] = z
        if ux is not None:
            phi[:, self.Dz + 1 :] = ux
        return phi

    def compute_expected_feature_vector(
        self, density: densities.GaussianDensity, ux: jnp.ndarray = None
    ) -> jnp.ndarray:
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
        Ephi[:, :, 0] = 1
        Ephi[:, :, 1 : self.Dz + 1] = density.integrate("x")[:, None]
        if ux is not None:
            Ephi[:, :, self.Dz + 1 :] = ux
        return Ephi

    def compute_expected_feature_outer_product(
        self, density: densities.GaussianDensity, ux: jnp.ndarray = None
    ) -> jnp.ndarray:
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

        Ez = density.integrate("x")
        Ezz = density.integrate("xx")

        Ephi_outer = jnp.zeros((T, self.Dx, self.Dphi, self.Dphi))
        Ephi_outer[:, :, 0, 0] = 1  # 1
        Ephi_outer[:, :, 1 : self.Dz + 1, 0] = Ez  # E[z']
        Ephi_outer[:, :, 0, 1 : self.Dz + 1] = Ez  # E[z]
        Ephi_outer[:, :, 1 : self.Dz + 1, 1 : self.Dz + 1] = Ezz  # E[zz']
        if ux is not None:
            if ux.ndim == 2:
                ux = ux.reshape((ux.shape[0], 1, ux.shape[1]))
            Ez_ux = Ez[:, None, :, None] * ux[:, :, None]
            uxux = ux[:, :, None] * ux[:, :, :, None]
            Ephi_outer[:, :, self.Dz + 1 :, 0] = ux  # u'
            Ephi_outer[:, :, 0, self.Dz + 1 :] = ux  # u
            Ephi_outer[:, :, 1 : self.Dz + 1, self.Dz + 1 :] = Ez_ux  # E[z] u'
            Ephi_outer[:, :, self.Dz + 1 :, 1 : self.Dz + 1] = jnp.swapaxes(
                Ez_ux, axis1=2, axis2=3
            )  # E[z'] u
            Ephi_outer[:, :, self.Dz + 1 :, self.Dz + 1 :] = uxux  # uu'
        return Ephi_outer

    def get_omega_star(
        self,
        density: densities.GaussianDensity,
        x_t: jnp.ndarray,
        ux_t: jnp.ndarray = None,
        conv_crit: float = 1e-4,
    ) -> jnp.ndarray:
        """ Gets the optimal variational parameter.
        """

        omega_old = jnp.ones((self.Dx))
        converged = False
        v = self.Theta[:, 1 : self.Dz + 1]
        while not converged:
            g = 1.0 / jnp.abs(omega_old) * jnp.tanh(0.5 * omega_old)
            sign = 2.0 * x_t[0] - 1.0
            nu = (
                self.Theta[:, 1 : self.Dz + 1]
                * (0.5 * sign - g * self.Theta[:, 0])[:, None]
            )
            if ux_t is not None:
                theta_uz = jnp.einsum("ab,b->a", self.Theta[:, self.Dz + 1 :], ux_t[0])
                nu = nu - self.Theta[:, 1 : self.Dz + 1] * (g * theta_ux)[:, None]
                # ln_beta = ln_beta + .5 * sign * theta_uz - g * (.5 * theta_uz ** 2 + self.Theta[:,0] * ux_t[0])

            sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu)
            sigma_density = density.multiply(sigma_lb).get_density()
            A_mat = self.Theta[:, 1 : self.Dz + 1]
            a_vec = self.Theta[:, 0]
            if ux_t is not None:
                a_vec = a_vec + theta_ux
            Eh2 = sigma_density.integrate(
                "Ax_aBx_b_inner", A_mat=A_mat, a_vec=a_vec, B_mat=A_mat, b_vec=a_vec
            )
            omega_star = jnp.sqrt(Eh2)
            omega_star[omega_star < 1e-10] = 1e-10
            converged = jnp.amax(jnp.abs(omega_star - omega_old)) < conv_crit
            omega_old = jnp.copy(omega_star)
        return omega_star

    def filtering(
        self,
        prediction_density: densities.GaussianDensity,
        x_t: jnp.ndarray,
        ux_t: jnp.ndarray = None,
        **kwargs
    ) -> densities.GaussianDensity:
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
        v = self.Theta[:, 1 : self.Dz + 1]
        g = 1.0 / jnp.abs(omega_star) * jnp.tanh(0.5 * omega_star)
        sign = 2.0 * x_t[0] - 1.0
        nu = (
            self.Theta[:, 1 : self.Dz + 1]
            * (0.5 * sign - g * self.Theta[:, 0])[:, None]
        )
        if ux_t is not None:
            theta_ux = jnp.einsum("ab,b->a", self.Theta[:, self.Dz + 1 :], ux_t[0])
            nu = nu - self.Theta[:, 1 : self.Dz + 1] * (g * theta_ux)[:, None]
        sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu)
        filter_measure = prediction_density
        for idx in range(self.Dx):
            filter_measure = filter_measure.hadamard(sigma_lb.slice([idx]))
        filter_density = filter_measure.get_density()
        return filter_density

    def get_omega_dagger(
        self,
        density: densities.GaussianDensity,
        ux_t: jnp.ndarray = None,
        conv_crit: float = 1e-4,
    ) -> jnp.ndarray:
        """ Gets the optimal variational parameter.
        """

        A_mat = self.Theta[:, 1 : self.Dz + 1]
        a_vec = self.Theta[:, 0]
        if ux_t is not None:
            theta_ux = jnp.einsum("ab,b->a", self.Theta[:, self.Dz + 1 :], ux_t[0])
            a_vec = a_vec + theta_ux
        Eh2 = density.integrate(
            "Ax_aBx_b_inner", A_mat=A_mat, a_vec=a_vec, B_mat=A_mat, b_vec=a_vec
        )
        omega_dagger = jnp.sqrt(Eh2)
        omega_dagger[omega_dagger < 1e-10] = 1e-10
        return omega_dagger

    def update_hyperparameters(
        self,
        smoothing_density: densities.GaussianDensity,
        X: jnp.ndarray,
        u_x: jnp.ndarray = None,
        **kwargs
    ):
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
            density_t = smoothing_density.slice([t + 1])
            if u_x is not None:
                ux_t = u_x[t : t + 1]
            else:
                ux_t = None
            omega_dagger = self.get_omega_dagger(density_t, ux_t=ux_t)
            g = 1.0 / jnp.abs(omega_dagger) * jnp.tanh(0.5 * omega_dagger)
            Ephiphi = self.compute_expected_feature_outer_product(density_t, ux=ux_t)[0]
            A_theta = A_theta + g[:, None, None] * Ephiphi
            Ephi = self.compute_expected_feature_vector(density_t, ux=ux_t)[0]
            sign = 2.0 * X[t] - 1.0
            b_theta = b_theta + 0.5 * sign[:, None] * Ephi
        # A_theta += 1e-4 * jnp.eye(self.Dphi)[None]
        self.Theta = jnp.linalg.solve(A_theta, b_theta)

    def get_lb_sigma(
        self,
        density: densities.GaussianDensity,
        x_t: jnp.ndarray,
        ux_t: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """ Computes the lower bounds for the data probability.
        """
        omega_star = self.get_omega_star(density, x_t, ux_t)
        v = self.Theta[:, 1 : self.Dz + 1]
        g = 1.0 / jnp.abs(omega_star) * jnp.tanh(0.5 * omega_star)
        sign = 2.0 * x_t[0] - 1.0
        nu = (
            self.Theta[:, 1 : self.Dz + 1]
            * (0.5 * sign - g * self.Theta[:, 0])[:, None]
        )
        ln_beta = (
            -jnp.log(2)
            - logcosh(0.5 * omega_star)
            + 0.5 * sign * self.Theta[:, 0]
            - 0.5 * g * (self.Theta[:, 0] ** 2 - omega_star ** 2)
        )
        if ux_t is not None:
            theta_ux = jnp.einsum("ab,b->a", self.Theta[:, self.Dz + 1 :], ux_t[0])
            nu = nu - self.Theta[:, 1 : self.Dz + 1] * (g * theta_ux)[:, None]
            ln_beta = (
                ln_beta
                + 0.5 * sign * theta_ux
                - g * (0.5 * theta_ux ** 2 + self.Theta[:, 0] * ux_t[0])
            )
        sigma_lb = factors.OneRankFactor(v=v, g=g, nu=nu, ln_beta=ln_beta)
        measure = density
        for idx in range(self.Dx):
            measure = measure.hadamard(sigma_lb.slice([idx]))
        prob_lb = measure.integrate()[0]
        return prob_lb

    def evaluate_llk(
        self,
        p_z: densities.GaussianDensity,
        X: jnp.ndarray,
        u_x: jnp.ndarray = None,
        **kwargs
    ) -> float:
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
        # p_x = self.emission_density.affine_marginal_transformation(p_z)
        for t in range(0, T):
            if u_x is not None:
                ux_t = u_x[t : t + 1]
            else:
                ux_t = None
            prob_lb = self.get_lb_sigma(p_z.slice([t]), X[t : t + 1], ux_t=ux_t)
            llk += jnp.log(prob_lb)
        return llk
