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
from typing import Union

sys.path.append("../")
from jax import numpy as jnp
import numpy as np
from jax import jit, value_and_grad
import objax
from scipy.optimize import minimize as minimize_sc
from utils.jax_minimize_wrapper import minimize as minimize_jax

# from src_jax
from src_jax import densities, conditionals, factors

# from jax.scipy.optimize import minimize
# from jax.experimental import optimizers
# from tensorflow_probability.substrates import jax as tfp


class StateModel(objax.Module):
    def __init__(self):
        """ This is the template class for state transition models in state space models. 
        Basically these classes should contain all functionality for transition between time steps
        the latent variables z, i.e. p(z_{t+1}|z_t). The object should 
        have an attribute `state_density`, which is be a `ConditionalDensity`. 
        Furthermore, it should be possible to optimize hyperparameters, when provided 
        with a density over the latent space.
        """
        self.state_density = None

    def prediction(
        self, pre_filter_density: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Here the prediction density is calculated.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1
        
        :param pre_filter_density: GaussianDensity
            Density p(z_t-1|x_{1:t-1})
            
        :return: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        """
        raise NotImplementedError("Prediction for state model not implemented.")

    def smoothing(
        self,
        cur_filter_density: densities.GaussianDensity,
        post_smoothing_density: densities.GaussianDensity,
        **kwargs
    ) -> densities.GaussianDensity:
        """ Here we do the smoothing step to acquire p(z_{t} | x_{1:T}), 
        given p(z_{t+1} | x_{1:T}) and p(z_{t} | x_{1:t}).
        
        :param cur_filter_density: GaussianDensity
            Density p(z_t|x_{1:t})
        :param post_smoothing_density: GaussianDensity
            Density p(z_{t+1}|x_{1:T})
            
        :return: [GaussianDensity, GaussianDensity]
            Smoothing density p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T})
        """
        raise NotImplementedError("Smoothing for state model not implemented.")

    def update_hyperparameters(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        **kwargs
    ):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        raise NotImplementedError("Hyperparamers for state model not implemented.")

    def update_init_density(
        self, init_smooth_density: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Finds the optimal distribution over the initial state z_0, 
        provided with the initial smoothing density.
        
        :param init_smooth_density: GaussianDensity
            Smoothing density over z_0.
        
        :return: GaussianDensity
            The optimal initial distribution.
        """
        raise NotImplementedError(
            "Initial distribution update for state model not implemented."
        )


class LinearStateModel(StateModel):
    def __init__(self, Dz: int, noise_z: float = 1.0):
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
        self.state_density = conditionals.ConditionalGaussianDensity(
            jnp.array([self.A]), jnp.array([self.b]), jnp.array([self.Qz])
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def prediction(
        self, pre_filter_density: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Here the prediction density is calculated.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1
        
        :param pre_filter_density: GaussianDensity
            Density p(z_t-1|x_{1:t-1})
            
        :return: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        """
        # p(z_t|x_{1:t-1})
        return self.state_density.affine_marginal_transformation(
            pre_filter_density, **kwargs
        )

    def smoothing(
        self,
        cur_filter_density: densities.GaussianDensity,
        post_smoothing_density: densities.GaussianDensity,
        **kwargs
    ) -> densities.GaussianDensity:
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
        backward_density = self.state_density.affine_conditional_transformation(
            cur_filter_density, **kwargs
        )
        # p(z_{t}, z_{t+1} | x_{1:T})
        cur_two_step_smoothing_density = backward_density.affine_joint_transformation(
            post_smoothing_density
        )
        # p(z_{t} | x_{1:T})
        cur_smoothing_density = cur_two_step_smoothing_density.get_marginal(
            jnp.arange(self.Dz, 2 * self.Dz)
        )

        return cur_smoothing_density, cur_two_step_smoothing_density

    def compute_Q_function(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        **kwargs
    ) -> float:
        return jnp.sum(
            self.state_density.integrate_log_conditional(
                two_step_smoothing_density,
                p_x=smoothing_density.slice(jnp.arange(1, smoothing_density.R)),
            )
        )

    def update_hyperparameters(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        **kwargs
    ):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        self.update_A(smoothing_density, two_step_smoothing_density, **kwargs)
        self.update_b(smoothing_density, **kwargs)
        self.update_Qz(smoothing_density, two_step_smoothing_density, **kwargs)
        self.update_state_density()

    def update_A(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        **kwargs
    ):
        """ The transition matrix is updated here, where the the densities
        p(z_{t+1}, z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        # Ezz = smoothing_density.integrate('xx')
        mu_b = smoothing_density.mu[:-1, None] * self.b[None, :, None]
        Ezz_two_step = two_step_smoothing_density.integrate("xx")
        Ezz = Ezz_two_step[:, self.Dz :, self.Dz :]
        Ezz_cross = Ezz_two_step[:, self.Dz :, : self.Dz]
        A = jnp.mean(Ezz, axis=0)  # + 1e-2 * jnp.eye(self.Dz)
        self.A = jnp.linalg.solve(A, jnp.mean(Ezz_cross - mu_b, axis=0)).T

    def update_b(self, smoothing_density: densities.GaussianDensity, **kwargs):
        """ The transition offset is updated here, where the the densities p(z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        """
        self.b = jnp.mean(
            smoothing_density.mu[1:] - jnp.dot(self.A, smoothing_density.mu[:-1].T).T,
            axis=0,
        )

    def update_Qz(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        **kwargs
    ):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        A_tilde = jnp.eye(2 * self.Dz, self.Dz)
        A_tilde = A_tilde.at[self.Dz :].set(-self.A.T)
        # A_tilde = jnp.block([[jnp.eye(self.Dz), -self.A.T]])
        b_tilde = -self.b
        self.Qz = jnp.mean(
            two_step_smoothing_density.integrate(
                "Ax_aBx_b_outer",
                A_mat=A_tilde.T,
                a_vec=b_tilde,
                B_mat=A_tilde.T,
                b_vec=b_tilde,
            ),
            axis=0,
        )

    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.ConditionalGaussianDensity(
            jnp.array([self.A]), jnp.array([self.b]), jnp.array([self.Qz])
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def update_init_density(
        self, init_smooth_density: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Finds the optimal distribution over the initial state z_0, 
        provided with the initial smoothing density.
        
        :param init_smooth_density: GaussianDensity
            Smoothing density over z_0.
        
        :return: GaussianDensity
            The optimal initial distribution.
        """
        mu0 = init_smooth_density.integrate("x")
        Sigma0 = init_smooth_density.integrate(
            "Ax_aBx_b_outer", A_mat=None, a_vec=-mu0[0], B_mat=None, b_vec=-mu0[0]
        )
        opt_init_density = densities.GaussianDensity(Sigma0, mu0)
        return opt_init_density

    def condition_on_past(
        self, z_old: jnp.ndarray, **kwargs
    ) -> densities.GaussianDensity:
        """ Return p(Z_t+1|Z_t=z)

        :param z_old: Vector of past latent variables.
        :type z_old: jnp.ndarray
        :return: The density of the latent variables at the next step.
        :rtype: densities.GaussianDensity
        """
        return self.state_density.condition_on_x(z_old, **kwargs)


class NNControlStateModel(LinearStateModel):
    def __init__(
        self,
        Dz: int,
        Du: int,
        noise_z: float = 1,
        hidden_units: list = [16,],
        non_linearity: callable = objax.functional.tanh,
        lr: float = 1e-4,
    ):
        self.Dz = Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.Du = Du
        self.state_density = conditionals.NNControlGaussianConditional(
            Sigma=jnp.array([self.Qz]),
            Dx=self.Dz,
            Du=self.Du,
            hidden_units=hidden_units,
            non_linearity=non_linearity,
        )
        self.lr = 1e-5

    def update_hyperparameters(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        u: jnp.ndarray,
        **kwargs
    ):
        """ The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        self.update_network_params(
            smoothing_density, two_step_smoothing_density, u, **kwargs
        )
        self.update_Qz(smoothing_density, two_step_smoothing_density, u, **kwargs)
        self.update_state_density()

    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density.update_Sigma(jnp.array([self.Qz]))

    def update_Qz(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        u: jnp.ndarray,
        **kwargs
    ):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        A_u, b_u = self.state_density.get_M_b(u)
        A_tilde = jnp.empty((two_step_smoothing_density.R, self.Dz, 2 * self.Dz))
        A_tilde = A_tilde.at[:, :, : self.Dz].set(jnp.eye(self.Dz))
        A_tilde = A_tilde.at[:, :, self.Dz :].set(-A_u)
        b_tilde = -b_u
        self.Qz = jnp.mean(
            two_step_smoothing_density.integrate(
                "Ax_aBx_b_outer",
                A_mat=A_tilde,
                a_vec=b_tilde,
                B_mat=A_tilde,
                b_vec=b_tilde,
            ),
            axis=0,
        )

    def update_network_params(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        u: jnp.ndarray,
        **kwargs
    ):
        gv = objax.GradValues(self.calc_neg_Q_function, self.vars())
        opt = objax.optimizer.Adam(self.vars())

        def train_op():
            g, v = gv(
                smoothing_density, two_step_smoothing_density, u
            )  # returns gradients, loss
            opt(self.lr, g)
            return v

        train_op = objax.Jit(train_op, gv.vars() + opt.vars())
        for i in range(1000):
            v = train_op()

    def calc_neg_Q_function(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        u: jnp.ndarray,
        **kwargs
    ):
        T = two_step_smoothing_density.R
        A_u, b_u = self.state_density.get_M_b(u)
        A_tilde = jnp.empty((two_step_smoothing_density.R, self.Dz, 2 * self.Dz))
        A_tilde = A_tilde.at[:, :, : self.Dz].set(jnp.eye(self.Dz))
        A_tilde = A_tilde.at[:, :, self.Dz :].set(-A_u)
        b_tilde = -b_u
        A_tilde2 = jnp.einsum("ab,cbd->cad", self.state_density.Lambda[0], A_tilde)
        b_tilde2 = jnp.einsum("ab,cb->ca", self.state_density.Lambda[0], b_tilde)
        expectation_term = jnp.mean(
            two_step_smoothing_density.integrate(
                "Ax_aBx_b_inner",
                A_mat=A_tilde,
                a_vec=b_tilde,
                B_mat=A_tilde2,
                b_vec=b_tilde2,
            ),
            axis=0,
        )
        Q_func = -0.5 * (
            expectation_term
            + self.Dz * jnp.log(2 * jnp.pi)
            + self.state_density.ln_det_Sigma
        )
        return -Q_func.squeeze()


class LSEMStateModel(LinearStateModel):
    def __init__(self, Dz: int, Dk: int, noise_z: float = 1.0):
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
        self.A = self.A.at[:, : self.Dz].set(jnp.eye(self.Dz))
        self.b = jnp.zeros((self.Dz,))
        self.W = objax.TrainVar(jnp.array(np.random.randn(self.Dk, self.Dz + 1)))
        self.state_density = conditionals.LSEMGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            W=self.W,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def update_hyperparameters(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        **kwargs
    ):
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
        self.A, self.b, self.Qz = self.update_AbQ(
            smoothing_density, two_step_smoothing_density
        )
        # print(self.b)
        # print('AbQ: Run Time %.1f' % (time.perf_counter() - time_start))
        # time_start = time.perf_counter()
        self.update_state_density()
        self.update_W(smoothing_density, two_step_smoothing_density)
        # print('W: Run Time %.1f' % (time.perf_counter() - time_start))
        # time_start = time.perf_counter()
        self.update_state_density()
        # print('Update: Run Time %.1f' % (time.perf_counter() - time_start))

    def update_AbQ(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
    ):
        """ The transition matrix is updated here, where the the densities
        p(z_{t+1}, z_t|x_{1:T}) is provided.

        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        phi = smoothing_density.slice(jnp.arange(T))

        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:, : self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.mean(
            two_step_smoothing_density.integrate(
                "Ax_aBx_b_outer",
                A_mat=A_tilde.T,
                a_vec=b_tilde,
                B_mat=A_tilde.T,
                b_vec=b_tilde,
            ),
            axis=0,
        )
        # v_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                      self.state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                       self.state_density.k_func.nu])
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta
        )
        two_step_k_measure = two_step_smoothing_density.multiply(
            joint_k_func, update_full=True
        )
        Ekz = jnp.mean(
            two_step_k_measure.integrate("x").reshape((T, self.Dk, 2 * self.Dz)), axis=0
        )
        Ekz_future, Ekz_past = Ekz[:, : self.Dz], Ekz[:, self.Dz :]
        phi_k = phi.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.mean(phi_k.integral_light().reshape((T, self.Dk)), axis=0)
        Qz_k_lin_err = jnp.dot(
            self.A[:, self.Dz :],
            (
                Ekz_future
                - jnp.dot(self.A[:, : self.Dz], Ekz_past.T).T
                - Ek[:, None] * self.b[None]
            ),
        )
        mean_Ekk = jnp.mean(
            phi_k.multiply(self.state_density.k_func, update_full=True)
            .integral_light()
            .reshape((T, self.Dk, self.Dk)),
            axis=0,
        )
        Qz_kk = jnp.dot(jnp.dot(self.A[:, self.Dz :], mean_Ekk), self.A[:, self.Dz :].T)
        Qz = Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T
        Qz = 0.5 * (Qz + Qz.T)

        # E[f(z)f(z)']
        Ekk = (
            phi_k.multiply(self.state_density.k_func, update_full=True)
            .integral_light()
            .reshape((T, self.Dk, self.Dk))
        )
        Ekz = phi_k.integrate("x").reshape((T, self.Dk, self.Dz))
        mean_Ekz = jnp.mean(Ekz, axis=0)
        mean_Ezz = jnp.mean(phi.integrate("xx"), axis=0)
        mean_Ekk = jnp.mean(Ekk, axis=0)
        Eff = jnp.block([[mean_Ezz, mean_Ekz.T], [mean_Ekz, mean_Ekk]])
        Eff += 0.001 * jnp.eye(Eff.shape[0])
        # mean_Ekk_reg = mean_Ekk + .0001 * jnp.eye(self.Dk)
        # mean_Ezz = jnp.mean(phi.integrate('xx'), axis=0)
        # Eff = jnp.block([[mean_Ezz, Ekz_past.T],
        #                  [Ekz_past, mean_Ekk_reg]])
        Ez = jnp.mean(phi.integrate("x"), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:, None]
        # E[z f(z)']
        Ezz_cross = jnp.mean(
            two_step_smoothing_density.integrate("xx")[:, self.Dz :, : self.Dz], axis=0
        )
        Ezf = jnp.concatenate([Ezz_cross.T, Ekz_future.T], axis=1)
        A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        b = jnp.mean(smoothing_density.mu[1:], axis=0) - jnp.dot(A, Ef).T
        # A = self.A
        # b = self.b
        return A, b, Qz

    def update_A(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
    ):
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
        Ekk = (
            phi_k.multiply(self.state_density.k_func, update_full=True)
            .integral_light()
            .reshape((T, self.Dk, self.Dk))
        )
        Ekz = phi_k.integrate("x").reshape((T, self.Dk, self.Dz))
        mean_Ekz = jnp.mean(Ekz, axis=0)
        mean_Ezz = jnp.mean(phi.integrate("xx"), axis=0)
        mean_Ekk = jnp.mean(Ekk, axis=0) + 0.0001 * jnp.eye(self.Dk)
        Eff = jnp.block([[mean_Ezz, mean_Ekz.T], [mean_Ekz, mean_Ekk]])
        # Eff = jnp.empty((self.Dphi, self.Dphi))
        # Eff[:self.Dz,:self.Dz] = jnp.mean(phi.integrate('xx'), axis=0)
        # Eff[self.Dz:,self.Dz:] = jnp.mean(Ekk, axis=0)
        # Eff[self.Dz:,:self.Dz] = jnp.mean(Ekz, axis=0)
        # Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        # E[f(z)] b'
        Ez = jnp.mean(phi.integrate("x"), axis=0)
        Ek = jnp.mean(phi_k.integral_light().reshape((T, self.Dk)), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:, None]
        # E[z f(z)']
        # v_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                      self.state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                       self.state_density.k_func.nu])
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta
        )
        Ezz_cross = jnp.mean(
            two_step_smoothing_density.integrate("xx")[:, self.Dz :, : self.Dz], axis=0
        )
        Ezk = jnp.mean(
            two_step_smoothing_density.multiply(joint_k_func, update_full=True)
            .integrate("x")
            .reshape((T, self.Dk, (2 * self.Dz)))[:, :, : self.Dz],
            axis=0,
        ).T
        Ezf = jnp.concatenate([Ezz_cross.T, Ezk], axis=1)
        return jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T

    def update_b(self, smoothing_density: densities.GaussianDensity):
        """ The transition offset is updated here, where the the densities p(z_t|x_{1:T}) is provided.
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        Ez = smoothing_density.integrate("x")
        Ek = (
            smoothing_density.multiply(self.state_density.k_func, update_full=True)
            .integral_light()
            .reshape((T + 1, self.Dk))
        )
        Ef = jnp.concatenate([Ez, Ek], axis=1)
        return jnp.mean(smoothing_density.mu[1:] - jnp.dot(self.A, Ef[:-1].T).T, axis=0)

    def update_Qz(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
    ):
        """ The transition covariance is updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)
        
        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        T = smoothing_density.R - 1
        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:, : self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.mean(
            two_step_smoothing_density.integrate(
                "Ax_aBx_b_outer",
                A_mat=A_tilde.T,
                a_vec=b_tilde,
                B_mat=A_tilde.T,
                b_vec=b_tilde,
            ),
            axis=0,
        )
        # v_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                      self.state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([self.Dk, int(self.Dz)]),
        #                       self.state_density.k_func.nu])
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta
        )
        two_step_k_measure = two_step_smoothing_density.multiply(
            joint_k_func, update_full=True
        )
        Ekz = jnp.mean(
            two_step_k_measure.integrate("x").reshape((T, self.Dk, 2 * self.Dz)), axis=0
        )
        phi_k = smoothing_density.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.mean(phi_k.integral_light().reshape((T + 1, self.Dk))[:-1], axis=0)
        Qz_k_lin_err = jnp.dot(
            self.A[:, self.Dz :],
            (
                Ekz[:, : self.Dz]
                - jnp.dot(self.A[:, : self.Dz], Ekz[:, self.Dz :].T).T
                - Ek[:, None] * self.b[None]
            ),
        )
        Ekk = (
            phi_k.multiply(self.state_density.k_func, update_full=True)
            .integral_light()
            .reshape((T + 1, self.Dk, self.Dk))
        )
        Qz_kk = jnp.dot(
            jnp.dot(self.A[:, self.Dz :], jnp.mean(Ekk[:-1], axis=0)),
            self.A[:, self.Dz :].T,
        )
        return Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T

    @staticmethod
    def _Wfunc(
        W,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
        A,
        b,
        Qz,
        Qz_inv,
        ln_det_Qz,
        Dk,
        Dz,
    ) -> Union[float, jnp.ndarray]:
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
        # W = jnp.reshape(W, (Dk, Dz + 1))
        # print(W.shape)
        state_density = conditionals.LSEMGaussianConditional(
            M=jnp.array([A]),
            b=jnp.array([b]),
            W=W,
            Sigma=jnp.array([Qz]),
            Lambda=jnp.array([Qz_inv]),
            ln_det_Sigma=jnp.array([ln_det_Qz]),
        )
        # self.state_density.update_phi()
        T = smoothing_density.R
        A_lower = A[:, Dz:]
        # E[z f(z)'] A'
        # v_joint = jnp.block([jnp.zeros([Dk, Dz]),
        #                      state_density.k_func.v])
        # nu_joint = jnp.block([jnp.zeros([Dk, Dz]),
        #                       state_density.k_func.nu])
        zero_arr = jnp.zeros([Dk, 2 * Dz])
        v_joint = zero_arr.at[:, Dz:].set(state_density.k_func.v)
        nu_joint = zero_arr.at[:, Dz:].set(state_density.k_func.nu)
        joint_k_func = factors.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=state_density.k_func.ln_beta
        )
        Ekz = jnp.sum(
            jnp.reshape(
                two_step_smoothing_density.multiply(
                    joint_k_func, update_full=True
                ).integrate("x"),
                (T, Dk, 2 * Dz),
            ),
            axis=0,
        )
        phi_k = smoothing_density.multiply(state_density.k_func, update_full=True)
        Ek = jnp.sum(jnp.reshape(phi_k.integral_light(), (T, Dk)), axis=0)
        Qz_k_lin_err = jnp.dot(
            A_lower,
            jnp.subtract(
                jnp.subtract(Ekz[:, :Dz], jnp.dot(Ekz[:, Dz:], A[:, :Dz].T)),
                jnp.outer(Ek, b),
            ),
        )
        Ekk = jnp.sum(
            jnp.reshape(
                phi_k.multiply(state_density.k_func, update_full=True).integral_light(),
                (T, Dk, Dk),
            ),
            axis=0,
        )
        Qz_kk = jnp.dot(jnp.dot(A_lower, Ekk), A_lower.T)
        Qfunc_W = 0.5 * jnp.trace(
            jnp.dot(Qz_inv, jnp.subtract(Qz_kk, jnp.add(Qz_k_lin_err, Qz_k_lin_err.T)))
        )
        return Qfunc_W

    def update_W(
        self,
        smoothing_density: densities.GaussianDensity,
        two_step_smoothing_density: densities.GaussianDensity,
    ):
        """ Updates the weights in the squared exponential of the state conditional mean.

        :param smoothing_density: GaussianDensity
            The smoothing density  p(z_t|x_{1:T}).
        :param two_step_smoothing_density: Gaussian Density
            The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        """
        #  This compiling takes a lot of time, and is only worth it for several iterations
        phi = smoothing_density.slice(jnp.arange(0, smoothing_density.R - 1))
        func = lambda W: self._Wfunc(
            W,
            phi,
            two_step_smoothing_density,
            self.A,
            self.b,
            self.Qz,
            self.Qz_inv,
            self.ln_det_Qz,
            self.Dk,
            self.Dz,
        )
        result = minimize_jax(
            func,
            self.W,
            method="L-BFGS-B",
            bounds=jnp.array([(-1e1, 1e1)] * (self.Dk * (self.Dz + 1))),
            options={"disp": False},
        )
        self.W = result.x

    def update_state_density(self):
        """ Updates the state density.
        """
        self.state_density = conditionals.LSEMGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            W=self.W,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    # TODO: Optimal initial state density
