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
from jax import numpy as jnp
import numpy as np
import objax
from typing import Tuple
from ..utils.jax_minimize_wrapper import ScipyMinimize


from .. import (
    pdf,
    conditional,
    approximate_conditional,
    factor,
)


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
        self, pre_filter_density: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Calculate prediction density.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1.

        :param pre_filter_density: Density p(z_t-1|x_{1:t-1}).
        :type pre_filter_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        :return: Prediction density p(z_t|x_{1:t-1}).
        :rtype: pdf.GaussianPDF
        """
        raise NotImplementedError("Prediction for state model not implemented.")

    def smoothing(
        self,
        cur_filter_density: pdf.GaussianPDF,
        post_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ) -> Tuple[pdf.GaussianPDF, pdf.GaussianPDF]:
        """Calculate smoothing density p(z_{t} | x_{1:T}), 
        given p(z_{t+1} | x_{1:T}) and p(z_{t} | x_{1:t}).
        
        :param cur_filter_density: Density p(z_t|x_{1:t})
        :type cur_filter_density: pdf.GaussianPDF
        :param post_smoothing_density: Density p(z_{t+1}|x_{1:T})
        :type post_smoothing_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        :return: Smoothing density p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T})
        :rtype: Tuple[pdf.GaussianPDF, pdf.GaussianPDF]
        """
        raise NotImplementedError("Smoothing for state model not implemented.")

    def update_hyperparameters(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """The hyperparameters are updated here, where the the densities p(z_t|x_{1:T}) and 
        p(z_{t+1}, z_t|x_{1:T}) are provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        """
        raise NotImplementedError("Hyperparamers for state model not implemented.")

    def update_init_density(
        self, init_smooth_density: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Find the optimal distribution over the initial state z_0, 
        provided with the initial smoothing density.

        :param init_smooth_density: Smoothing density over z_0.
        :type init_smooth_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        :return: The optimal initial distribution.
        :rtype: pdf.GaussianPDF
        """
        raise NotImplementedError(
            "Initial distribution update for state model not implemented."
        )


class LinearStateModel(StateModel):
    def __init__(self, Dz: int, noise_z: float = 1.0):
        """This implements a linear state transition model
        
        z_t = A z_{t-1} + b + zeta_t     with      zeta_t ~ N(0,Qz).

        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param noise_z: Intial isoptropic std. on the state transition, defaults to 1.0
        :type noise_z: float, optional
        """
        self.Dz = Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.A, self.b = jnp.eye(self.Dz), jnp.zeros((self.Dz,))
        self.state_density = conditional.ConditionalGaussianPDF(
            jnp.array([self.A]), jnp.array([self.b]), jnp.array([self.Qz])
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def prediction(
        self, pre_filter_density: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Calculate the prediction density.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1

        :param pre_filter_density: Density p(z_t-1|x_{1:t-1})
        :type pre_filter_density: pdf.GaussianPDF
        :return: Prediction density p(z_t|x_{1:t-1}).
        :rtype: pdf.GaussianPDF
        """
        # p(z_t|x_{1:t-1})
        return self.state_density.affine_marginal_transformation(
            pre_filter_density, **kwargs
        )

    def smoothing(
        self,
        cur_filter_density: pdf.GaussianPDF,
        post_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ) -> Tuple[pdf.GaussianPDF, pdf.GaussianPDF]:
        """ Perform smoothing step.
        
        First we calculate the backward density
        
        $$
        p(z_{t} | z_{t+1}, x_{1:t}) = p(z_{t+1}|z_t)p(z_t | x_{1:t}) / p(z_{t+1}| x_{1:t}) 
        $$
        
        and finally we get the smoothing density
        
        $$
        p(z_{t} | x_{1:T}) = int p(z_{t} | z_{t+1}, x_{1:t}) p(z_{t+1}|x_{1:T}) dz_{t+1}
        $$

        :param cur_filter_density: Density p(z_t|x_{1:t})
        :type cur_filter_density: pdf.GaussianPDF
        :param post_smoothing_density: Density p(z_{t+1}|x_{1:T})
        :type post_smoothing_density: pdf.GaussianPDF
        :return: Smoothing density p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T})
        :rtype: Tuple[pdf.GaussianPDF, pdf.GaussianPDF]
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
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ) -> float:
        """Compute Q-function.
        
        Q = \sum E[log p(z_t|z_{t-1})], 
        
        where the expection is over the smoothing density.
        
        :param smoothing_density: Smoothing density p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        :return: Evaluated Q-function.
        :rtype: float
        """
        return jnp.sum(
            self.state_density.integrate_log_conditional(
                two_step_smoothing_density,
                p_x=smoothing_density.slice(jnp.arange(1, smoothing_density.R)),
            )
        )

    def update_hyperparameters(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """Update hyperparameters. 
        
        The densities p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T}) need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        self.update_A(smoothing_density, two_step_smoothing_density, **kwargs)
        self.update_b(smoothing_density, **kwargs)
        self.update_Qz(smoothing_density, two_step_smoothing_density, **kwargs)
        self.update_state_density()

    def update_A(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """ Update transition matrix.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        # Ezz = smoothing_density.integrate("xx'")
        mu_b = smoothing_density.mu[:-1, None] * self.b[None, :, None]
        Ezz_two_step = two_step_smoothing_density.integrate("xx'")
        Ezz = Ezz_two_step[:, self.Dz :, self.Dz :]
        Ezz_cross = Ezz_two_step[:, self.Dz :, : self.Dz]
        A = jnp.mean(Ezz, axis=0)  # + 1e-2 * jnp.eye(self.Dz)
        self.A = jnp.linalg.solve(A, jnp.mean(Ezz_cross - mu_b, axis=0)).T

    def update_b(self, smoothing_density: pdf.GaussianPDF, **kwargs):
        """Update transition offset.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        """
        self.b = jnp.mean(
            smoothing_density.mu[1:] - jnp.dot(self.A, smoothing_density.mu[:-1].T).T,
            axis=0,
        )

    def update_Qz(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """Update transition covariance.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        A_tilde = jnp.eye(2 * self.Dz, self.Dz)
        A_tilde = A_tilde.at[self.Dz :].set(-self.A.T)
        # A_tilde = jnp.block([[jnp.eye(self.Dz), -self.A.T]])
        b_tilde = -self.b
        self.Qz = jnp.mean(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
                A_mat=A_tilde.T,
                a_vec=b_tilde,
                B_mat=A_tilde.T,
                b_vec=b_tilde,
            ),
            axis=0,
        )

    def update_state_density(self):
        """ Update the state density.
        """
        self.state_density = conditional.ConditionalGaussianPDF(
            jnp.array([self.A]), jnp.array([self.b]), jnp.array([self.Qz])
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def update_init_density(
        self, init_smooth_density: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        """Find the optimal distribution over the initial state z_0, 
        provided with the initial smoothing density.

        :param init_smooth_density: Smoothing density over z_0.
        :type init_smooth_density: pdf.GaussianPDF
        :return: The optimal initial distribution.
        :rtype: pdf.GaussianPDF
        """
        mu0 = init_smooth_density.integrate("x")
        Sigma0 = init_smooth_density.integrate(
            "(Ax+a)(Bx+b)'", A_mat=None, a_vec=-mu0[0], B_mat=None, b_vec=-mu0[0]
        )
        opt_init_density = pdf.GaussianPDF(Sigma0, mu0)
        return opt_init_density

    def condition_on_past(self, z_old: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """ Return p(Z_t+1|Z_t=z)

        :param z_old: Vector of past latent variables.
        :type z_old: jnp.ndarray
        :return: The density of the latent variables at the next step.
        :rtype: pdf.GaussianPDF
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
        """Model with linear state equation
        
        z_t = A(u_t) z_{t-1} + b(u_t) + zeta_t     with      zeta_t ~ N(0,Qz),
        
        where the coefficients are output of a neural network, which gets control variables u as input.

        :param Dz: Dimension of latent space
        :type Dz: int
        :param Du: Dimension of control variables.
        :type Du: int
        :param noise_z: Initial state noise, defaults to 1
        :type noise_z: float, optional
        :param hidden_units: List of number of hidden units in each layer, defaults to [16,]
        :type hidden_units: list, optional
        :param non_linearity: Which non-linearity between layers, defaults to objax.functional.tanh
        :type non_linearity: callable, optional
        :param lr: Learning rate for learning the network, defaults to 1e-4
        :type lr: float, optional
        """
        self.Dz = Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.Du = Du
        self.state_density = conditional.NNControlGaussianConditional(
            Sigma=jnp.array([self.Qz]),
            Dx=self.Dz,
            Du=self.Du,
            hidden_units=hidden_units,
            non_linearity=non_linearity,
        )
        self.lr = lr

    def update_hyperparameters(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        u: jnp.ndarray,
        **kwargs
    ):
        """Update hyperparameters. 
        
        The densities p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T}) need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        """
        self.update_network_params(
            smoothing_density, two_step_smoothing_density, u, **kwargs
        )
        self.update_Qz(smoothing_density, two_step_smoothing_density, u, **kwargs)
        self.update_state_density()

    def update_state_density(self):
        """ Update the state density.
        """
        self.state_density.update_Sigma(jnp.array([self.Qz]))

    def update_Qz(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        u: jnp.ndarray,
        **kwargs
    ):
        """ Update the transition covariance.
        
        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        """
        A_u, b_u = self.state_density.get_M_b(u)
        A_tilde = jnp.empty((two_step_smoothing_density.R, self.Dz, 2 * self.Dz))
        A_tilde = A_tilde.at[:, :, : self.Dz].set(jnp.eye(self.Dz))
        A_tilde = A_tilde.at[:, :, self.Dz :].set(-A_u)
        b_tilde = -b_u
        self.Qz = jnp.mean(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
                A_mat=A_tilde,
                a_vec=b_tilde,
                B_mat=A_tilde,
                b_vec=b_tilde,
            ),
            axis=0,
        )

    def update_network_params(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        u: jnp.ndarray,
        **kwargs
    ):
        """Update the network parameters by gradient descent.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        """
        # TODO: Check for convergence or use other optimizer
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
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        u: jnp.ndarray,
        **kwargs
    ) -> float:
        """Calculate negative Q-function,

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        :return: Negative Q-function
        :rtype: float
        """
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
                "(Ax+a)'(Bx+b)",
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
        """This implements a linear+squared exponential mean (LSEM) state model
        
            z_t = A phi(z_{t-1}) + b + zeta_t     with      zeta_t ~ N(0,Qz).
            
            The feature function is 
            
            phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).
            
            The kernel and linear activation function are given by
            
            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}. 
            
        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param Dk: Number of kernels to use.
        :type Dk: int
        :param noise_z: Initial isoptropic std. on the state transition., defaults to 1.0
        :type noise_z: float, optional
        """
        self.Dz, self.Dk = Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.A = jnp.array(np.random.randn(self.Dz, self.Dphi))
        self.A = self.A.at[:, : self.Dz].set(jnp.eye(self.Dz))
        self.b = jnp.zeros((self.Dz,))
        self.W = objax.TrainVar(jnp.array(np.random.randn(self.Dk, self.Dz + 1)))
        self.state_density = approximate_conditional.LSEMGaussianConditional(
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
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """Update hyperparameters. 
        
        The densities p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T}) need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
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
        self.update_kernel_params(smoothing_density, two_step_smoothing_density)
        # print('W: Run Time %.1f' % (time.perf_counter() - time_start))
        # time_start = time.perf_counter()
        self.update_state_density()
        # print('Update: Run Time %.1f' % (time.perf_counter() - time_start))

    def update_AbQ(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
    ):
        """Update transition matrix, offset and covariance are updated here.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        T = smoothing_density.R - 1
        phi = smoothing_density.slice(jnp.arange(T))

        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:, : self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.mean(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
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
        joint_k_func = factor.OneRankFactor(
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
        phi_kk = phi_k.multiply(self.state_density.k_func, update_full=True)
        mean_Ekk = jnp.mean(
            phi_kk.integral_light().reshape((T, self.Dk, self.Dk)), axis=0,
        )
        Qz_kk = jnp.dot(jnp.dot(self.A[:, self.Dz :], mean_Ekk), self.A[:, self.Dz :].T)
        Qz = Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T
        Qz = 0.5 * (Qz + Qz.T)

        # E[f(z)f(z)']
        Ekk = phi_kk.integral_light().reshape((T, self.Dk, self.Dk))
        Ekz = phi_k.integrate("x").reshape((T, self.Dk, self.Dz))
        mean_Ekz = jnp.mean(Ekz, axis=0)
        mean_Ezz = jnp.mean(phi.integrate("xx'"), axis=0)
        mean_Ekk = jnp.mean(Ekk, axis=0)
        Eff = jnp.block([[mean_Ezz, mean_Ekz.T], [mean_Ekz, mean_Ekk]])
        Eff += 0.001 * jnp.eye(Eff.shape[0])
        # mean_Ekk_reg = mean_Ekk + .0001 * jnp.eye(self.Dk)
        # mean_Ezz = jnp.mean(phi.integrate("xx'"), axis=0)
        # Eff = jnp.block([[mean_Ezz, Ekz_past.T],
        #                  [Ekz_past, mean_Ekk_reg]])
        Ez = jnp.mean(phi.integrate("x"), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:, None]
        # E[z f(z)']
        Ezz_cross = jnp.mean(
            two_step_smoothing_density.integrate("xx'")[:, self.Dz :, : self.Dz], axis=0
        )
        Ezf = jnp.concatenate([Ezz_cross.T, Ekz_future.T], axis=1)
        A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        b = jnp.mean(smoothing_density.mu[1:], axis=0) - jnp.dot(A, Ef).T
        # A = self.A
        # b = self.b
        return A, b, Qz

    def update_state_density(self):
        """ Update the state density.
        """
        self.state_density = approximate_conditional.LSEMGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            W=self.W,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def update_kernel_params(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
    ):
        """Update the kernel weights.
        
        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """

        @objax.Function.with_vars(self.vars())
        def loss():
            self.state_density.w0 = self.W.value[:, 0]
            self.state_density.W = self.W.value[:, 1:]
            self.state_density.update_phi()
            return -self.compute_Q_function(
                smoothing_density, two_step_smoothing_density
            )

        minimizer = ScipyMinimize(
            loss,
            self.vars(),
            method="L-BFGS-B",
            bounds=jnp.array([(-1e1, 1e1)] * (self.Dk * (self.Dz + 1))),
        )
        minimizer.minimize()

    # TODO: Optimal initial state density


class LRBFMStateModel(LinearStateModel):
    def __init__(
        self, Dz: int, Dk: int, noise_z: float = 1.0, kernel_type: bool = "isotropic",
    ):
        """This implements a linear+RBF mean (LRBFM) state model
        
            z_t = A phi(z_{t-1}) + b + zeta_t     with      zeta_t ~ N(0,Qz).
            
            The feature function is 
            
            phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).
            
            The kernel and linear activation function are given by
            
            k(h) = exp(-h^2 / 2) and h_i(x) = (x_i + mu_i) / l_i. 
            
        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param Dk: Number of kernels to use.
        :type Dk: int
        :param noise_z: Initial isoptropic std. on the state transition., defaults to 1.0
        :type noise_z: float, optional
        :param kernel_type: Parameter determining, which kernel is used. 'scalar' same length scale for all kernels and 
            dimensions. 'isotropic' same length scale for dimensions, but different for each kernel. 'anisotropic' 
            different length scale for all kernels and dimensions., defaults to 'isotropic
        :type kernel_type: str
        """
        self.Dz, self.Dk = Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qz = noise_z ** 2 * jnp.eye(self.Dz)
        self.A = jnp.array(np.random.randn(self.Dz, self.Dphi))
        self.A = self.A.at[:, : self.Dz].set(jnp.eye(self.Dz))
        self.b = jnp.zeros((self.Dz,))
        self.mu = objax.TrainVar(objax.random.normal((self.Dk, self.Dz)))
        self.kernel_type = kernel_type
        if self.kernel_type == "scalar":
            self.log_length_scale = objax.TrainVar(objax.random.normal((1, 1),))
        elif self.kernel_type == "isotropic":
            self.log_length_scale = objax.TrainVar(objax.random.normal((self.Dk, 1),))
        elif self.kernel_type == "anisotropic":
            self.log_length_scale = objax.TrainVar(
                objax.random.normal((self.Dk, self.Dz))
            )
        else:
            raise NotImplementedError("Kernel type not implemented.")

        self.state_density = approximate_conditional.LRBFGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            mu=self.mu,
            length_scale=self.length_scale,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    @property
    def length_scale(self):
        if self.kernel_type == "scalar":
            return jnp.tile(jnp.exp(self.log_length_scale), (self.Dk, self.Dz))
        elif self.kernel_type == "isotropic":
            return jnp.tile(jnp.exp(self.log_length_scale), (1, self.Dz))
        elif self.kernel_type == "anisotropic":
            return jnp.exp(self.log_length_scale)

    def update_hyperparameters(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """Update hyperparameters. 
        
        The densities p(z_t|x_{1:T}) and p(z_{t+1}, z_t|x_{1:T}) need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
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
        self.update_kernel_params(smoothing_density, two_step_smoothing_density)
        # print('W: Run Time %.1f' % (time.perf_counter() - time_start))
        # time_start = time.perf_counter()
        self.update_state_density()
        # print('Update: Run Time %.1f' % (time.perf_counter() - time_start))

    def update_AbQ(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
    ):
        """Update transition matrix, offset and covariance are updated here.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T}).
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        T = smoothing_density.R - 1
        phi = smoothing_density.slice(jnp.arange(T))

        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:, : self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.mean(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
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

        Lambda_joint = jnp.zeros((self.Dk, 2 * self.Dz, 2 * self.Dz))
        Lambda_joint = Lambda_joint.at[:, self.Dz :, self.Dz :].set(
            self.state_density.k_func.Lambda
        )
        nu_joint = jnp.zeros([self.Dk, 2 * self.Dz])
        nu_joint = nu_joint.at[:, self.Dz :].set(self.state_density.k_func.nu)
        joint_k_func = factor.ConjugateFactor(
            Lambda=Lambda_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta
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
        phi_kk = phi_k.multiply(self.state_density.k_func, update_full=True)
        mean_Ekk = jnp.mean(
            phi_kk.integral_light().reshape((T, self.Dk, self.Dk)), axis=0,
        )
        Qz_kk = jnp.dot(jnp.dot(self.A[:, self.Dz :], mean_Ekk), self.A[:, self.Dz :].T)
        Qz = Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T
        Qz = 0.5 * (Qz + Qz.T)

        # E[f(z)f(z)']
        Ekk = phi_kk.integral_light().reshape((T, self.Dk, self.Dk))
        Ekz = phi_k.integrate("x").reshape((T, self.Dk, self.Dz))
        mean_Ekz = jnp.mean(Ekz, axis=0)
        mean_Ezz = jnp.mean(phi.integrate("xx'"), axis=0)
        mean_Ekk = jnp.mean(Ekk, axis=0)
        Eff = jnp.block([[mean_Ezz, mean_Ekz.T], [mean_Ekz, mean_Ekk]])
        Eff += 0.001 * jnp.eye(Eff.shape[0])
        # mean_Ekk_reg = mean_Ekk + .0001 * jnp.eye(self.Dk)
        # mean_Ezz = jnp.mean(phi.integrate("xx'"), axis=0)
        # Eff = jnp.block([[mean_Ezz, Ekz_past.T],
        #                  [Ekz_past, mean_Ekk_reg]])
        Ez = jnp.mean(phi.integrate("x"), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:, None]
        # E[z f(z)']
        Ezz_cross = jnp.mean(
            two_step_smoothing_density.integrate("xx'")[:, self.Dz :, : self.Dz], axis=0
        )
        Ezf = jnp.concatenate([Ezz_cross.T, Ekz_future.T], axis=1)
        A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        b = jnp.mean(smoothing_density.mu[1:], axis=0) - jnp.dot(A, Ef).T
        # A = self.A
        # b = self.b
        return A, b, Qz

    def update_state_density(self):
        """ Update the state density.
        """
        self.state_density = approximate_conditional.LRBFGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            mu=self.mu,
            length_scale=self.length_scale,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def update_kernel_params(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
    ):
        """Update the kernel weights.
        
        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  p(z_{t+1}, z_t|x_{1:T}).
        :type two_step_smoothing_density: pdf.GaussianPDF
        """

        @objax.Function.with_vars(self.vars())
        def loss():
            self.state_density.mu = self.mu.value
            self.state_density.length_scale = jnp.exp(self.log_length_scale.value)
            self.state_density.update_phi()
            return -self.compute_Q_function(
                smoothing_density, two_step_smoothing_density
            )

        minimizer = ScipyMinimize(loss, self.vars(), method="L-BFGS-B")
        minimizer.minimize()

    # TODO: Optimal initial state density
