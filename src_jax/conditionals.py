##################################################################################################
# This file is part of the Gaussian Toolbox.                                                     #
#                                                                                                #
# It contains the functionality for conditional Gaussian densities, that can be seen as          #
# operators.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"

# import jnp
# from densities import GaussianDensity
from jax import numpy as jnp
from typing import Tuple
from src_jax import densities, factors, measures
import objax
from utils.linalg import invert_matrix


class ConditionalGaussianDensity:
    def __init__(
        self, M, b=None, Sigma=None, Lambda=None, ln_det_Sigma=None,
    ):
        """ A conditional Gaussian density

            p(y|x) = N(mu(x), Sigma)

            with the conditional mean function mu(x) = M x + b.

        :param M: jnp.ndarray [R, Dy, Dx]
            Matrix in the mean function.
        :param b: jnp.ndarray [R, Dy]
            Vector in the conditional mean function. If None all entries are 0. (Default=None)
        :param Sigma: jnp.ndarray [R, Dy, Dy]
            The covariance matrix of the conditional. (Default=None)
        :param Lambda: jnp.ndarray [R, Dy, Dy] or None
            Information (precision) matrix of the Gaussians. (Default=None)
        :param ln_det_Sigma: jnp.ndarray [R] or None
            Log determinant of the covariance matrix. (Default=None)
        """

        self.R, self.Dy, self.Dx = M.shape

        self.M = M
        if b is None:
            self.b = jnp.zeros((self.R, self.Dy))
        else:
            self.b = b
        if Sigma is None and Lambda is None:
            raise RuntimeError("Either Sigma or Lambda need to be specified.")
        elif Sigma is not None:
            self.Sigma = Sigma
            if Lambda is None or ln_det_Sigma is None:
                self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
            else:
                self.Lambda, self.ln_det_Sigma = Lambda, ln_det_Sigma
            self.ln_det_Lambda = -self.ln_det_Sigma
        else:
            self.Lambda = Lambda
            if Sigma is None or ln_det_Sigma is None:
                self.Sigma, self.ln_det_Lambda = invert_matrix(self.Sigma)
            else:
                self.Sigma, self.ln_det_Lambda = Lambda, ln_det_Sigma
            self.ln_det_Sigma = -self.ln_det_Lambda

    def __str__(self) -> str:
        return "Conditional Gaussian density p(y|x)"

    def __call__(self, x: jnp.ndarray, **kwargs) -> densities.GaussianDensity:
        return self.condition_on_x(x)

    def slice(self, indices: list) -> "ConditionalGaussianDensity":
        """ Returns an object with only the specified entries.

        :param indices: list
            The entries that should be contained in the returned object.

        :return: ConditionalGaussianDensity
            The resulting Gaussian diagonal density.
        """
        M_new = jnp.take(self.M, indices, axis=0)
        b_new = jnp.take(self.b, indices, axis=0)
        Lambda_new = jnp.take(self.Lambda, indices, axis=0)
        Sigma_new = jnp.take(self.Sigma, indices, axis=0)
        ln_det_Sigma_new = jnp.take(self.ln_det_Sigma, indices, axis=0)
        new_measure = ConditionalGaussianDensity(
            M_new, b_new, Sigma_new, Lambda_new, ln_det_Sigma_new
        )
        return new_measure

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """ Computest the conditional mu function

            mu(x) = M x + b.

        :param x: jnp.ndarray [N, Dx]
            Instances, the mu should be conditioned on.

        :return: jnp.ndarray [R, N, Dy]
            Conditional means.
        """
        mu_y = jnp.einsum("abc,dc->adb", self.M, x) + self.b[:, None]
        return mu_y

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> densities.GaussianDensity:
        """ Generates the corresponding Gaussian Density conditioned on x.

        :param x: jnp.ndarray [N, Dx]
            Instances, the mu should be conditioned on.

        :return: GaussianDensity
            The density conditioned on x.
        """

        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((self.R * N, self.Dy))
        Sigma_new = jnp.tile(self.Sigma[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        Lambda_new = jnp.tile(self.Lambda[:, None], (1, N, 1, 1)).reshape(
            self.R * N, self.Dy, self.Dy
        )
        ln_det_Sigma_new = jnp.tile(self.ln_det_Sigma[:, None], (1, N)).reshape(
            self.R * N
        )
        return densities.GaussianDensity(
            Sigma=Sigma_new,
            mu=mu_new,
            Lambda=Lambda_new,
            ln_det_Sigma=ln_det_Sigma_new,
        )

    def set_y(self, y: jnp.ndarray, **kwargs) -> factors.ConjugateFactor:
        """ Sets a specific value for y in p(y|x) and returns the corresponding conjugate factor. 

        :param y: Data for y, where the rth entry is associated with the rth conditional density. 
        :type y: jnp.ndarray [R, Dy]
        :return: The conjugate factor where the first dimension is R.
        :rtype: factors.ConjugateFactor
        """
        y_minus_b = y - self.b
        Lambda_new = jnp.einsum(
            "abc,acd->abd", jnp.einsum("abd, abc -> adc", self.M, self.Lambda), self.M,
        )
        nu_new = jnp.einsum(
            "abc, ab -> ac",
            jnp.einsum("abc, acd -> abd", self.Lambda, self.M),
            y_minus_b,
        )
        yb_Lambda_yb = jnp.einsum(
            "ab, ab-> a",
            jnp.einsum("ab, abc -> ac", y_minus_b, self.Lambda),
            y_minus_b,
        )
        ln_beta_new = -0.5 * (
            yb_Lambda_yb + self.Dx * jnp.log(2 * jnp.pi) + self.ln_det_Sigma
        )
        factor_new = factors.ConjugateFactor(Lambda_new, nu_new, ln_beta_new)
        return factor_new

    def affine_joint_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Returns the joint density 

            p(x,y) = p(y|x)p(x),

            where p(y|x) is the object itself.

        :param p_x: GaussianDensity
            Marginal density over x.

        :return: GaussianDensity
            The joint density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditionals is not implemented."
            )
        R = p_x.R * self.R
        D_xy = p_x.D + self.Dy
        # Mean
        mu_x = jnp.tile(p_x.mu[None], (self.R, 1, 1,)).reshape((R, p_x.D))
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        mu_xy = jnp.hstack([mu_x, mu_y])
        # Sigma
        Sigma_x = jnp.tile(p_x.Sigma[None], (self.R, 1, 1, 1)).reshape(R, p_x.D, p_x.D)
        MSigma_x = jnp.einsum("abc,dce->adbe", self.M, p_x.Sigma)  # [R1,R,Dy,D]
        MSigmaM = jnp.einsum("abcd,aed->abce", MSigma_x, self.M)
        Sigma_y = (self.Sigma[:, None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        C_xy = MSigma_x.reshape((R, self.Dy, p_x.D))
        Sigma_xy = jnp.block([[Sigma_x, jnp.swapaxes(C_xy, 1, 2)], [C_xy, Sigma_y]])
        # Sigma_xy = jnp.empty((R, D_xy, D_xy))
        # Sigma_xy[:,:p_x.D,:p_x.D] = Sigma_x
        # Sigma_xy[:,p_x.D:,p_x.D:] = Sigma_y
        # Sigma_xy[:,p_x.D:,:p_x.D] = C_xy
        # Sigma_xy[:,:p_x.D,p_x.D:] = jnp.swapaxes(C_xy, 1, 2)
        # Lambda
        Lambda_y = jnp.tile(self.Lambda[:, None], (1, p_x.R, 1, 1)).reshape(
            (R, self.Dy, self.Dy)
        )
        Lambda_yM = jnp.einsum("abc,abd->acd", self.Lambda, self.M)  # [R1,Dy,D]
        MLambdaM = jnp.einsum("abc,abd->acd", self.M, Lambda_yM)
        Lambda_x = (p_x.Lambda[None] + MLambdaM[:, None]).reshape((R, p_x.D, p_x.D))
        L_xy = jnp.tile(-Lambda_yM[:, None], (1, p_x.R, 1, 1)).reshape(
            (R, self.Dy, p_x.D)
        )
        Lambda_xy = jnp.block([[Lambda_x, jnp.swapaxes(L_xy, 1, 2)], [L_xy, Lambda_y]])
        # Lambda_xy = jnp.empty((R, D_xy, D_xy))
        # Lambda_xy[:,:p_x.D,:p_x.D] = Lambda_x
        # Lambda_xy[:,p_x.D:,p_x.D:] = Lambda_y
        # Lambda_xy[:,p_x.D:,:p_x.D] = L_xy
        # Lambda_xy[:,:p_x.D,p_x.D:] = jnp.swapaxes(L_xy, 1, 2)
        # Log determinant
        if p_x.D > self.Dy:
            CLambda_x = jnp.einsum(
                "abcd,bde->abce", MSigma_x, p_x.Lambda
            )  # [R1,R,Dy,D]
            CLambdaC = jnp.einsum(
                "abcd,abed->abce", CLambda_x, MSigma_x
            )  # [R1,R,Dy,Dy]
            delta_ln_det = jnp.linalg.slogdet(Sigma_y[:, None] - CLambdaC)[1].reshape(
                (R,)
            )
            ln_det_Sigma_xy = p_x.ln_det_Sigma + delta_ln_det
        else:
            # [R1,Dy,Dy] x [R1, Dy, D] = [R1, Dy, D]
            Sigma_yL = jnp.einsum("abc,acd->abd", self.Sigma, -Lambda_yM)
            # [R1, Dy, D] x [R1, Dy, D] = [R1, D, D]
            LSigmaL = jnp.einsum("abc,abd->acd", -Lambda_yM, Sigma_yL)
            LSigmaL = jnp.tile(LSigmaL[:, None], (1, p_x.R)).reshape((R, p_x.D, p_x.D))
            delta_ln_det = jnp.linalg.slogdet(Lambda_x - LSigmaL)[1]
            ln_det_Sigma_xy = -(
                jnp.tile(self.ln_det_Lambda[:, None], (1, p_x.R)).reshape((R,))
                + delta_ln_det
            )
        return densities.GaussianDensity(Sigma_xy, mu_xy, Lambda_xy, ln_det_Sigma_xy)

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Returns the marginal density p(y) given  p(y|x) and p(x), 
            where p(y|x) is the object itself.

        :param p_x: GaussianDensity
            Marginal density over x.

        :return: GaussianDensity
            The marginal density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditionals is not implemented."
            )
        R = p_x.R * self.R
        # Mean
        mu_y = self.get_conditional_mu(p_x.mu).reshape((R, self.Dy))
        # Sigma
        MSigma_x = jnp.einsum("abc,dce->adbe", self.M, p_x.Sigma)  # [R1,R,Dy,D]
        MSigmaM = jnp.einsum("abcd,aed->abce", MSigma_x, self.M)
        Sigma_y = (self.Sigma[:, None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        return densities.GaussianDensity(Sigma_y, mu_y)

    def affine_conditional_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> "ConditionalGaussianDensity":
        """ Returns the conditional density p(x|y), given p(y|x) and p(x),           
            where p(y|x) is the object itself.

        :param p_x: GaussianDensity
            Marginal density over x.

        :return: GaussianDensity
            The marginal density.
        """
        # At the moment, I am not sure whether it makes sense to consider the case, where you have a combination of
        # multiple marginals and multiple cond
        try:
            assert p_x.R == 1 or self.R == 1
        except AssertionError:
            raise RuntimeError(
                "The combination of combining multiple marginals with multiple conditionals is not implemented."
            )
        R = p_x.R * self.R
        # TODO: Could be flexibly made more effiecient here.
        # Marginal Sigma y
        # MSigma_x = jnp.einsum('abc,dce->adbe', self.M, p_xSigma) # [R1,R,Dy,D]
        # MSigmaM = jnp.einsum('abcd,aed->abce', MSigma_x, self.M)
        # Sigma_y = (self.Sigma[:,None] + MSigmaM).reshape((R, self.Dy, self.Dy))
        # Lambda_y, ln_det_Sigma_y = p_x.invert_matrix(Sigma_y)
        # Lambda
        Lambda_yM = jnp.einsum("abc,abd->acd", self.Lambda, self.M)  # [R1,Dy,D]
        MLambdaM = jnp.einsum("abc,abd->acd", self.M, Lambda_yM)
        Lambda_x = (p_x.Lambda[None] + MLambdaM[:, None]).reshape((R, p_x.D, p_x.D))
        # Sigma
        Sigma_x, ln_det_Lambda_x = invert_matrix(Lambda_x)
        # M_x
        M_Lambda_y = jnp.einsum("abc,abd->acd", self.M, self.Lambda)  # [R1, D, Dy]
        M_x = jnp.einsum(
            "abcd,ade->abce", Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), M_Lambda_y
        )  # [R1, R, D, Dy]
        # [R1, R, D, Dy] x [R1, Dy] = [R1, R, D]
        b_x = -jnp.einsum("abcd,ad->abc", M_x, self.b)
        b_x += jnp.einsum(
            "abcd,bd->abc", Sigma_x.reshape((self.R, p_x.R, p_x.D, p_x.D)), p_x.nu
        )
        b_x = b_x.reshape((R, p_x.D))
        M_x = M_x.reshape((R, p_x.D, self.Dy))
        return ConditionalGaussianDensity(
            M_x, b_x, Sigma_x, Lambda_x, -ln_det_Lambda_x,
        )

    def integrate_log_conditional(
        self, phi_yx: measures.GaussianMeasure, **kwargs
    ) -> jnp.ndarray:
        """Integrates over the log conditional with respect to the pdf p_yx. I.e.
        
        int log(p(y|x))p(y,x)dydx.

        :param p_yx: Probability density function (first dimensions are y, last ones are x).
        :type p_yx: measures.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density p_yx.
        :rtype: jnp.ndarray
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")
        int_phi = phi_yx.integrate()
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(-self.M)
        b = -self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = phi_yx.integrate(
            "Ax_aBx_b_inner", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        log_expectation = -0.5 * (
            quadratic_integral
            + int_phi * (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        return log_expectation

    def integrate_log_conditional_y(
        self, phi_x: measures.GaussianMeasure, **kwargs
    ) -> callable:
        """Computes the expectation over the log conditional, but just over x. I.e. it returns

           f(y) = int log(p(y|x))p(x)dx.
        
        :param p_x: Density over x.
        :type p_x: measures.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of y.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        int_phi = phi_x.integrate()
        A = self.M
        b = self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = phi_x.integrate(
            "Ax_aBx_b_inner", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        linear_integral = phi_x.integrate("Ax_a", A_mat=A_tilde, a_vec=b_tilde)
        log_expectation_constant = -0.5 * (
            quadratic_integral
            + int_phi * (self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi))
        )
        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            * int_phi
            + jnp.einsum("ab,ab->a", y, linear_integral)
            + log_expectation_constant
        )
        return log_expectation_y

    def conditional_entropy(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> jnp.ndarray:
        """Computes the conditional entropy
        
         H(y|x) = H(y,x) - H(x) = -\int p(x,y)\ln p(y|x) dx dy

        :param p_x: Marginal over condtional variable
        :type p_x: densities.GaussianDensity
        :return: Conditional entropy 
        :rtype: jnp.ndarray [R]
        """
        p_xy = self.affine_joint_transformation(p_x)
        cond_entropy = p_xy.entropy() - p_x.entropy()
        return cond_entropy

    def mutual_information(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> jnp.ndarray:
        """Computes the mutual information
        
         I(y,x) = H(y,x) - H(x) - H(y) 

        :param p_x: Marginal over condtional variable
        :type p_x: densities.GaussianDensity
        :return: Mututal information
        :rtype: jnp.ndarray [R]
        """
        cond_entropy = self.conditional_entropy(p_x, **kwargs)
        p_y = self.affine_marginal_transformation(p_x, **kwargs)
        mutual_info = cond_entropy - p_y.entropy()
        return mutual_info

    def update_Sigma(self, Sigma_new: jnp.ndarray):
        """Updates the covariance matrix.

        :param Sigma_new: The new covariance matrix.
        :type Sigma_new: jnp.ndarray [R, Dy, Dy]
        :raises ValueError: Raised when dimension of old and new covariance do not match.
        """
        if self.Sigma.shape != Sigma_new.shape:
            raise ValueError("Dimensions of the new Sigma don't match.")
        self.Sigma = Sigma_new
        self.Lambda, self.ln_det_Sigma = invert_matrix(Sigma_new)
        self.ln_det_Lambda = -self.ln_det_Sigma


class NNControlGaussianConditional(objax.Module, ConditionalGaussianDensity):
    def __init__(
        self,
        Sigma: jnp.ndarray,
        Dx: int,
        Du: int,
        hidden_units: list = [16,],
        non_linearity: callable = objax.functional.tanh,
    ):
        """A conditional Gaussian density, where the transition model is determined through a (known) control variable u.
        
            p(y|x, u) = N(mu(x|u), Sigma)

            with the conditional mean function mu(x) = M(u) x + b(u),
            
            where M(u) and b(u) come from the same neural network.

        :param Sigma: Covariance matrix [1, Dy, Dy]
        :type Sigma: jnp.ndarray
        :param Dx: Dimension of the conditional variable.
        :type Dx: int
        :param Du: Dimension of the control variable
        :type Du: int
        :param hidden_units: Determines how many hidden layers and how many units in each layer, defaults to [16,]
        :type hidden_units: list, optional
        :param non_linearity: Non linearity after each layer, defaults to objax.functional.tanh
        :type non_linearity: callable, optional
        :raises NotImplementedError: Raised when the leading dimension of Sigma is not 1.
        """
        self.Sigma = Sigma
        self.R = Sigma.shape[0]
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        self.Lambda, self.ln_det_Sigma = invert_matrix(self.Sigma)
        self.Dy, self.Dx, self.Du = self.Sigma.shape[1], Dx, Du
        self.hidden_units = hidden_units
        self.non_linearity = non_linearity
        self.network = self._build_network()

    def _build_network(self) -> objax.Module:
        """Constructs the network

        :return: The network.
        :rtype: objax.Module
        """
        nn_list = []
        prev_layer = self.Du
        for num_hidden in self.hidden_units:
            nn_list += [objax.nn.Linear(prev_layer, num_hidden), self.non_linearity]
            prev_layer = num_hidden
        nn_list += [objax.nn.Linear(prev_layer, self.Dy * (self.Dx + 1))]
        network = objax.nn.Sequential(nn_list)
        return network

    def get_M_b(self, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Construct M(u) and b(u) from the output.

        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Returns M(u) [R, Dy, Dx] and b(u) [R, Dy]
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        output = self.network(u)
        M = output[:, : self.Dy * self.Dx].reshape((-1, self.Dy, self.Dx))
        b = output[:, self.Dy * self.Dx :]
        return M, b

    def set_control_variable(self, u: jnp.ndarray) -> ConditionalGaussianDensity:
        """Creates the conditional for a given control variable u,
        
            p(Y|X, U=u).

        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: The conditional
        :rtype: ConditionalGaussianDensity
        """
        R = u.shape[0]
        M, b = self.get_M_b(u)
        tile_dims = (R, 1, 1)
        return ConditionalGaussianDensity(
            M=M,
            b=b,
            Sigma=jnp.tile(self.Sigma, tile_dims),
            Lambda=jnp.tile(self.Lambda, tile_dims),
            ln_det_Sigma=jnp.tile(self.ln_det_Sigma, (R,)),
        )

    def get_conditional_mu(self, x: jnp.ndarray, u: jnp.array, **kwargs) -> jnp.ndarray:
        """ Computes the conditional mean given an x and an u,
        
        mu(x|u) = M(u)x + b(u)

        :param x: Conditional variable [N, Dx]
        :type x: jnp.ndarray
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Conditional mean [R, N, Dy]
        :rtype: jnp.ndarray
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.get_conditional_mu(x)

    def condition_on_x(
        self, x: jnp.ndarray, u: jnp.array, **kwargs
    ) -> densities.GaussianDensity:
        """Returns the Gaussian density
        
        p(Y|X=x, U=u)

        :param x: Conditional variable [N, Dx]
        :type x: jnp.ndarray
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Gaussian density conditioned on instances x, and u.
        :rtype: densities.GaussianDensity
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.condition_on_x(x)

    def set_y(self, y: jnp.ndarray, u: jnp.array, **kwargs) -> factors.ConjugateFactor:
        """Sets an instance of Y and U and returns
        
        p(Y=y|X, U=u)

        :param y: Random variable [R, Dy]
        :type y: jnp.ndarray
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: The factor with the instantiation.
        :rtype: factors.ConjugateFactor
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.set_y(y)

    def affine_joint_transformation(
        self, p_x: densities.GaussianDensity, u: jnp.array, **kwargs
    ) -> densities.GaussianDensity:
        """Does the affine joint transformation with a given control variable
        
        
            p(X,Y|U=u) = p(Y|X,U=u)p(X),

            where p(Y|X,U=u) is the object itself.

        :param p_x: Marginal over X
        :type p_x: densities.GaussianDensity
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: The joint density
        :rtype: densities.GaussianDensity
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.affine_joint_transformation(p_x)

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity, u: jnp.array, **kwargs
    ) -> densities.GaussianDensity:
        """Returns the marginal density p(Y) given  p(Y|X,U=u) and p(X), 
        where p(Y|X,U=u) is the object itself.

        :param p_x: Marginal over X
        :type p_x: densities.GaussianDensity
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Marginal density p(Y|U=u)
        :rtype: densities.GaussianDensity
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.affine_marginal_transformation(p_x)

    def affine_conditional_transformation(
        self, p_x: densities.GaussianDensity, u: jnp.array, **kwargs
    ) -> "ConditionalGaussianDensity":
        """ Returns the conditional density p(X|Y, U=u), given p(Y|X,U=u) and p(X),           
            where p(Y|X,U=u) is the object itself.

        :param p_x: Marginal over X
        :type p_x: densities.GaussianDensity
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Conditional density p(X|Y, U=u)
        :rtype: ConditionalGaussianDensity
        """
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.affine_conditional_transformation(p_x)

    def conditional_entropy(
        self, p_x: densities.GaussianDensity, u: jnp.array, **kwargs
    ) -> jnp.ndarray:
        """Computes the conditional entropy
        
         H(y|x) = H(y,x) - H(x) = -\int p(x,y)\ln p(y|x) dx dy

        :param p_x: Marginal over condtional variable
        :type p_x: densities.GaussianDensity
        :param u: Control variables [R, Du]
        :type u: jnp.ndarray
        :return: Conditional entropy 
        :rtype: jnp.ndarray [R]
        """
        p_xy = self.affine_joint_transformation(p_x, u)
        cond_entropy = p_xy.entropy() - p_x.entropy()
        return cond_entropy

    def integrate_log_conditional(
        self, phi_yx: measures.GaussianMeasure, u: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Integrates over the log conditional with respect to the pdf p_yx. I.e.
        
        int log(p(y|x))p(y,x)dydx.

        :param p_yx: Probability density function (first dimensions are y, last ones are x).
        :type p_yx: measures.GaussianMeasure
        :param u: Control variables [1, Du]
        :type u: jnp.ndarray
        :raises NotImplementedError: Only one network input allowed.
        :return: Returns the integral with respect to density p_yx.
        :rtype: jnp.ndarray
        """
        if u.shape[0] != 1:
            raise NotImplementedError("Only implemented for a single input.")
        cond_gauss = self.set_control_variable(u)
        return cond_gauss.integrate_log_conditional(phi_yx)

    def integrate_log_conditional_y(
        self, phi_x: measures.GaussianMeasure, u: jnp.ndarray, **kwargs
    ) -> callable:
        """Computes the expectation over the log conditional, but just over x. I.e. it returns

           f(y) = int log(p(y|x))p(x)dx.
        
        :param p_x: Density over x.
        :type p_x: measures.GaussianMeasure
        :param u: Control variables [1, Du]
        :type u: jnp.ndarray
        :raises NotImplementedError: Only one network input allowed.
        :return: The integral as function of y.
        :rtype: callable
        """
        if u.shape[0] != 1:
            raise NotImplementedError("Only implemented for a single input.")

        cond_gauss = self.set_control_variable(u)
        return cond_gauss.integrate_log_conditional_y(phi_x)


class LSEMGaussianConditional(ConditionalGaussianDensity):
    def __init__(
        self,
        M: jnp.ndarray,
        b: jnp.ndarray,
        W: jnp.ndarray,
        Sigma: jnp.ndarray = None,
        Lambda: jnp.ndarray = None,
        ln_det_Sigma: jnp.ndarray = None,
    ):
        """ A conditional Gaussian density, with a linear squared exponential mean (LSEM) function,

            p(y|x) = N(mu(x), Sigma)

            with the conditional mean function mu(x) = M phi(x) + b. 
            phi(x) is a feature vector of the form

            phi(x) = (1,x_1,...,x_m,k(h_1(x)),...,k(h_n(x))),

            with

            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.

            Note, that the affine transformations will be approximated via moment matching.

            :param M: jnp.ndarray [1, Dy, Dphi]
                Matrix in the mean function.
            :param b: jnp.ndarray [1, Dy]
                Vector in the conditional mean function.
            :param W: jnp.ndarray [Dk, Dx + 1]
                Parameters for linear mapping in the nonlinear functions
            :param Sigma: jnp.ndarray [1, Dy, Dy]
                The covariance matrix of the conditional. (Default=None)
            :param Lambda: jnp.ndarray [1, Dy, Dy] or None
                Information (precision) matrix of the Gaussians. (Default=None)
            :param ln_det_Sigma: jnp.ndarray [1] or None
                Log determinant of the covariance matrix. (Default=None)
        """
        super().__init__(M, b, Sigma, Lambda, ln_det_Sigma)
        self.w0 = W[:, 0]
        self.W = W[:, 1:]
        self.Dx = self.W.shape[1]
        self.Dk = self.W.shape[0]
        self.Dphi = self.Dk + self.Dx
        self.update_phi()

    def update_phi(self):
        """ Sets up the non-linear kernel function in phi(x).
        """
        v = self.W
        nu = self.W * self.w0[:, None]
        ln_beta = -0.5 * self.w0 ** 2
        self.k_func = factors.OneRankFactor(v=v, nu=nu, ln_beta=ln_beta)

    def evaluate_phi(self, x: jnp.ndarray):
        """ Evaluates the phi

        phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).

        :param x: jnp.ndarray [N, Dx]
            Points where f should be evaluated.

        :return: jnp.ndarray [N, Dphi]
            Deature vector.
        """
        N = x.shape[0]
        # phi_x = jnp.empty((N, self.Dphi))
        phi_x = jnp.block([x, self.k_func.evaluate(x).T])
        # phi_x[:,self.Dx:] = self.k_func.evaluate(x).T
        return phi_x

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """ Computes the conditional mu function

            mu(x) = mu(x) = M phi(x) + b

        :param x: jnp.ndarray [N, Dx]
            Instances, the mu should be conditioned on.

        :return: jnp.ndarray [1, N, Dy]
            Conditional means.
        """
        phi_x = self.evaluate_phi(x)
        mu_y = jnp.einsum("ab,cb->ca", self.M[0], phi_x) + self.b[0][None]
        return mu_y

    def set_y(self, y: jnp.ndarray, **kwargs):
        """Not valid function for this model class.

        :param y: Data for y, where the rth entry is associated with the rth conditional density. 
        :type y: jnp.ndarray [R, Dy]
        :raises AttributeError: Raised because doesn't p(y|x) is not a ConjugateFactor for x. 
        """
        raise AttributeError("LSEMGaussianConditional doesn't have attributee set_y.")

    def get_expected_moments(self, p_x: densities.GaussianDensity) -> jnp.ndarray:
        """ Computes the expected covariance

            Sigma_y = E[yy'] - E[y]E[y]'

        :param p_x: GaussianDensity
            The density which we average over.

        :return: jnp.ndarray [p_R, Dy, Dy]
            Returns the expected mean
        """

        #### E[f(x)] ####
        # E[x] [R, Dx]
        Ex = p_x.integrate("x")
        # E[k(x)] [R, Dphi - Dx]
        p_k = p_x.multiply(self.k_func, update_full=True)
        Ekx = p_k.integrate().reshape((p_x.R, self.Dphi - self.Dx))
        # E[f(x)]
        Ef = jnp.concatenate([Ex, Ekx], axis=1)

        #### E[f(x)f(x)'] ####
        # Eff = jnp.empty([p_x.R, self.Dphi, self.Dphi])
        # Linear terms E[xx']
        Exx = p_x.integrate("xx")
        # Eff[:,:self.Dx,:self.Dx] =
        # Cross terms E[x k(x)']
        Ekx = p_k.integrate("x").reshape((p_x.R, self.Dk, self.Dx))
        # Eff[:,:self.Dx,self.Dx:] = jnp.swapaxes(Ekx, axis1=1, axis2=2)
        # Eff[:,self.Dx:,:self.Dx] = Ekx
        # kernel terms E[k(x)k(x)']
        Ekk = (
            p_x.multiply(self.k_func, update_full=True)
            .multiply(self.k_func, update_full=True)
            .integrate()
            .reshape((p_x.R, self.Dk, self.Dk))
        )
        # Eff[:,self.Dx:,self.Dx:] = Ekk
        Eff = jnp.block([[Exx, jnp.swapaxes(Ekx, axis1=1, axis2=2)], [Ekx, Ekk]])

        ### mu_y = E[mu(x)] = ME[f(x)] + b ###
        mu_y = jnp.einsum("ab,cb->ca", self.M[0], Ef) + self.b[0][None]

        # Sigma_y = E[yy'] - mu_ymu_y' = Sigma + E[mu(x)mu(x)'] - mu_ymu_y'
        #                                = Sigma + ME[f(x)f(x)']M' + bE[f(x)']M' + ME[f(x)]b' + bb' - mu_ymu_y'
        Sigma_y = jnp.tile(self.Sigma, (p_x.R, 1, 1))
        Sigma_y += jnp.einsum(
            "ab,cbd->cad", self.M[0], jnp.einsum("abc,dc->abd", Eff, self.M[0])
        )
        MEfb = jnp.einsum(
            "ab,c->abc", jnp.einsum("ab,cb->ca", self.M[0], Ef), self.b[0]
        )
        Sigma_y += MEfb + jnp.swapaxes(MEfb, axis1=1, axis2=2)
        Sigma_y += (self.b[0, None] * self.b[0, :, None])[None]
        Sigma_y -= mu_y[:, None] * mu_y[:, :, None]
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: densities.GaussianDensity) -> jnp.ndarray:
        """ Computes

            E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M f(x) + b)x' p(x) dx

        :param p_x: GaussianDensity
            The density which we average over.

        :return: jnp.ndarray [p_R, Dx, Dy]
            Returns the cross expectations.
        """

        # E[xx']
        Exx = p_x.integrate("xx")
        # E[k(x)x']
        Ekx = (
            p_x.multiply(self.k_func, update_full=True)
            .integrate("x")
            .reshape((p_x.R, self.Dk, self.Dx))
        )
        # E[f(x)x']
        Ef_x = jnp.concatenate([Exx, Ekx], axis=1)
        # M E[f(x)x']
        MEf_x = jnp.einsum("ab,cbd->cad", self.M[0], Ef_x)
        # bE[x']
        bEx = self.b[0][None, :, None] * p_x.integrate("x")[:, None]
        # E[yx']
        Eyx = MEf_x + bEx
        return Eyx

    def affine_joint_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Gets an approximation of the joint density

            p(x,y) ~= N(mu_{xy},Sigma_{xy}),

        The mean is given by

            mu_{xy} = (mu_x, mu_y)'

        with mu_y = E[mu_y(x)]. The covariance is given by

            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        mu_xy = jnp.concatenate([mu_x, mu_y], axis=1)
        Sigma_xy = jnp.block(
            [[p_x.Sigma, cov_yx], [jnp.swapaxes(cov_yx, axis1=1, axis2=2), Sigma_y]]
        )
        # Sigma_xy = jnp.empty((p_x.R, self.Dy + self.Dx, self.Dy + self.Dx))
        # Sigma_xy[:,:self.Dx,:self.Dx] = p_x.Sigma
        # Sigma_xy[:,self.Dx:,:self.Dx] = cov_yx
        # Sigma_xy[:,:self.Dx,self.Dx:] = jnp.swapaxes(cov_yx, axis1=1, axis2=2)
        # Sigma_xy[:,self.Dx:,self.Dx:] = Sigma_y
        p_xy = densities.GaussianDensity(Sigma=Sigma_xy, mu=mu_xy)
        return p_xy

    def affine_conditional_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> "ConditionalGaussianDensity":
        """ Gets an approximation of the joint density via moment matching

            p(x|y) ~= N(mu_{x|y},Sigma_{x|y}),

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: ConditionalDensity
            Returns the conditional density of x given y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = ConditionalGaussianDensity(M=M_new, b=b_new, Sigma=Sigma_new,)
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Gets an approximation of the marginal density

            p(y) ~= N(mu_y,Sigma_y),

        The mean is given by

            mu_y = E[mu_y(x)]. 

        The covariance is given by

            Sigma_y = E[yy'] - mu_ymu_y'.

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = densities.GaussianDensity(Sigma=Sigma_y, mu=mu_y,)
        return p_y

    def integrate_log_conditional(
        self,
        p_yx: densities.GaussianDensity,
        p_x: densities.GaussianDensity = None,
        **kwargs
    ) -> jnp.ndarray:
        """Integrates over the log conditional with respect to the pdf p_yx. I.e.
        
        int log(p(y|x))p(y,x)dydx.

        :param p_yx: Probability density function (first dimensions are y, last ones are x).
        :type p_yx: measures.GaussianMeasure
        :raises NotImplementedError: Only implemented for R=1.
        :return: Returns the integral with respect to density p_yx.
        :rtype: jnp.ndarray
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        # E[(y - Mx - b)' Lambda (y - Mx - b)]
        A = jnp.empty((self.R, self.Dy, self.Dy + self.Dx))
        A = A.at[:, :, : self.Dy].set(jnp.eye(self.Dy, self.Dy)[None])
        A = A.at[:, :, self.Dy :].set(-self.M[:, :, : self.Dx])
        b = -self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)
        quadratic_integral = p_yx.integrate(
            "Ax_aBx_b_inner", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        # E[(y - Mx - b) Lambda Mk phi(x)]
        zero_arr = jnp.zeros([self.Dk, self.Dy + self.Dx])
        v_joint = zero_arr.at[:, self.Dy :].set(self.k_func.v)
        nu_joint = zero_arr.at[:, self.Dy :].set(self.k_func.nu)
        joint_k_func = factors.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=self.k_func.ln_beta
        )
        p_yx_k = p_yx.multiply(joint_k_func, update_full=True)
        E_k_lin_term = jnp.reshape(
            p_yx_k.integrate("Ax_a", A_mat=A_tilde, a_vec=b_tilde),
            (p_yx.R, self.Dk, self.Dy),
        )
        Mk = self.M[:, :, self.Dx :]
        lin_kernel_integral = jnp.einsum("abc,acb->a", Mk, E_k_lin_term)

        # E[phi(x)' Mk'  Lambda Mk phi(x)]
        if p_x is None:
            p_x = p_yx.get_marginal(jnp.arange(self.Dy, self.Dy + self.Dx))
        p_x_kk = p_x.multiply(self.k_func, update_full=True).multiply(
            self.k_func, update_full=True
        )
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant = self.ln_det_Sigma + self.Dy * jnp.log(2.0 * jnp.pi)
        log_expectation = -0.5 * (
            quadratic_integral
            - 2 * lin_kernel_integral
            + kernel_kernel_integral
            + constant
        )
        return log_expectation


class HCCovGaussianConditional(ConditionalGaussianDensity):
    def __init__(
        self,
        M: jnp.ndarray,
        b: jnp.ndarray,
        sigma_x: jnp.ndarray,
        U: jnp.ndarray,
        W: jnp.ndarray,
        beta: jnp.ndarray,
    ):
        """ A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

            p(y|x) = N(mu(x), Sigma(x))

            with the conditional mean function mu(x) = M x + b. 
            The covariance matrix has the form

            Sigma_y(x) = sigma_x^2 I + \sum_i U_i D_i(x) U_i',

            and D_i(x) = 2 * beta_i * cosh(h_i(x)) and h_i(x) = w_i'x + b_i

            Note, that the affine transformations will be approximated via moment matching.

            :param M: jnp.ndarray [1, Dy, Dx]
                Matrix in the mean function.
            :param b: jnp.ndarray [1, Dy]
                Vector in the conditional mean function.
            :param W: jnp.ndarray [Du, Dx + 1]
                Parameters for linear mapping in the nonlinear functions
            :param sigma_x: float
                Diagonal noise parameter.
            :param U: jnp.ndarray [Dy, Du]
                Othonormal vectors for low rank noise part.
            :param W: jnp.ndarray [Du, Dx + 1]
                Noise weights for low rank components (w_i & b_i).
            :param beta: jnp.ndarray [Du]
                Scaling for low rank noise components.
        """
        self.R, self.Dy, self.Dx = M.shape
        if self.R != 1:
            raise NotImplementedError("So far only R=1 is supported.")
        self.Du = beta.shape[0]
        self.M = M
        self.b = b
        self.U = U
        self.W = W
        self.beta = beta
        self.sigma2_x = sigma_x ** 2
        self._setup_noise_diagonal_functions()

    def _setup_noise_diagonal_functions(self):
        """ Creates the functions, that later need to be integrated over, i.e.

        exp(h_i(z)) and exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factors.LinearFactor(nu, ln_beta)
        self.exp_h_minus = factors.LinearFactor(-nu, -ln_beta)

    def get_conditional_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the covariance at a given x, i.e.

        Sigma_y(x) = sigma_x^2 I + \sum_i U_i D_i(x) U_i',

        with D_i(x) = 2 * beta_i * cosh(h_i(x)) and h_i(x) = w_i'x + b_i.

        :param x: jnp.ndarray [N, Dx]
            Instances, the mu should be conditioned on.

        :return: jnp.ndarray [N, Dy, Dy]
            Conditional covariance.
        """
        D_x = self.beta[None, :, None] * (self.exp_h_plus(x) + self.exp_h_minus(x))
        Sigma_0 = self.sigma2_x * jnp.eye(self.Dy)
        Sigma_y_x = Sigma_0[None] + jnp.einsum(
            "ab,cb->ac", jnp.einsum("ab,cb->ca", self.U, D_x), self.U
        )
        return Sigma_y_x

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> densities.GaussianDensity:
        """ Generates the corresponding Gaussian Density conditioned on x.

        :param x: jnp.ndarray [N, Dx]
            Instances, the mu should be conditioned on.

        :return: GaussianDensity
            The density conditioned on x.
        """
        N = x.shape[0]
        mu_new = self.get_conditional_mu(x).reshape((N, self.Dy))
        Sigma_new = self.get_conditional_cov(x)
        return densities.GaussianDensity(Sigma=Sigma_new, mu=mu_new)

    def set_y(self, y: jnp.ndarray, **kwargs):
        """Not valid function for this model class.

        :param y: Data for y, where the rth entry is associated with the rth conditional density. 
        :type y: jnp.ndarray [R, Dy]
        :raises AttributeError: Raised because doesn't p(y|x) is not a ConjugateFactor for x. 
        """
        raise AttributeError("HCCovGaussianConditional doesn't have attributee set_y.")

    def integrate_Sigma_x(self, p_x: densities.GaussianDensity) -> jnp.ndarray:
        """ Returns the integral

        int Sigma_y(x)p(x) dx.

        :param p_x: GaussianDensity
            The density the covatiance is integrated with.

        :return: jnp.ndarray [Dy, Dy]
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = (
            p_x.multiply(self.exp_h_plus).integrate()
            + p_x.multiply(self.exp_h_minus).integrate()
        )
        D_int = self.beta[None] * D_int.reshape((p_x.R, self.Du))
        return self.sigma2_x * jnp.eye(self.Dy)[None] + jnp.einsum(
            "abc,dc->abd", self.U[None] * D_int[:, None], self.U
        )

    def get_expected_moments(self, p_x: densities.GaussianDensity) -> jnp.ndarray:
        """ Computes the expected mean and covariance

            mu_y = E[y] = M E[x] + b

            Sigma_y = E[yy'] - mu_y mu_y' = sigma_x^2 I + \sum_i U_i E[D_i(x)] U_i' + E[mu(x)mu(x)'] - mu_y mu_y'

        :param p_x: GaussianDensity
            The density which we average over.

        :return: (jnp.ndarray [p_R, Dy], jnp.ndarray [p_R, Dy, Dy])
            Returns the expected mean and covariance.
        """

        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "Ax_aBx_b_outer", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        # Sigma_y = .5 * (Sigma_y + Sigma_y.T)
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: densities.GaussianDensity) -> jnp.ndarray:
        """ Computes

            E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M f(x) + b)x' p(x) dx

        :param p_x: GaussianDensity
            The density which we average over.

        :return: jnp.ndarray [p_R, Dx, Dy]
            Returns the cross expectations.
        """

        Eyx = p_x.integrate(
            "Ax_aBx_b_outer", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Gets an approximation of the joint density

            p(x,y) ~= N(mu_{xy},Sigma_{xy}),

        The mean is given by

            mu_{xy} = (mu_x, mu_y)'

        with mu_y = E[mu_y(x)]. The covariance is given by

            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        mu_xy = jnp.concatenate([mu_x, mu_y], axis=1)
        # Sigma_xy = jnp.empty((p_x.R, self.Dy + self.Dx, self.Dy + self.Dx))
        Sigma_xy1 = jnp.concatenate(
            [p_x.Sigma, jnp.swapaxes(cov_yx, axis1=1, axis2=2)], axis=2
        )
        Sigma_xy2 = jnp.concatenate([cov_yx, Sigma_y], axis=2)
        Sigma_xy = jnp.concatenate([Sigma_xy1, Sigma_xy2], axis=1)
        # Sigma_xy[:,:self.Dx,:self.Dx] = p_x.Sigma
        # Sigma_xy[:,self.Dx:,:self.Dx] = cov_yx
        # Sigma_xy[:,:self.Dx,self.Dx:] = jnp.swapaxes(cov_yx, axis1=1, axis2=2)
        # Sigma_xy[:,self.Dx:,self.Dx:] = Sigma_y
        p_xy = densities.GaussianDensity(Sigma=Sigma_xy, mu=mu_xy)
        return p_xy

    def affine_conditional_transformation(
        self, p_x: densities.GaussianDensity
    ) -> ConditionalGaussianDensity:
        """ Gets an approximation of the joint density via moment matching

            p(x|y) ~= N(mu_{x|y},Sigma_{x|y}),

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: ConditionalDensity
            Returns the conditional density of x given y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = ConditionalGaussianDensity(M=M_new, b=b_new, Sigma=Sigma_new,)
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Gets an approximation of the marginal density

            p(y) ~= N(mu_y,Sigma_y),

        The mean is given by

            mu_y = E[mu_y(x)]. 

        The covariance is given by

            Sigma_y = E[yy'] - mu_ymu_y'.

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """

        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = densities.GaussianDensity(Sigma=Sigma_y, mu=mu_y)
        return p_y
