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
from jax import random
from typing import Tuple
from src_jax import densities, factors


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
                self.Lambda, self.ln_det_Sigma = self.invert_matrix(self.Sigma)
            else:
                self.Lambda, self.ln_det_Sigma = Lambda, ln_det_Sigma
            self.ln_det_Lambda = -self.ln_det_Sigma
        else:
            self.Lambda = Lambda
            if Sigma is None or ln_det_Sigma is None:
                self.Sigma, self.ln_det_Lambda = self.invert_matrix(self.Sigma)
            else:
                self.Sigma, self.ln_det_Lambda = Lambda, ln_det_Sigma
            self.ln_det_Sigma = -self.ln_det_Lambda

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

    def get_conditional_mu(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Computest the conditional mu function

            mu(x) = M x + b.

        :param x: jnp.ndarray [N, Dx]
            Instances, the mu should be conditioned on.

        :return: jnp.ndarray [R, N, Dy]
            Conditional means.
        """
        mu_y = jnp.einsum("abc,dc->adb", self.M, x) + self.b[:, None]
        return mu_y

    def condition_on_x(self, x: jnp.ndarray) -> densities.GaussianDensity:
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

    def set_y(self, y: jnp.ndarray) -> factors.ConjugateFactor:
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
        ln_beta_new = -0.5 * (yb_Lambda_yb + jnp.log(2 * jnp.pi * self.ln_det_Sigma))
        factor_new = factors.ConjugateFactor(Lambda_new, nu_new, ln_beta_new)
        return factor_new

    @staticmethod
    def invert_matrix(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        L = jnp.linalg.cholesky(A)
        # TODO: Check whether we can make it mor efficienty with solve_triangular.
        # L_inv = solve_triangular(L, jnp.eye(L.shape[0]), lower=True,
        #                         check_finite=False)
        L_inv = jnp.linalg.solve(L, jnp.eye(L.shape[1])[None])
        A_inv = jnp.einsum("acb,acd->abd", L_inv, L_inv)
        ln_det_A = 2.0 * jnp.sum(jnp.log(L.diagonal(axis1=1, axis2=2)), axis=1)
        return A_inv, ln_det_A

    def affine_joint_transformation(
        self, p_x: densities.GaussianDensity
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
        self, p_x: "ConditionalGaussianDensity"
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
        self, p_x: "ConditionalGaussianDensity"
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
        Sigma_x, ln_det_Lambda_x = p_x.invert_matrix(Lambda_x)
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
            :param W: jnp.ndarray [Dphi, Dx + 1]
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

    def get_conditional_mu(self, x: jnp.ndarray) -> jnp.ndarray:
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
    
    def set_y(self, y: jnp.ndarray):
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
        self, p_x: densities.GaussianDensity
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
        self, p_x: densities.GaussianDensity
    ) -> "ConditionalGaussianDensity":
        """ Gets an approximation of the joint density via moment matching

            p(x|y) ~= N(mu_{x|y},Sigma_{x|y}),

        :param p_x: GaussianDensity
            Marginal Gaussian density over x.

        :return: ConditionalDensity
            Returns the conditional density of x given y.
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = self.invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = ConditionalGaussianDensity(M=M_new, b=b_new, Sigma=Sigma_new,)
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity
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

    def condition_on_x(self, x: jnp.ndarray) -> densities.GaussianDensity:
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
    
    def set_y(self, y: jnp.ndarray):
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
        self, p_x: densities.GaussianDensity
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
        Lambda_y = self.invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = ConditionalGaussianDensity(M=M_new, b=b_new, Sigma=Sigma_new,)
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity
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
