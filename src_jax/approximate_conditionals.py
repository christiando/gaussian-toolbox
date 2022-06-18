from jax import numpy as jnp
from typing import Tuple
from src_jax import densities, factors, measures, conditionals
from utils.linalg import invert_matrix


class LConjugateFactorMGaussianConditional(conditionals.ConditionalGaussianDensity):
    def evaluate_phi(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the feature vector

        phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).

        :param x: Points where phi should be evaluated. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Feature vector. Dimensions are [N, Dphi].
        :rtype: jnp.ndarray
        """
        # phi_x = jnp.empty((N, self.Dphi))
        phi_x = jnp.block([x, self.k_func.evaluate(x).T])
        # phi_x[:,self.Dx:] = self.k_func.evaluate(x).T
        return phi_x

    def get_conditional_mu(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the conditional mu function mu(x) = mu(x) = M phi(x) + b.


        :param x: Points where phi should be evaluated. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional mu. Dimensions are [1, N, Dy].
        :rtype: jnp.ndarray
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

    def get_expected_moments(
        self, p_x: densities.GaussianDensity
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the expected covariance Sigma_y = E[yy'] - E[y]E[y]'.

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: Returns the expected mean and covariance. Dimensions are [p_R, Dy], and  [p_R, Dy, Dy].
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
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
        Exx = p_x.integrate("xx'")
        # Eff[:,:self.Dx,:self.Dx] =
        # Cross terms E[x k(x)']
        Ekx = p_k.integrate("x").reshape((p_x.R, self.Dk, self.Dx))
        # Eff[:,:self.Dx,self.Dx:] = jnp.swapaxes(Ekx, axis1=1, axis2=2)
        # Eff[:,self.Dx:,:self.Dx] = Ekx
        # kernel terms E[k(x)k(x)']
        Ekk = (
            p_k.multiply(self.k_func, update_full=True)
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
        """Compute E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M f(x) + b)x' p(x) dx.

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: Returns the cross expectations. Dimensions are [p_R, Dx, Dy].
        :rtype: jnp.ndarray
        """
        # E[xx']
        Exx = p_x.integrate("xx'")
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
        """Get an approximation of the joint density

            p(x,y) ~= N(mu_{xy},Sigma_{xy}),

        The mean is given by

            mu_{xy} = (mu_x, mu_y)'

        with mu_y = E[mu_y(x)]. The covariance is given by

            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: The joint distribution p(x,y).
        :rtype: densities.GaussianDensity
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        mu_xy = jnp.concatenate([mu_x, mu_y], axis=1)

        Sigma_xy = jnp.block(
            [[p_x.Sigma, jnp.swapaxes(cov_yx, axis1=1, axis2=2)], [cov_yx, Sigma_y]]
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
    ) -> conditionals.ConditionalGaussianDensity:
        """Get an approximation of the joint density via moment matching

        p(x|y) ~= N(mu_{x|y},Sigma_{x|y}),

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: The conditional density p(x|y).
        :rtype: conditionals.ConditionalGaussianDensity
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = conditionals.ConditionalGaussianDensity(
            M=M_new, b=b_new, Sigma=Sigma_new,
        )
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """ Get an approximation of the marginal density

            p(y) ~= N(mu_y,Sigma_y),

        The mean is given by

            mu_y = E[mu_y(x)]. 

        The covariance is given by

            Sigma_y = E[yy'] - mu_ymu_y'.

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: The joint distribution p(y).
        :rtype: densities.GaussianDensity
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = densities.GaussianDensity(Sigma=Sigma_y, mu=mu_y,)
        return p_y


class LRBFGaussianConditional(LConjugateFactorMGaussianConditional):
    def __init__(
        self,
        M: jnp.ndarray,
        b: jnp.ndarray,
        mu: jnp.ndarray,
        length_scale: jnp.ndarray,
        Sigma: jnp.ndarray = None,
        Lambda: jnp.ndarray = None,
        ln_det_Sigma: jnp.ndarray = None,
    ):
        """A conditional Gaussian density, with a linear RBF mean (LRBFM) function,

        p(y|x) = N(mu(x), Sigma)

        with the conditional mean function mu(x) = M phi(x) + b. 
        phi(x) is a feature vector of the form

        phi(x) = (1,x_1,...,x_m,k(h_1(x)),...,k(h_n(x))),

        with

        k(h) = exp(-h^2 / 2) and h_i(x) = (x_i - mu_{i}) / length_scale_i.

        Note, that the affine transformations will be approximated via moment matching.


        :param M: Matrix in the mean function. Dimensions should be [1, Dy, Dphi]
        :type M: jnp.ndarray
        :param b: Vector in the conditional mean function. Dimensions should be [1, Dy]
        :type b: jnp.ndarray
        :param mu: Parameters for linear mapping in the nonlinear functions. Dimensions should be [Dk, Dx]
        :type mu: jnp.ndarray
        :param length_scale: Length-scale of the kernels. Dimensions should be [Dk, Dx]
        :type length_scale: jnp.ndarray
        :param Sigma: The covariance matrix of the conditional. Dimensions should be [1, Dy, Dy], defaults to None
        :type Sigma: jnp.ndarray, optional
        :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [1, Dy, Dy], defaults to None
        :type Lambda: jnp.ndarray, optional
        :param ln_det_Sigma:  Log determinant of the covariance matrix. Dimensions should be [1], defaults to None
        :type ln_det_Sigma: jnp.ndarray, optional
        """
        super().__init__(M, b, Sigma, Lambda, ln_det_Sigma)
        self.mu = mu
        self.length_scale = length_scale
        self.Dk, self.Dx = self.mu.shape
        self.Dphi = self.Dk + self.Dx
        self.update_phi()

    def update_phi(self):
        """ Set up the non-linear kernel function in phi(x).
        """
        Lambda = jnp.eye(self.Dx)[None] / self.length_scale[:, None] ** 2
        nu = self.mu / self.length_scale ** 2
        ln_beta = -0.5 * jnp.sum((self.mu / self.length_scale) ** 2, axis=1)
        self.k_func = measures.GaussianDiagMeasure(
            Lambda=Lambda, nu=nu, ln_beta=ln_beta
        )

    def integrate_log_conditional(
        self,
        p_yx: densities.GaussianDensity,
        p_x: densities.GaussianDensity = None,
        **kwargs
    ) -> jnp.ndarray:
        """Integrate over the log conditional with respect to the pdf p_yx. I.e.
        
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
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        # E[(y - Mx - b) Lambda Mk phi(x)]

        Lambda_joint = jnp.zeros((self.Dk, self.Dy + self.Dx, self.Dy + self.Dx))
        Lambda_joint = Lambda_joint.at[:, self.Dy :, self.Dy :].set(self.k_func.Lambda)
        nu_joint = jnp.zeros([self.Dk, self.Dy + self.Dx])
        nu_joint = nu_joint.at[:, self.Dy :].set(self.k_func.nu)
        joint_k_func = factors.ConjugateFactor(
            Lambda=Lambda_joint, nu=nu_joint, ln_beta=self.k_func.ln_beta
        )
        p_yx_k = p_yx.multiply(joint_k_func, update_full=True)
        E_k_lin_term = jnp.reshape(
            p_yx_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_yx.R, self.Dk, self.Dy),
        )
        Mk = self.M[:, :, self.Dx :]
        lin_kernel_integral = jnp.einsum("abc,acb->a", Mk, E_k_lin_term)

        # E[phi(x)' Mk'  Lambda Mk phi(x)]
        if p_x is None:
            p_x = p_yx.get_marginal(jnp.arange(self.Dy, self.Dy + self.Dx))
        p_x_kk = p_x.multiply(self.k_func).multiply(self.k_func, update_full=True)
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

    def integrate_log_conditional_y(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> callable:
        """Compute the expectation over the log conditional, but just over x. I.e. it returns

        f(y) = int log(p(y|x))p(x)dx.
    
        :param p_x: Density over x.
        :type p_x: measures.GaussianDensity
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of y.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = self.M[:, :, : self.Dx]
        b = self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)

        Mk = self.M[:, :, self.Dx :]
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde)
        p_x_k = p_x.multiply(self.k_func, update_full=True)
        E_k = jnp.reshape(p_x_k.integrate(), (p_x.R, self.Dk))
        E_Mk = jnp.einsum("abc,ac->ab", self.Lambda, jnp.einsum("abc,ac->ab", Mk, E_k))
        linear_term = linear_integral + E_Mk
        quadratic_integral = p_x.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        E_k_lin = jnp.reshape(
            p_x_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_x.R, self.Dk, self.Dy),
        )
        E_Mk_lin = jnp.einsum("abc,acb->a", Mk, E_k_lin)
        p_x_kk = p_x_k.multiply(self.k_func, update_full=True)
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant_term = -0.5 * (
            quadratic_integral
            + 2 * E_Mk_lin
            + kernel_kernel_integral
            + self.ln_det_Sigma
            + self.Dy * jnp.log(2.0 * jnp.pi)
        )

        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_term)
            + constant_term
        )
        return log_expectation_y


class LSEMGaussianConditional(LConjugateFactorMGaussianConditional):
    def __init__(
        self,
        M: jnp.ndarray,
        b: jnp.ndarray,
        W: jnp.ndarray,
        Sigma: jnp.ndarray = None,
        Lambda: jnp.ndarray = None,
        ln_det_Sigma: jnp.ndarray = None,
    ):
        """A conditional Gaussian density, with a linear squared exponential mean (LSEM) function,

        p(y|x) = N(mu(x), Sigma)

        with the conditional mean function mu(x) = M phi(x) + b. 
        phi(x) is a feature vector of the form

        phi(x) = (1,x_1,...,x_m,k(h_1(x)),...,k(h_n(x))),

        with

        k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.

        Note, that the affine transformations will be approximated via moment matching.


        :param M: Matrix in the mean function. Dimensions should be [1, Dy, Dphi]
        :type M: jnp.ndarray
        :param b: Vector in the conditional mean function. Dimensions should be [1, Dy]
        :type b: jnp.ndarray
        :param W: Parameters for linear mapping in the nonlinear functions. Dimensions should be [Dk, Dx + 1]
        :type W: jnp.ndarray
        :param Sigma: The covariance matrix of the conditional. Dimensions should be [1, Dy, Dy], defaults to None
        :type Sigma: jnp.ndarray, optional
        :param Lambda: Information (precision) matrix of the Gaussians. Dimensions should be [1, Dy, Dy], defaults to None
        :type Lambda: jnp.ndarray, optional
        :param ln_det_Sigma:  Log determinant of the covariance matrix. Dimensions should be [1], defaults to None
        :type ln_det_Sigma: jnp.ndarray, optional
        """
        super().__init__(M, b, Sigma, Lambda, ln_det_Sigma)
        self.w0 = W[:, 0]
        self.W = W[:, 1:]
        self.Dx = self.W.shape[1]
        self.Dk = self.W.shape[0]
        self.Dphi = self.Dk + self.Dx
        self.update_phi()

    def update_phi(self):
        """ Set up the non-linear kernel function in phi(x).
        """
        v = self.W
        nu = self.W * self.w0[:, None]
        ln_beta = -0.5 * self.w0 ** 2
        self.k_func = factors.OneRankFactor(v=v, nu=nu, ln_beta=ln_beta)

    def integrate_log_conditional(
        self,
        p_yx: densities.GaussianDensity,
        p_x: densities.GaussianDensity = None,
        **kwargs
    ) -> jnp.ndarray:
        """Integrate over the log conditional with respect to the pdf p_yx. I.e.
        
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
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
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
            p_yx_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
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

    def integrate_log_conditional_y(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> callable:
        """Compute the expectation over the log conditional, but just over x. I.e. it returns

        f(y) = int log(p(y|x))p(x)dx.
    
        :param p_x: Density over x.
        :type p_x: measures.GaussianDensity
        :raises NotImplementedError: Only implemented for R=1.
        :return: The integral as function of y.
        :rtype: callable
        """
        if self.R != 1:
            raise NotImplementedError("Only implemented for R=1.")

        A = self.M[:, :, : self.Dx]
        b = self.b
        A_tilde = jnp.einsum("abc,acd->abd", self.Lambda, A)
        b_tilde = jnp.einsum("abc,ac->ab", self.Lambda, b)

        Mk = self.M[:, :, self.Dx :]
        linear_integral = p_x.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde)
        p_x_k = p_x.multiply(self.k_func, update_full=True)
        E_k = jnp.reshape(p_x_k.integrate(), (p_x.R, self.Dk))
        E_Mk = jnp.einsum("abc,ac->ab", self.Lambda, jnp.einsum("abc,ac->ab", Mk, E_k))
        linear_term = linear_integral + E_Mk
        quadratic_integral = p_x.integrate(
            "(Ax+a)'(Bx+b)", A_mat=A, a_vec=b, B_mat=A_tilde, b_vec=b_tilde
        )
        E_k_lin = jnp.reshape(
            p_x_k.integrate("(Ax+a)", A_mat=A_tilde, a_vec=b_tilde),
            (p_x.R, self.Dk, self.Dy),
        )
        E_Mk_lin = jnp.einsum("abc,acb->a", Mk, E_k_lin)
        p_x_kk = p_x_k.multiply(self.k_func, update_full=True)
        E_kk = jnp.reshape(p_x_kk.integral_light(), (p_x.R, self.Dk, self.Dk))
        E_MkkM = jnp.einsum("abc,adc->adb", jnp.einsum("abc,acd-> abd", Mk, E_kk), Mk)
        kernel_kernel_integral = jnp.trace(
            jnp.einsum("abc,acd->abd", self.Lambda, E_MkkM), axis1=-2, axis2=-1
        )
        constant_term = -0.5 * (
            quadratic_integral
            + 2 * E_Mk_lin
            + kernel_kernel_integral
            + self.ln_det_Sigma
            + self.Dy * jnp.log(2.0 * jnp.pi)
        )

        log_expectation_y = (
            lambda y: -0.5
            * jnp.einsum("ab,ab -> a", y, jnp.einsum("abc,ac->ab", self.Lambda, y))
            + jnp.einsum("ab,ab->a", y, linear_term)
            + constant_term
        )
        return log_expectation_y


class HCCovGaussianConditional(conditionals.ConditionalGaussianDensity):
    def __init__(
        self,
        M: jnp.ndarray,
        b: jnp.ndarray,
        sigma_x: jnp.ndarray,
        U: jnp.ndarray,
        W: jnp.ndarray,
        beta: jnp.ndarray,
    ):
        """A conditional Gaussian density, with a heteroscedastic cosh covariance (HCCov) function,

        p(y|x) = N(mu(x), Sigma(x))

        with the conditional mean function mu(x) = M x + b. 
        The covariance matrix has the form

        Sigma_y(x) = sigma_x^2 I + \sum_i U_i D_i(x) U_i',

        and D_i(x) = 2 * beta_i * cosh(h_i(x)) and h_i(x) = w_i'x + b_i

        Note, that the affine transformations will be approximated via moment matching.

        :param M: Matrix in the mean function. Dimensions should be [1, Dy, Dx].
        :type M: jnp.ndarray
        :param b: Vector in the conditional mean function. Dimensions should be [1, Dy].
        :type b: jnp.ndarray
        :param sigma_x: Diagonal noise parameter.
        :type sigma_x: jnp.ndarray
        :param U: Othonormal vectors for low rank noise part. Dimensions should be [Dy, Du].
        :type U: jnp.ndarray
        :param W: Noise weights for low rank components (w_i & b_i). Dimensions should be [Du, Dx + 1].
        :type W: jnp.ndarray
        :param beta: Scaling for low rank noise components. Dimensions should be [Du].
        :type beta: jnp.ndarray
        :raises NotImplementedError: Only works with R==1.
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
        """ Create the functions, that later need to be integrated over, i.e.

        exp(h_i(z)) and exp(-h_i(z))
        """
        nu = self.W[:, 1:]
        ln_beta = self.W[:, 0]
        self.exp_h_plus = factors.LinearFactor(nu, ln_beta)
        self.exp_h_minus = factors.LinearFactor(-nu, -ln_beta)

    def get_conditional_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the covariance at a given x, i.e.

        Sigma_y(x) = sigma_x^2 I + \sum_i U_i D_i(x) U_i',

        with D_i(x) = 2 * beta_i * cosh(h_i(x)) and h_i(x) = w_i'x + b_i.


        :param x: Instances, the mu should be conditioned on. Dimensions should be [N, Dx].
        :type x: jnp.ndarray
        :return: Conditional covariance. Dimensions are [N, Dy, Dy]
        :rtype: jnp.ndarray
        """
        D_x = self.beta[None, :, None] * (self.exp_h_plus(x) + self.exp_h_minus(x))
        Sigma_0 = self.sigma2_x * jnp.eye(self.Dy)
        Sigma_y_x = Sigma_0[None] + jnp.einsum(
            "ab,cb->ac", jnp.einsum("ab,cb->ca", self.U, D_x), self.U
        )
        return Sigma_y_x

    def condition_on_x(self, x: jnp.ndarray, **kwargs) -> densities.GaussianDensity:
        """Get Gaussian Density conditioned on x.

        :param x: Instances, the mu and Sigma should be conditioned on. Dimensions should be [N, Dx]
        :type x: jnp.ndarray
        :return: The density conditioned on x.
        :rtype: densities.GaussianDensity
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
        """Integrate covariance with respect to p(x).

        int Sigma_y(x)p(x) dx.

        :param p_x: The density the covatiance is integrated with.
        :type p_x: densities.GaussianDensity
        :return: Integrated covariance matrix. Dimensions are [Dy, Dy]
        :rtype: jnp.ndarray
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

    def get_expected_moments(
        self, p_x: densities.GaussianDensity
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the expected mean and covariance

        mu_y = E[y] = M E[x] + b

        Sigma_y = E[yy'] - mu_y mu_y' = sigma_x^2 I + \sum_i U_i E[D_i(x)] U_i' + E[mu(x)mu(x)'] - mu_y mu_y'


        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: Returns the expected mean and covariance. Dimensions should be [p_R, Dy] and [p_R, Dy, Dy].
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        mu_y = self.get_conditional_mu(p_x.mu)[0]
        Eyy = self.integrate_Sigma_x(p_x) + p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
        )
        Sigma_y = Eyy - mu_y[:, None] * mu_y[:, :, None]
        # Sigma_y = .5 * (Sigma_y + Sigma_y.T)
        return mu_y, Sigma_y

    def get_expected_cross_terms(self, p_x: densities.GaussianDensity) -> jnp.ndarray:
        """Compute E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M x + b)x' p(x) dx

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: Cross expectations. Dimensions are [p_R, Dx, Dy]
        :rtype: jnp.ndarray
        """
        Eyx = p_x.integrate(
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
        )
        return Eyx

    def affine_joint_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """Get an approximation of the joint density

            p(x,y) ~= N(mu_{xy},Sigma_{xy}),

        The mean is given by

            mu_{xy} = (mu_x, mu_y)'

        with mu_y = E[mu_y(x)]. The covariance is given by

            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').

        :param p_x: The density which we average over.
        :type p_x: densities.GaussianDensity
        :return: Joint distribution of p(x,y).
        :rtype: densities.GaussianDensity
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
    ) -> conditionals.ConditionalGaussianDensity:
        """Get an approximation of the joint density via moment matching

            p(x|y) ~= N(mu_{x|y},Sigma_{x|y}).

        :param p_x: Marginal Gaussian density over x.
        :type p_x: densities.GaussianDensity
        :return: Conditional density of p(x|y).
        :rtype: conditionals.ConditionalGaussianDensity
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        Lambda_y = invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(p_x)
        mu_x = p_x.mu
        cov_yx = Eyx - mu_y[:, :, None] * mu_x[:, None]
        M_new = jnp.einsum("abc,abd->acd", cov_yx, Lambda_y)
        b_new = mu_x - jnp.einsum("abc,ac->ab", M_new, mu_y)
        Sigma_new = p_x.Sigma - jnp.einsum("abc,acd->abd", M_new, cov_yx)
        cond_p_xy = conditionals.ConditionalGaussianDensity(
            M=M_new, b=b_new, Sigma=Sigma_new,
        )
        return cond_p_xy

    def affine_marginal_transformation(
        self, p_x: densities.GaussianDensity, **kwargs
    ) -> densities.GaussianDensity:
        """Get an approximation of the marginal density

            p(y) ~= N(mu_y,Sigma_y),

        The mean is given by

            mu_y = E[mu_y(x)]. 

        The covariance is given by

            Sigma_y = E[yy'] - mu_ymu_y'.

        :param p_x: Marginal Gaussian density over x.
        :type p_x: densities.GaussianDensity
        :return: The marginal density p(y).
        :rtype: densities.GaussianDensity
        """
        mu_y, Sigma_y = self.get_expected_moments(p_x)
        p_y = densities.GaussianDensity(Sigma=Sigma_y, mu=mu_y)
        return p_y

    def integrate_log_conditional(
        self, p_yx: measures.GaussianMeasure, **kwargs
    ) -> jnp.ndarray:
        raise NotImplementedError("Log integal not implemented!")

    def integrate_log_conditional_y(
        self, p_x: measures.GaussianMeasure, **kwargs
    ) -> callable:
        raise NotImplementedError("Log integal not implemented!")

