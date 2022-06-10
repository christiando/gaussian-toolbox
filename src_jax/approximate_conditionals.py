from jax import numpy as jnp
from src_jax import densities, factors, measures, conditionals
from utils.linalg import invert_matrix


class LSEMGaussianConditional(conditionals.ConditionalGaussianDensity):
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
        """ Computes

            E[yx'] = \int\int yx' p(y|x)p(x) dydx = int (M f(x) + b)x' p(x) dx

        :param p_x: GaussianDensity
            The density which we average over.

        :return: jnp.ndarray [p_R, Dx, Dy]
            Returns the cross expectations.
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
        cond_p_xy = conditionals.ConditionalGaussianDensity(
            M=M_new, b=b_new, Sigma=Sigma_new,
        )
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
        """Computes the expectation over the log conditional, but just over x. I.e. it returns

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
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=self.M, b_vec=self.b
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
            "(Ax+a)(Bx+b)'", A_mat=self.M, a_vec=self.b, B_mat=None, b_vec=None
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
    ) -> conditionals.ConditionalGaussianDensity:
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
        cond_p_xy = conditionals.ConditionalGaussianDensity(
            M=M_new, b=b_new, Sigma=Sigma_new,
        )
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

    def integrate_log_conditional(
        self, p_yx: measures.GaussianMeasure, **kwargs
    ) -> jnp.ndarray:
        raise NotImplementedError("Log integal not implemented!")

    def integrate_log_conditional_y(
        self, p_x: measures.GaussianMeasure, **kwargs
    ) -> callable:
        raise NotImplementedError("Log integal not implemented!")

