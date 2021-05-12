import numpy, scipy
from scipy.optimize import minimize
import sys
sys.path.append('../src/')
import densities, conditionals, factors, measures
from linear_ssm import KalmanSmoother, StateSpace_EM

class HeteroscedasticKalmanSmoother(KalmanSmoother):
    
    def __init__(self, X: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, Qz: numpy.ndarray, 
                 C: numpy.ndarray, d: numpy.ndarray, U: numpy.ndarray, W:  numpy.ndarray, beta: numpy.ndarray, 
                 sigma_x: float):
        """ This is a heteroscedastic Kalman filter, where the observation covariance is
        
        Sigma_x(z_t) = sigma_x^2 I + \sum_i U_i D_i(z_t) U_i',
        
        and D_i(z) = 2 * beta_i * cosh(h_i(z)) and h_i(z) = w_i'z + b_i
        
        
        :param X: numpy.ndarray [N, Dx]
            The observed data.
        :param A: numpy.ndarray [Dz, Dz]
            The state transition matrix.
        :param b: numpy.ndarray [Dz]
            The state transition offset.
        :param Qz: numpy.ndarray [Dz, Dz]
            The state covariance.
        :param C: numpy.ndarray [Dx, Dz]
            The observation matrix.
        :param d: numpy.ndarray [Dx]
            The observation offset.
        :param U: numpy.ndarray [Dx, D_noise]
            Othonormal vectors for low rank noise part.
        :param W: numpy.ndarray [Dz + 1, D_noise]
            Noise weights for low rank components (w_i & b_i).
        :param beta: numpy.ndarray [D_noise]
            Scaling for low rank noise components.
        :param sigma_x: float
            Diagonal noise parameter.
        """
        self.Dz, self.Dx = Qz.shape[0], X.shape[1]
        self.T = X.shape[0]
        self.X = X
        self.state_density = conditionals.ConditionalGaussianDensity(numpy.array([A]), numpy.array([b]), numpy.array([Qz]))
        self.C = C
        self.d = d
        self.U = U
        self.W = W
        self.beta = beta
        self.sigma2_x = sigma_x ** 2
        self.prediction_density = self._setup_density()
        self.filter_density = self._setup_density()
        self.smoothing_density = self._setup_density()
        self._setup_noise_diagonal_functions()
        
    def _setup_noise_diagonal_functions(self):
        """ Creates the functions, that later need to be integrated over, i.e.
        
        exp(h_i(z)) and exp(-h_i(z))
        """
        nu =  self.W[1:].T
        ln_beta = self.W[0]
        self.exp_h_plus = factors.LinearFactor(nu, ln_beta)
        self.exp_h_minus = factors.LinearFactor(-nu, -ln_beta)
    
    def integrate_Sigma_x(self, phi: densities.GaussianDensity):
        """ Returns the integral
        
        int Sigma_x(z) dphi(z).
        
        :param phi: GaussianDensity
            The density the covatiance is integrated with.
            
        :return: numpy.ndarray [Dx, Dx]
            Integrated covariance matrix.
        """
        # int 2 cosh(h(z)) dphi(z)
        D_int = phi.multiply(self.exp_h_plus).integrate() + phi.multiply(self.exp_h_minus).integrate()
        #print(D_int.shape)
        return self.sigma2_x * numpy.eye(self.Dx) + numpy.dot(numpy.dot(self.U, numpy.diag(self.beta * D_int)), self.U.T)
    
    def filtering(self, t: int):
        """ Here the approximate filtering density is calculated via moment matching.
        
        :param t: int
            Time index.
        """
        # p(z_t|x_{1:t-1})
        cur_prediction_density = self.prediction_density.slice([t])
        # get mu_x
        mu_x = numpy.dot(self.C, cur_prediction_density.mu[0]) + self.d
        # get Sigma_x
        Exx = self.integrate_Sigma_x(cur_prediction_density) + cur_prediction_density.integrate('Ax_aBx_b_outer',
                                                                                                A_mat=self.C, a_vec=self.d,
                                                                                                B_mat=self.C, b_vec=self.d)[0]
        Sigma_x = Exx - mu_x[None] * mu_x[:,None]
        Sigma_x = .5 * (Sigma_x + Sigma_x.T)
        data_density = densities.GaussianDensity(Sigma=numpy.array([Sigma_x]), mu=numpy.array([mu_x]))
        # get Covariances E[xz]
        Ezx = cur_prediction_density.integrate('Ax_aBx_b_outer', A_mat=None, a_vec=None, B_mat=self.C, b_vec=self.d)[0]
        Sigma_zx = Ezx - cur_prediction_density.integrate('x')[0][:,None] * mu_x[None]
        # Filter moments
        M = numpy.dot(Sigma_zx, data_density.Lambda[0])
        mu_f = cur_prediction_density.mu[0] + numpy.dot(M, self.X[t-1] - mu_x)
        Sigma_f = cur_prediction_density.Sigma[0] - numpy.dot(M, Sigma_zx.T)
        Sigma_f = .5 * (Sigma_f + Sigma_f.T)
        cur_filter_density = densities.GaussianDensity(Sigma=numpy.array([Sigma_f]), mu=numpy.array([mu_f]))

        self.filter_density.update([t], cur_filter_density)
        
    def compute_log_likelihood(self) -> float:
        llk = 0
        for t in range(1,self.T+1):
            cur_prediction_density = self.prediction_density.slice([t])
            # get mu_x
            mu_x = numpy.dot(self.C, cur_prediction_density.mu[0]) + self.d
            # get Sigma_x
            Exx = self.integrate_Sigma_x(cur_prediction_density) + cur_prediction_density.integrate('Ax_aBx_b_outer',
                                                                                                A_mat=self.C, a_vec=self.d,
                                                                                                B_mat=self.C, b_vec=self.d)[0]
            Sigma_x = Exx - mu_x[None] * mu_x[:,None]
            Sigma_x = .5 * (Sigma_x + Sigma_x.T)
            cur_px = densities.GaussianDensity(Sigma=numpy.array([Sigma_x]), mu=numpy.array([mu_x]))
            llk += cur_px.evaluate_ln(self.X[t-1:t])[0,0]
        return llk
    
    def compute_data_density(self) -> densities.GaussianDensity:
        mu_x, Sigma_x = numpy.empty((self.T, self.Dx)), numpy.empty((self.T, self.Dx, self.Dx))
        for t in range(1,self.T+1):
            cur_prediction_density = self.prediction_density.slice([t])
            # get mu_x
            mu_x_cur = numpy.dot(self.C, cur_prediction_density.mu[0]) + self.d
            # get Sigma_x
            Exx = self.integrate_Sigma_x(cur_prediction_density) + cur_prediction_density.integrate('Ax_aBx_b_outer',
                                                                                                A_mat=self.C, a_vec=self.d,
                                                                                                B_mat=self.C, b_vec=self.d)[0]
            Sigma_x_cur = Exx - mu_x_cur[None] * mu_x_cur[:,None]
            Sigma_x_cur = .5 * (Sigma_x_cur + Sigma_x_cur.T)
            mu_x[t-1] = mu_x_cur
            Sigma_x[t-1] = Sigma_x_cur
        data_density = densities.GaussianDensity(Sigma=Sigma_x, mu=mu_x)
        return data_density
    
    
class HeteroscedasticStateSpace_EM(StateSpace_EM):
    
    def __init__(self, X: numpy.ndarray, Dz: int, Du: int, noise_x: float=.1, noise_z: float=.1):
        """ This object implements a linear state-space model and optimizes it according to the 
            expectation-maximization (EM) pocedure.
            
        :param X: numpy.ndarray [N, Dx]
            The observed data.
        :param Dz: int
            Number of latent dimensions.
        :param Du: int
            Number of noise components.
        :param noise_x: float
            Initial standard deviation of observations. (Default=1e-1)
        :param noise_z: float
            Initial standard deviation of state transition matrix. (Default=1e-1) 
        """
        self.T, self.Dx = X.shape
        self.Dz = Dz
        self.Du = Du
        self.Qz = noise_z ** 2 * numpy.eye(self.Dz)
        self.A, self.b = .98 * numpy.eye(self.Dz), numpy.zeros(self.Dz)
        self.d = numpy.mean(X, axis=0)
        self.C = scipy.linalg.eigh(numpy.dot((X-self.d[None]).T, X-self.d[None]), eigvals=(self.Dx-self.Dz, self.Dx-1))[1]
        if self.Dx == self.Dz:
            self.C = numpy.eye(self.Dz)
        z_hat = numpy.dot(numpy.linalg.pinv(self.C), (X - self.d).T).T
        delta_X = X - numpy.dot(z_hat, self.C.T) - self.d
        cov = numpy.dot(delta_X.T, delta_X)
        self.U = scipy.linalg.eigh(cov, eigvals=(self.Dx-self.Du, self.Dx-1))[1]
        self.W = 1e-5 * numpy.random.randn(self.Dz + 1, self.Du)
        self.beta = 1e-3 * numpy.ones(self.Du)
        self.sigma_x = noise_x
        self.ks = HeteroscedasticKalmanSmoother(X, self.A, self.b, self.Qz, self.C, self.d, self.U, self.W, self.beta, self.sigma_x)
        self.Qz_inv, self.ln_det_Qz = self.ks.state_density.Lambda[0], self.ks.state_density.ln_det_Sigma[0]
        
    def mstep(self):
        """ Performs the maximization step, i.e. updates all model parameters.
        """
        self.update_A()
        self.update_b()
        self.update_Qz()
        self.update_state_density()
        self.update_C()
        self.update_d()
        self.update_U()
        self.update_sigma_beta()
        #self.update_beta()
        #self.update_W()
        #self.update_Qx()
        #
        self.update_emission_density()
        self.update_init_density()
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.ks.C = self.C
        self.ks.d = self.d
        self.ks.W = self.W
        self.ks.U = self.U
        self.ks.beta = self.beta
        self.ks.sigma2_x = self.sigma_x ** 2
        print(self.W)
        self.ks._setup_noise_diagonal_functions()
        
    ###### Functions for bounds #####
    def f(self, h, beta):
        return 2 * beta * numpy.cosh(h)
    
    def f_prime(self, h, beta):
        return 2 * beta * numpy.sinh(h)
    
    def g(self, omega, beta):
        return self.f_prime(omega, beta) / (self.sigma_x ** 2 + self.f(omega, beta)) / numpy.abs(omega)
                                      
    def k(self, h, omega):
        return numpy.log(self.sigma_x ** 2 + self.f(omega)) + .5 * self.g(omega) * (h ** 2 - omega ** 2)
        
    def get_lb_i(self, iu: int, phi: densities.GaussianDensity, conv_crit: float=1e-4, update: str=None):
        w_i = self.W[1:,iu:iu+1].T
        v = numpy.tile(w_i, (self.T, 1))
        b_i = self.W[0,iu:iu+1]
        u_i = self.U[:,iu:iu+1]
        beta = self.beta[iu:iu+1]
        uC = numpy.dot(u_i.T, -self.C)
        ux_d = numpy.dot(u_i.T, self.ks.X.T-self.d[:,None])
        # Lower bound for E[ln (sigma_x^2 + f(h))]
        omega_dagger = numpy.sqrt(phi.integrate('Ax_aBx_b_inner', A_mat=w_i, a_vec=b_i,
                                                                  B_mat=w_i, b_vec=b_i))
        f_omega_dagger = self.f(omega_dagger, beta)
        log_lb = numpy.log(self.sigma_x ** 2 + f_omega_dagger)
        # Lower bound for E[f(h) / (sigma_x^2 + f(h)) * (u'epsilon(z))^2]
        omega_star = numpy.ones(self.T)
        converged = False
        num_iter = 0
        while not converged and num_iter < 10:
            # From the lower bound term
            g_omega = self.g(omega_star, beta)
            nu_plus = (1. - g_omega[:,None] * b_i) * w_i
            nu_minus = (-1. - g_omega[:,None] * b_i) * w_i
            ln_beta = - numpy.log(self.sigma_x ** 2 + self.f(omega_star, beta)) - .5 * g_omega * (b_i ** 2 - omega_star ** 2) + numpy.log(beta)
            ln_beta_plus = ln_beta + b_i
            ln_beta_minus = ln_beta - b_i
            # Create OneRankFactors
            exp_factor_plus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
            exp_factor_minus = factors.OneRankFactor(v=v, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
            # Create the two measures
            exp_phi_plus = phi.hadamard(exp_factor_plus)
            exp_phi_minus = phi.hadamard(exp_factor_minus)
            # Fourth order integrals E[h^2 (x-Cz-d)^2]
            mat1 = uC
            vec1 = ux_d.T
            mat2 = w_i
            vec2 = b_i
            quart_int_plus = exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1, 
                                                                              C_mat=mat2, c_vec=vec2, D_mat=mat2, d_vec=vec2)
            quart_int_minus = exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1, 
                                                                              C_mat=mat2, c_vec=vec2, D_mat=mat2, d_vec=vec2)
            quart_int = quart_int_plus + quart_int_minus
            # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate('Ax_aBx_b_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
            quad_int_minus = exp_phi_minus.integrate('Ax_aBx_b_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
            quad_int = quad_int_plus + quad_int_minus
            omega_old = omega_star
            #omega_star = numpy.amin([numpy.amax([numpy.sqrt(quart_int / quad_int), 1e-10]), 1e2])
            #quad_int[quad_int < 1e-4] = 1e-4
            omega_star = numpy.sqrt(numpy.abs(quart_int / quad_int))
            # For numerical stability
            omega_star[omega_star < 1e-8] = 1e-8
            #omega_star[omega_star > 30] = 30
            #print(numpy.amax(numpy.abs(omega_star - omega_old)))
            converged = numpy.amax(numpy.abs(omega_star - omega_old)) < conv_crit
            num_iter += 1
        #print(numpy.amax(numpy.abs(omega_star - omega_old)))
        mat1 = -self.C
        vec1 = self.ks.X - self.d[None]
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
        R = R_plus + R_minus
        uRu = numpy.sum(u_i * numpy.dot(numpy.sum(R, axis=0), u_i))
        log_lb_sum = numpy.sum(log_lb)
        if update == 'gradients':
            uRu = numpy.sum(u_i * numpy.dot(numpy.sum(R, axis=0), u_i))
            ##### w_i gradiend ######################################################################
            # E[f'(h)exp(-k(h,omega^*)) dh/dw (u'epsilon(z))^2]
            # Matrix and vector for dh/dw
            dW = numpy.zeros((self.Dz + 1, self.Dz))
            dW[1:] = numpy.eye(self.Dz)
            db = numpy.zeros(self.Dz + 1)
            db[0] = 1
            dw_i = numpy.sum(exp_phi_plus.integrate('Ax_aBx_bCx_c_outer', A_mat=uC, a_vec=ux_d.T,
                                                    B_mat=uC, b_vec=ux_d.T, C_mat=dW, c_vec=db), axis=0)
            dw_i -= numpy.sum(exp_phi_minus.integrate('Ax_aBx_bCx_c_outer', A_mat=uC, a_vec=ux_d.T, 
                                                      B_mat=uC, b_vec=ux_d.T, C_mat=dW, c_vec=db), axis=0)
            # -g(omega) * E[f(h)exp(-k(h,omega^*)) h dh/dw (u'epsilon(z))^2]
            dw_i -= numpy.einsum('a,ab->b', g_omega, exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_outer', A_mat=w_i, a_vec=b_i,
                                                                            B_mat=uC, b_vec=ux_d.T, C_mat=uC, c_vec=ux_d.T,
                                                                            D_mat=dW, d_vec=db)[:,0])
            dw_i -= numpy.einsum('a,ab->b', g_omega, exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_outer', A_mat=w_i, a_vec=b_i,
                                                                             B_mat=uC, b_vec=ux_d.T, C_mat=uC, c_vec=ux_d.T,
                                                                             D_mat=dW, d_vec=db)[:,0])
            dw_i /= self.sigma_x ** 2
            # g(omega^+)E[h dh/dw]
            dw_i -= numpy.einsum('a,ab->b', self.g(omega_dagger, beta), phi.integrate('Ax_aBx_b_outer', A_mat=w_i, a_vec=b_i, 
                                                                                      B_mat=dW, b_vec=db)[:,0])
            dw_i /= 2.
            ###########################################################################################
            ##### beta_i gradient #####################################################################
            weighted_R = numpy.einsum('abc,a->bc', R, 1. / (self.sigma_x ** 2 + self.f(omega_star, beta))) 
            #  u'R u / (sigma_x^2 + f(omega^*))
            dln_beta_i = numpy.sum(u_i * numpy.dot(weighted_R, u_i))
            dln_beta_i -= numpy.sum(f_omega_dagger / (self.sigma_x ** 2 + f_omega_dagger))
            dln_beta_i /= 2.
            ##### sigma_x ** 2 gradient ###############################################################
            dlnsigma2 =  - uRu / self.sigma_x ** 2
            dlnsigma2 -= numpy.sum(u_i * numpy.dot(weighted_R, u_i))
            dlnsigma2 -= numpy.sum(self.sigma_x ** 2 / (self.sigma_x ** 2 + f_omega_dagger))
            dlnsigma2 /= 2.
            return uRu, log_lb_sum, dw_i, dln_beta_i, dlnsigma2
        elif update == 'C':
            intD_inv_zz_plus = exp_phi_plus.integrate('xx')
            intD_inv_zz_minus = exp_phi_minus.integrate('xx')
            intD_inv_zz = intD_inv_zz_plus + intD_inv_zz_minus
            intD_inv_z_plus = exp_phi_plus.integrate('x')
            intD_inv_z_minus = exp_phi_minus.integrate('x')
            intD_inv_z = intD_inv_z_plus + intD_inv_z_minus
            return intD_inv_z, intD_inv_zz
        elif update == 'd':
            intD_inv_z_plus = exp_phi_plus.integrate('x')
            intD_inv_z_minus = exp_phi_minus.integrate('x')
            intD_inv_z = intD_inv_z_plus + intD_inv_z_minus
            intD_inv_plus = exp_phi_plus.integrate()
            intD_inv_minus = exp_phi_minus.integrate()
            intD_inv = intD_inv_plus + intD_inv_minus
            return intD_inv, intD_inv_z
        elif update == 'U':
            return numpy.sum(R, axis=0)
        else:
            return uRu, log_lb_sum
    
    def update_sigma_beta(self):
        x0 = numpy.concatenate([numpy.array([numpy.log(self.sigma_x ** 2)]), numpy.log(self.beta), self.W.flatten()])
        bounds = [(-8, 5)] + [(-10, 10)] * self.Du + [(-10,10)] * (self.Du * (self.Dz + 1))
        result = minimize(self.parameter_optimization_sigma_beta, x0, jac=True, method='L-BFGS-B', bounds=bounds)
        #print(result)
        self.sigma_x = numpy.exp(.5*result.x[0])
        self.beta = numpy.exp(result.x[1:self.Du + 1])
        self.W = result.x[self.Du + 1:].reshape((self.Dz+1, self.Du))
    
    def update_parameters_sigma_beta(self, params):
        self.sigma_x = numpy.exp(.5 * params[0])
        self.beta = numpy.exp(params[1:self.Du + 1])
        self.W = params[self.Du + 1:].reshape((self.Dz + 1, self.Du))
        
    def parameter_optimization_sigma_beta(self, params):
        self.update_parameters_sigma_beta(params)
        dW = numpy.zeros(self.W.shape)
        dln_beta = numpy.zeros(self.Du)
        dlnsigma2_x = numpy.zeros(1)
        phi = self.ks.smoothing_density.slice(range(1,self.T+1))
        # E[epsilon(z)^2]
        mat = -self.C
        vec = self.ks.X - self.d
        E_epsilon2 = numpy.sum(phi.integrate('Ax_aBx_b_inner', A_mat=mat, a_vec=vec, B_mat=mat, b_vec=vec), axis=0)
        dlnsigma2_x += .5 * E_epsilon2 / self.sigma_x ** 2
        # E[D_inv epsilon(z)^2(z)] & E[log(sigma^2 + f(h))]
        E_D_inv_epsilon2 = 0
        E_ln_sigma2_f = 0
        for iu in range(self.Du):
            uRu_i, log_lb_sum_i, dw_i, dln_beta_i, dlnsigma2_i  = self.get_lb_i(iu, phi, update='gradients')
            E_D_inv_epsilon2 += uRu_i
            E_ln_sigma2_f += log_lb_sum_i
            dW[:,iu] = dw_i
            dln_beta[iu] = dln_beta_i
            dlnsigma2_x += dlnsigma2_i
        # data part
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / self.sigma_x ** 2
        # determinant part
        Qm -= .5 * E_ln_sigma2_f + .5 * self.T * (self.Dx - self.Du) * numpy.log(self.sigma_x ** 2)
        # constant part
        Qm -= self.T * self.Dx * numpy.log(2 * numpy.pi)
        dlnsigma2_x -= .5 * self.T * (self.Dx - self.Du) #/ self.sigma_x ** 2
        #print(numpy.array([dlnsigma2_x]).shape, dln_beta.shape)
        gradients = numpy.concatenate([dlnsigma2_x, dln_beta, dW.flatten()])
        return -Qm, -gradients
        
        
    def get_Qm(self):
        phi = self.ks.smoothing_density.slice(range(1,self.T+1))
        # E[epsilon(z)^2]
        mat = -self.C
        vec = self.ks.X - self.d
        E_epsilon2 = numpy.sum(phi.integrate('Ax_aBx_b_inner', A_mat=mat, a_vec=vec, B_mat=mat, b_vec=vec), axis=0)
        # E[D_inv epsilon(z)^2(z)] & E[log(sigma^2 + f(h))]
        E_D_inv_epsilon2 = 0
        E_ln_sigma2_f = 0
        for iu in range(self.Du):
            uRu_i, log_lb_sum_i  = self.get_lb_i(iu, phi)
            E_D_inv_epsilon2 += uRu_i
            E_ln_sigma2_f += log_lb_sum_i
        # data part
        Qm = -.5 * (E_epsilon2 - E_D_inv_epsilon2) / self.sigma_x ** 2
        # determinant part
        Qm -= .5 * E_ln_sigma2_f + .5 * self.T * (self.Dx - self.Du) * numpy.log(self.sigma_x ** 2)
        # constant part
        Qm -= self.T * self.Dx * numpy.log(2 * numpy.pi)
        return Qm
        
    def update_C(self):
        phi = self.ks.smoothing_density.slice(range(1,self.T+1))
        intD_inv_z, intD_inv_zz = numpy.zeros((self.Du, self.T, self.Dz)), numpy.zeros((self.Du, self.Dz, self.Dz))
        for iu in range(self.Du):
            intD_inv_z_i, intD_inv_zz_i = self.get_lb_i(iu, phi, update='C')
            intD_inv_z[iu] = intD_inv_z_i
            intD_inv_zz[iu] += numpy.sum(intD_inv_zz_i, axis=0)
        Ez = phi.integrate('x')
        Ezz = numpy.sum(phi.integrate('xx'), axis=0)
        Ezx_d = numpy.einsum('ab,ac->bc', Ez, self.ks.X - self.d)
        UU = numpy.einsum('ab,cb->bac', self.U, self.U)
        intD_inv_zx_d = numpy.einsum('abc,bd->adc', intD_inv_z, self.ks.X - self.d)
        
        def Q_C_func(params: numpy.ndarray) -> (float, numpy.ndarray):
            C = numpy.reshape(params, (self.Dx, self.Dz))
            tr_CEzx_d = numpy.trace(numpy.dot(C, Ezx_d))
            tr_CC_Ezz = numpy.trace(numpy.dot(numpy.dot(C.T, C), Ezz))
            tr_uu_CC_Dinv_zx_d = numpy.sum(numpy.trace(numpy.einsum('abc,acd->abd', UU, numpy.einsum('ab,cdb->cad', C, intD_inv_zx_d)), axis1=1, axis2=2))
            CD_inv_zz = numpy.einsum('ab,cbd->cad', C, intD_inv_zz)
            CD_inv_zzC = numpy.einsum('abc,dc->abd', CD_inv_zz, C)
            uCD_inv_zzCu = numpy.sum(numpy.einsum('ab,ba->b',self.U, numpy.einsum('abc,ca->ab', CD_inv_zzC, self.U)))
            Q_C = 2 * tr_CEzx_d - tr_CC_Ezz - 2 * tr_uu_CC_Dinv_zx_d + uCD_inv_zzCu #- 2 * tr_uu_CC_Dinv_zx_d + uCD_inv_zzCu
            Q_C /= 2 * self.sigma_x ** 2
            C_Ezz = numpy.dot(C, Ezz)
            UU_C_Dinv_zz = numpy.sum(numpy.einsum('abc,abd->acd', UU, CD_inv_zz), axis=0)
            UU_Dinv_zx_d = numpy.sum(numpy.einsum('abc,abd->acd', UU, intD_inv_zx_d), axis=0)
            dQ_C = Ezx_d.T - UU_Dinv_zx_d + UU_C_Dinv_zz - C_Ezz #- UU_Dinv_zx_d + UU_C_Dinv_zz
            dQ_C /= self.sigma_x ** 2
            return -Q_C, -dQ_C.flatten() 
        
        x0 = self.C.flatten()
        result = minimize(Q_C_func, x0, method='L-BFGS-B', jac=True)
        self.C = result.x.reshape(self.Dx, self.Dz)
    
    def update_d(self):
        phi = self.ks.smoothing_density.slice(range(1,self.T+1))
        intD_inv, intD_inv_z = numpy.zeros((self.Du, self.T)), numpy.zeros((self.Du, self.T, self.Dz))
        for iu in range(self.Du):
            intD_inv_i, intD_inv_z_i = self.get_lb_i(iu, phi, update='d')
            intD_inv[iu] = intD_inv_i
            intD_inv_z[iu] += intD_inv_z_i
        Ez = phi.integrate('x')
        CEz = numpy.dot(self.C, numpy.sum(Ez,axis=0))
        UU = numpy.einsum('ab,cb->bac', self.U, self.U)
        A = numpy.eye(self.Dx) * self.T - numpy.sum(numpy.sum(intD_inv, axis=1)[:,None,None] * UU, axis=0)
        sum_X = numpy.sum(self.ks.X, axis=0)
        intDinv_X_UU = numpy.sum(numpy.einsum('ab,abc->ac', numpy.einsum('ab,bc->ac',intD_inv[:,], self.ks.X), UU), axis=0)
        UU_C_intDinv_z = numpy.sum(numpy.einsum('abc,ac->ab', UU, numpy.einsum('ab,cb->ca', self.C, numpy.sum(intD_inv_z,axis=1))), axis=0)
        b = sum_X - intDinv_X_UU - CEz + UU_C_intDinv_z
        self.d = numpy.linalg.solve(A,b)
    
    def get_lower_bounds(self):
        phi = self.ks.smoothing_density.slice(range(1,self.T+1))
        for iu in range(self.Du):
            uRu, log_lb_sum, intD_inv, intD_inv_z, intD_inv_zz = self.get_lb_i(iu)
            self.R_mat[:,iu] = R_mat
            self.omega_star[:,iu] = omega_star
            self.intD_inv_zz += intD_inv_zz
            self.intD_inv_z += intD_inv_z
            self.intD_inv += intD_inv 
            log_lb, omega_dagger = self.get_log_lb(iu)
            self.log_lb[iu] = numpy.sum(log_lb, axis=0)
            self.omega_dagger[:, iu] = omega_dagger

    def compute_Q(self):
        """ Calculates the Qfunction. (TODO: Still a small bug here, somewhere!)
        """
        # E[(z_t - Az_{t-1} - b)'Qz^{1}(z_{t} - Az_{t-1} - b)]
        joint_density = self.ks.smoothing_density.affine_joint_transformation(self.ks.state_density)
        A = numpy.hstack([-self.A, numpy.eye(self.Dz)])
        a = -self.b
        B = numpy.dot(self.Qz_inv, A)
        b = -numpy.dot(self.Qz_inv, self.b)
        Ezz = numpy.sum(joint_density.integrate('Ax_aBx_b_inner', A_mat=A, a_vec=a, B_mat=B, b_vec=b)[:-1])
        Ezz += self.T * (self.ln_det_Qz + self.Dz * numpy.log(2 * numpy.pi))
        # E[(x_t - Cz_{t} - d)'Qx^{-1}(x_{t} - Cz_{t} - d)]
        A = -self.C
        a_t = self.ks.X - self.d[None]
        B = A / self.sigma_x ** 2
        b_t = a_t / self.sigma_x ** 2
        Exx = 0
        for t in range(1,self.T+1):
            cur_smooth_density = self.ks.smoothing_density.slice([t])
            Exx += cur_smooth_density.integrate('Ax_aBx_b_inner', A_mat=A, a_vec=a_t[t-1], B_mat=B, b_vec=b_t[t-1])[0]
        
        uRu = numpy.einsum('ab,ba->b', self.U, numpy.einsum('abc,ca->ab', self.R_mat, self.U))
        Exx += numpy.sum(uRu)
        Exx += numpy.sum(self.log_lb) + self.T * self.Dx * numpy.log(2 * numpy.pi)
        # E[(z_0 - mu0)'Sigma0^{-1}(z_0 - mu0)]
        init_smooth_density = self.ks.smoothing_density.slice([0])
        A = numpy.eye(self.Dz)
        a = -self.ks.filter_density.mu[0]
        B = self.ks.filter_density.Lambda[0]
        b = numpy.dot(self.ks.filter_density.Lambda[0], a)
        Ez0 = init_smooth_density.integrate('Ax_aBx_b_inner', A_mat=A, a_vec=a, B_mat=B, b_vec=b)[0]
        Ez0 += self.ks.filter_density.ln_det_Sigma[0] + self.Dz * numpy.log(2 * numpy.pi)
        return - .5 * (Exx + Ezz + Ez0)
    
    def update_U(self):
        converged = False
        phi = self.ks.smoothing_density.slice(range(1,self.T+1))
        intD_inv_z, intD_inv_zz = numpy.zeros((self.Du, self.T, self.Dz)), numpy.zeros((self.Du, self.Dz, self.Dz))
        R = numpy.empty([self.Du, self.Dx, self.Dx])
        for iu in range(self.Du):
            R[iu] = self.get_lb_i(iu, phi, update='U')
        num_iter = 0
        while not converged and num_iter < 50:
            U_old = numpy.copy(self.U)
            for iu in range(self.Du):
                U_not_i = numpy.delete(self.U, [iu], axis=1)
                V = self.partial_gs(U_not_i)
                VRV = numpy.dot(numpy.dot(V.T, R[iu]), V)
                alpha = scipy.linalg.eigh(VRV, eigvals=(VRV.shape[0]-1,VRV.shape[0]-1))[1]
                u_new = numpy.dot(V, alpha)[:,0]
                self.U[:,iu] = u_new
            converged = numpy.sum(numpy.abs(self.U - U_old)) < 1e-4
            num_iter += 1
            
    @staticmethod
    def gen_lin_ind_vecs(U):
        N, M = U.shape
        rand_vecs = numpy.random.rand(N, N - M)
        V_fixed = numpy.hstack([U, rand_vecs])
        V = numpy.copy(V_fixed)
        for m in range(N - M):
            v = rand_vecs[:,m]
            V[:,M+m] -= numpy.dot(V_fixed.T, v) / numpy.sqrt(numpy.sum(v ** 2))
        return V[:,M:]
    
    @staticmethod
    def proj(U, v):
        return numpy.dot(numpy.dot(v, U) / numpy.linalg.norm(U, axis=0), U.T)

    def partial_gs(self, U):
        """ Partial Gram-Schmidt process, to generate 
        """
        N, M = U.shape
        V = numpy.empty((N, N - M))
        I = self.gen_lin_ind_vecs(U)#numpy.random.randn(N,N-M)
        #I = numpy.eye(N)[:,M:]
        #I[-1,0] = 1
        for d in range(N - M):
            v = I[:,d]
            V[:,d] = v - self.proj(U, v) - self.proj(V[:,:d], v)
            V[:,d] /= numpy.sqrt(numpy.sum(V[:,d] ** 2))  
        return V