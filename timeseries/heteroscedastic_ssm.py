import numpy, scipy
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
        D_int = phi.multiply(self.exp_h_plus).integrate()[0] + phi.multiply(self.exp_h_minus).integrate()[0]
        return self.sigma2_x * numpy.eye(self.Dx) + numpy.dot(self.U, (self.beta * D_int * self.U).T)
    
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
        data_density = densities.GaussianDensity(Sigma=numpy.array([Sigma_x]), mu=numpy.array([mu_x]))
        # get Covariances E[xz]
        Ezx = cur_prediction_density.integrate('Ax_aBx_b_outer', A_mat=None, a_vec=None, B_mat=self.C, b_vec=None)[0]
        Sigma_zx = Ezx - cur_prediction_density.integrate('x')[0][:,None] * mu_x[None]
        # Filter moments
        M = numpy.dot(Sigma_zx, data_density.Lambda[0])
        mu_f = cur_prediction_density.mu[0] + numpy.dot(M, self.X[t-1] - mu_x)
        Sigma_f = cur_prediction_density.Sigma[0] - numpy.dot(M, Sigma_zx.T)
        cur_filter_density = densities.GaussianDensity(Sigma=numpy.array([Sigma_f]), mu=numpy.array([mu_f]))
        self.filter_density.update([t], cur_filter_density)
        
    def compute_log_likelihood(self) -> float:
        llk = 0
        for t in range(1,self.T+1):
            cur_prediction_density = self.prediction_density.slice([t])
            # get mu_x
            mu_x = numpy.dot(self.C, cur_prediction_density.mu[0]) + self.d
            # get Sigma_x
            Sigma_x = self.integrate_Sigma_x(cur_prediction_density)
            cur_px = densities.GaussianDensity(Sigma=numpy.array([Sigma_x]), mu=numpy.array([mu_x]))
            llk += cur_px.evaluate_ln(self.X[t-1:t])[0,0]
        return llk
    
    
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
        self.A, self.b = numpy.eye(self.Dz), numpy.zeros((self.Dz,))
        self.C, self.d = numpy.random.randn(self.Dx, self.Dz), numpy.zeros((self.Dx,))
        if self.Dx == self.Dz:
            self.C = numpy.eye(self.Dz)
        self.U = scipy.linalg.eigh(numpy.dot(X.T, X), eigvals=(self.Dx-self.Du, self.Dx-1))[1]
        self.W = numpy.random.randn(self.Dz + 1, self.Du)
        self.beta = noise_z ** 2 * numpy.ones(self.Du)
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
        #self.update_C()
        #self.update_d()
        #self.update_Qx()
        #self.update_U()
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
        self.ks.sigma_x = self.sigma_x
        
    ###### Functions for bounds #####
    def f(self, h, beta):
        return 2 * beta * numpy.cosh(h)
    
    def f_prime(self, h, beta):
        return 2 * beta * numpy.sinh(h)
    
    def g(self, omega, beta):
        return self.f_prime(omega, beta) / (numpy.abs(omega) * (self.sigma_x ** 2 + self.f(omega, beta)))
                                      
    def k(self, h, omega):
        return numpy.log(self.sigma_x ** 2 + self.f(omega_dagger)) + .5 * self.g(omega) * (h ** 2 - omega ** 2)
        
    def get_inv_lb(self, t: int, iu: int, conv_crit: float=1e-4):
        phi = self.ks.smoothing_density.slice([t])
        omega_star = 1.
        w_i = self.W[1:,iu:iu+1].T
        b_i = self.W[0,iu:iu+1]
        u_i = self.U[:,iu:iu+1]
        beta = self.beta[iu:iu+1]
        uC = numpy.dot(u_i.T, -self.C)
        ux_d = numpy.dot(u_i.T, self.ks.X[t-1]-self.d)
        converged = False
        while not converged:
            # From the lower bound term
            g_omega = self.g(omega_star, beta)
            nu_plus = (1. - g_omega) * b_i * w_i
            nu_minus = (-1. - g_omega) * b_i * w_i
            ln_beta = - numpy.log(self.sigma_x ** 2 + self.f(omega_star, beta)) + .5 * g_omega * omega_star ** 2 + numpy.log(beta)
            ln_beta_plus = ln_beta + b_i
            ln_beta_minus = ln_beta - b_i
            # Create OneRankFactors
            exp_factor_plus = factors.OneRankFactor(v=w_i, g=g_omega, nu=nu_plus, ln_beta=ln_beta_plus)
            exp_factor_minus = factors.OneRankFactor(v=w_i, g=g_omega, nu=nu_minus, ln_beta=ln_beta_minus)
            # Create the two measures
            exp_phi_plus = phi.multiply(exp_factor_plus)
            exp_phi_minus = phi.multiply(exp_factor_minus)
            # Fourth order integrals E[h^2 (x-Cz-d)^2]
            mat1 = uC
            vec1 = ux_d
            mat2 = w_i
            vec2 = b_i
            quart_int_plus = exp_phi_plus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1, 
                                                                              C_mat=mat2, c_vec=vec2, D_mat=mat2, d_vec=vec2)
            quart_int_minus = exp_phi_minus.integrate('Ax_aBx_bCx_cDx_d_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1, 
                                                                              C_mat=mat2, c_vec=vec2, D_mat=mat2, d_vec=vec2)
            quart_int = quart_int_plus[0] + quart_int_minus[0]
            # Second order integrals E[(x-Cz-d)^2] Dims: [Du, Dx, Dx]
            quad_int_plus = exp_phi_plus.integrate('Ax_aBx_b_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
            quad_int_minus = exp_phi_plus.integrate('Ax_aBx_b_inner', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)
            quad_int = quad_int_plus[0] + quad_int_minus[0]
            omega_old = omega_star
            omega_star = numpy.amin([numpy.amax([numpy.sqrt(quart_int / quad_int), 1e-10]), 1e2])
            converged = numpy.abs(omega_star - omega_old) < conv_crit
            
        mat1 = -self.C
        vec1 = self.ks.X[t-1]-self.d
        R_plus = exp_phi_plus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)[0]
        R_minus = exp_phi_minus.integrate('Ax_aBx_b_outer', A_mat=mat1, a_vec=vec1, B_mat=mat1, b_vec=vec1)[0]
        R = R_plus + R_minus
        intD_inv_zz_plus = exp_phi_plus.integrate('xx')[0]
        intD_inv_zz_minus = exp_phi_minus.integrate('xx')[0]
        intD_inv_zz = intD_inv_zz_plus + intD_inv_zz_minus
        intD_inv_z_plus = exp_phi_plus.integrate('x')[0]
        intD_inv_z_minus = exp_phi_minus.integrate('x')[0]
        intD_inv_z = intD_inv_z_plus + intD_inv_z_minus
        intD_inv_zx_d = intD_inv_z[:,None] * (self.ks.X[t-1] - self.d)[None]
        intD_inv_plus = exp_phi_plus.integrate()[0]
        intD_inv_minus = exp_phi_minus.integrate()[0]
        intD_inv = intD_inv_plus + intD_inv_minus
        intD_inv_x_Cz = intD_inv * self.ks.X[t-1] - numpy.dot(self.C, intD_inv_z)
        return R, intD_inv, intD_inv_zz, intD_inv_zx_d, intD_inv_x_Cz, omega_star
    
    def get_log_lb(self, t: int, iu: int):
        w_i = self.W[1:,iu:iu+1].T
        b_i = self.W[0,iu:iu+1]
        beta = self.beta[iu:iu+1]
        phi = self.ks.smoothing_density.slice([t])
        omega_dagger = numpy.sqrt(phi.integrate('Ax_aBx_b_inner', A_mat=w_i, a_vec=b_i,
                                                                  B_mat=w_i, b_vec=b_i))
        log_lb = numpy.log(self.sigma_x ** 2 + self.f(omega_dagger, beta))
        return log_lb, omega_dagger
    
    def get_lower_bounds(self):
        self.omega_star = numpy.empty((self.T, self.Du))
        self.R_mat = numpy.zeros((self.Du, self.Dx, self.Dx))
        self.intD_inv = numpy.zeros(1,)
        self.intD_inv_zz = numpy.zeros((self.Dz, self.Dz))
        self.intD_inv_zx_d = numpy.zeros((self.Dz, self.Dx))
        self.intD_inv_x_Cz = numpy.zeros((self.Dx))
        self.omega_dagger = numpy.empty((self.T, self.Du))
        self.log_lb = numpy.zeros(self.Du)
        for t in range(1,self.T+1):
            for iu in range(self.Du):
                R_mat, intD_inv, intD_inv_zz, intD_inv_zx_d, intD_inv_x_Cz, omega_star = self.get_inv_lb(t, iu)
                self.R_mat[iu] += R_mat
                self.omega_star[t-1, iu] = omega_star
                self.intD_inv_zz += intD_inv_zz
                self.intD_inv_zx_d += intD_inv_zx_d
                self.intD_inv_x_Cz += intD_inv_x_Cz
                self.intD_inv += intD_inv 
                log_lb, omega_dagger = self.get_log_lb(t, iu)
                self.log_lb[iu] += log_lb
                self.omega_dagger[t-1, iu] = omega_dagger

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
        while not converged:
            U_old = numpy.copy(self.U)
            for iu in range(self.Du):
                U_not_i = numpy.delete(self.U, [iu], axis=1)
                V = self.partial_gs(U_not_i)
                VRV = numpy.dot(numpy.dot(V.T, self.R_mat[iu]), V)
                alpha = scipy.linalg.eigh(VRV, eigvals=(VRV.shape[0]-1,VRV.shape[0]-1))[1]
                u_new = numpy.dot(V, alpha)[:,0]
                self.U[:,iu] = u_new
            converged = numpy.sum(numpy.abs(self.U - U_old)) < 1e-4
            
    def update_C(self):
        A = numpy.sum(self.ks.smoothing_density.integrate('xx')[1:], axis=0) / self.ks.sigma_x ** 2 - self.Sigma_inv_zz
        B = numpy.dot((self.ks.X - self.d).T, self.ks.smoothing_density.integrate('x')[1:])
        B -= self.intD_inv_zx_d.T
        self.C = numpy.linalg.solve(A, B.T).T
        
    def update_d(self):
        denominator = self.T / self.sigma_x ** 2 - self.intD_inv
        nominator = numpy.sum(self.ks.X - numpy.dot(self.ks.smoothing_density.integrate('x')[1:], self.C.T), axis=0) / self.sigma_x ** 2
        nominator -= self.intD_inv_x_Cz
        self.d = nominator / denominator
        
            
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