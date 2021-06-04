import numpy, scipy
import sys
sys.path.append('../src/')
import densities, conditionals

class KalmanFilter:
    
    def __init__(self, X: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, Qz: numpy.ndarray, 
                 C: numpy.ndarray, d: numpy.ndarray, Qx: numpy.ndarray):
        """ This is a linear Kalman filter.

        
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
        :param Qx: numpy.ndarray [Dx, Dx]
            The observation covariances.
            
        """
        self.Dz, self.Dx = Qz.shape[0], Qx.shape[0]
        self.T = X.shape[0]
        self.X = X
        self.state_density = conditionals.ConditionalGaussianDensity(numpy.array([A]), numpy.array([b]), numpy.array([Qz]))
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([C]), numpy.array([d]), numpy.array([Qx]))
        self.prediction_density = self._setup_density()
        self.filter_density = self._setup_density()
        
    def _setup_density(self, D: int=None) -> densities.GaussianDensity:
        """ Initializes a density object (with uniform densities).
        """
        if D is None:
            D = self.Dz
        Sigma = numpy.tile(numpy.eye(D)[None], (self.T+1,1,1))
        Lambda = numpy.tile(numpy.eye(D)[None], (self.T+1,1,1))
        mu = numpy.zeros((self.T + 1, D))
        ln_det_Sigma = D * numpy.log(numpy.ones(self.T+1))
        return densities.GaussianDensity(Sigma, mu, Lambda, ln_det_Sigma)
        
        
    def forward_path(self):
        """ Forward iteration.
        """
        for t in range(1, self.T+1):
            self.prediction(t)
            self.filtering(t)
        
        
    def prediction(self, t: int):
        """ Here the prediction density is calculated.
        
        p(z_t|x_{1:t-1}) = int p(z_t|z_t-1)p(z_t-1|x_1:t-1) dz_t-1
        
        :param t: int
            Time index.
        """
        # p(z_t-1|x_{1:t-1})
        pre_filter_density = self.filter_density.slice([t-1])
        # p(z_t|x_{1:t-1})
        cur_prediction_density = pre_filter_density.affine_marginal_transformation(self.state_density)
        # Write result into prediction density collection
        self.prediction_density.update([t], cur_prediction_density)
        
        
    def filtering(self, t: int):
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param t: int
            Time index.
        """
        # p(z_t|x_{1:t-1})
        cur_prediction_density = self.prediction_density.slice([t])
        # p(z_t| x_t, x_{1:t-1})
        p_z_given_x = cur_prediction_density.affine_conditional_transformation(self.emission_density)
        # Condition on x_t
        cur_filter_density = p_z_given_x.condition_on_x(self.X[t-1:t])
        # Write result into filter density collection
        self.filter_density.update([t], cur_filter_density)
        
    def compute_log_likelihood(self) -> float:
        """ Computes the log-likelihood of the model, given by
        
        $$
        \ell = \sum_t \ln p(x_t|x_{1:t-1}).
        $$
        """
        llk = 0
        px = self.prediction_density.affine_marginal_transformation(self.emission_density)
        for t in range(1,self.T+1):
            cur_px = px.slice([t])
            llk += cur_px.evaluate_ln(self.X[t-1:t])[0,0]
        return llk
    
    def compute_data_density(self) -> densities.GaussianDensity:
        px = self.prediction_density.affine_marginal_transformation(self.emission_density)
        return px.slice(numpy.arange(1, self.T+1))
            
class KalmanSmoother(KalmanFilter):
    
    def __init__(self, X: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, Qz: numpy.ndarray, 
                 C: numpy.ndarray, d: numpy.ndarray, Qx: numpy.ndarray):
        """ This is a linear Kalman smoother, which also iterates backwards in time.
        
        
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
        :param Qx: numpy.ndarray [Dx, Dx]
            The observation covariances.
        """
            
        super().__init__(X, A, b, Qz, C, d, Qx)
        self.smoothing_density = self._setup_density()
        # First dimension z_{t+1}, second z_t (Note that this can become expensive if Dz is large)
        self.twostep_smoothing_density = self._setup_density(D= int(2*self.Dz))
        self.twostep_smoothing_density = self.twostep_smoothing_density.slice(range(self.T))
        
    def backward_path(self):
        """ Backward iteration.
        """
        last_filter_density = self.filter_density.slice([self.T])
        self.smoothing_density.update([self.T], last_filter_density)
        
        for t in numpy.arange(self.T-1,-1,-1):
            self.smoothing(t)
                    
    def smoothing(self, t: int):
        """ Here we do the smoothing step.
        
        First we calculate the backward density
        
        $$
        p(z_{t} | z_{t+1}, x_{1:t}) = p(z_{t+1}|z_t)p(z_t | x_{1:t}) / p(z_{t+1}| x_{1:t}) 
        $$
        
        and finally we get the smoothing density
        
        $$
        p(z_{t} | x_{1:T}) = int p(z_{t} | z_{t+1}, x_{1:t}) p(z_{t+1}|x_{1:T}) dz_{t+1}
        $$
        
        :param t: int
            Time index.
        """
        # p(z_{t} | x_{1:t}) 
        cur_filter_density = self.filter_density.slice([t])
        # p(z_{t} | z_{t+1}, x_{1:t}) 
        backward_density = cur_filter_density.affine_conditional_transformation(self.state_density)
        # p(z_{t+1} | x_{1:T})
        post_smoothing_density = self.smoothing_density.slice([t+1])
        # p(z_{t} | x_{1:T})
        cur_smoothing_density = post_smoothing_density.affine_marginal_transformation(backward_density)
        # p(z_{t}, z_{t+1} | x_{1:T})
        cur_two_step_smoothing_density = post_smoothing_density.affine_joint_transformation(backward_density)
        # Write result into smoothing density collection
        self.smoothing_density.update([t], cur_smoothing_density)
        self.twostep_smoothing_density.update([t], cur_two_step_smoothing_density)
        
        
class StateSpace_EM:
    
    def __init__(self, X: numpy.ndarray, Dz: int, noise_x: float=.1, noise_z: float=.1):
        """ This object implements a linear state-space model and optimizes it according to the 
            expectation-maximization (EM) pocedure.
            
        :param X: numpy.ndarray [N, Dx]
            The observed data.
        :param Dz: int
            Number of latent dimensions.
        :param noise_x: float
            Initial standard deviation of observations. (Default=1e-1)
        :param noise_z: float
            Initial standard deviation of state transition matrix. (Default=1e-1) 
        """
        self.T, self.Dx = X.shape
        self.Dz = Dz
        self.Qx = noise_x ** 2 * numpy.eye(self.Dx)
        self.Qz = noise_z ** 2 * numpy.eye(self.Dz)
        self.A, self.b = numpy.eye(self.Dz), numpy.zeros((self.Dz,))
        self.d = numpy.mean(X, axis=0)
        X_smoothed = numpy.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = numpy.convolve(X[:,i], numpy.ones(10) / 10., mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(numpy.dot((X_smoothed-self.d[None]).T, X_smoothed-self.d[None]), eigvals=(self.Dx-self.Dz, self.Dx-1))
        self.C =  eig_vecs * eig_vals / self.T
        z_hat = numpy.dot(numpy.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - numpy.dot(z_hat, self.C.T) - self.d
        self.Qx = numpy.dot(delta_X.T, delta_X)
        self.ks = KalmanSmoother(X, self.A, self.b, self.Qz, self.C, self.d, self.Qx)
        self.Qz_inv, self.ln_det_Qz = self.ks.state_density.Lambda[0], self.ks.state_density.ln_det_Sigma[0]
        self.Qx_inv, self.ln_det_Qx = self.ks.emission_density.Lambda[0], self.ks.emission_density.ln_det_Sigma[0]
        
    def run_em(self, n_iter: int=100, conv_crit: float=1e-4) -> list:
        """ Runs the EM procedure.
        
        :param n_iter: int
            Number of maximal iterations. (Default=100)
        :param conv_crit: float
            Convergence criterion. If relative loglikelihood improvement is below EM stops. (Default=1e-4)
            
        :return: list
            List with log likelihood at each iteration.
        """
        llk_list = []
        converged = False
        self.iteration = 0
        while self.iteration < n_iter and not converged:
            self.estep()
            llk_list.append(self.compute_log_likelihood())
            self.mstep()
            if self.iteration>1:
                conv = (llk_list[-1] - llk_list[-2]) / numpy.amax([1, numpy.abs(llk_list[-1]), numpy.abs(llk_list[-2])])
                converged = conv < conv_crit
            self.iteration += 1
            print('Iteration %d - llk=%.1f' %(self.iteration,llk_list[-1]))
        return llk_list
    
    def estep(self):
        """ Performs the expectation step, i.e. the forward-backward algorithm.
        """
        self.ks.forward_path()
        self.ks.backward_path()
        
    def mstep(self):
        """ Performs the maximization step, i.e. updates all model parameters.
        """
        self.update_A()
        self.update_b()
        self.update_Qz()
        self.update_state_density()
        self.update_C()
        self.update_d()
        self.update_Qx()
        self.update_emission_density()
        self.update_init_density()
        
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
        B = numpy.dot(self.Qx_inv, A)
        b_t = numpy.dot(self.Qx_inv, a_t.T).T
        Exx = 0
        for t in range(1,self.T+1):
            cur_smooth_density = self.ks.smoothing_density.slice([t])
            Exx += cur_smooth_density.integrate('Ax_aBx_b_inner', A_mat=A, a_vec=a_t[t-1], B_mat=B, b_vec=b_t[t-1])[0]
        Exx += self.T * (self.ln_det_Qx + self.Dx * numpy.log(2 * numpy.pi))
        # E[(z_0 - mu0)'Sigma0^{-1}(z_0 - mu0)]
        init_smooth_density = self.ks.smoothing_density.slice([0])
        A = numpy.eye(self.Dz)
        a = -self.ks.filter_density.mu[0]
        B = self.ks.filter_density.Lambda[0]
        b = numpy.dot(self.ks.filter_density.Lambda[0], a)
        Ez0 = init_smooth_density.integrate('Ax_aBx_b_inner', A_mat=A, a_vec=a, B_mat=B, b_vec=b)[0]
        Ez0 += self.ks.filter_density.ln_det_Sigma[0] + self.Dz * numpy.log(2 * numpy.pi)
        return - .5 * (Exx + Ezz + Ez0)
    
    def update_A(self):
        """ Computes the optimal state transition matrix.
        """
        Ezz = self.ks.smoothing_density.integrate('xx')
        mu_b = self.ks.smoothing_density.mu[:-1,None] * self.b[None,:,None]
        M_fp = numpy.einsum('ab,cbd->cad', self.A, self.ks.filter_density.Sigma[:-1])
        M_l = numpy.einsum('abc,acd->abd', M_fp, self.ks.prediction_density.Lambda[1:])
        Ezz_cross = numpy.einsum('abc,acd->abd', M_l, self.ks.smoothing_density.Sigma[1:])
        Ezz_cross += self.ks.smoothing_density.mu[:-1,None] * self.ks.smoothing_density.mu[1:,:,None]
        self.A = numpy.linalg.solve(numpy.sum(Ezz[:-1], axis=0), numpy.sum(Ezz_cross -  mu_b, axis=0)).T
        
    def update_b(self):
        """ Computes the optimal state offset.
        """
        self.b = numpy.mean(self.ks.smoothing_density.mu[1:] - numpy.dot(self.A, self.ks.smoothing_density.mu[:-1].T).T, axis=0)
    
    def update_Qz(self):
        """ Computes the optimal state covariance.
        """
        M_fp = numpy.einsum('ab,cbd->cad', self.A, self.ks.filter_density.Sigma[:-1])
        M_l = numpy.einsum('abc,acd->abd', M_fp, self.ks.prediction_density.Lambda[1:])
        Ezz_cross = numpy.einsum('abc,acd->abd', M_l, self.ks.smoothing_density.Sigma[1:])
        Ezz_cross += self.ks.smoothing_density.mu[:-1,None] * self.ks.smoothing_density.mu[1:,:,None]
        Ezz = self.ks.smoothing_density.integrate('xx')
        AEzz_cross = numpy.sum(numpy.einsum('ab,cbd->cad', self.A, Ezz_cross), axis=0)
        Ez_b = self.ks.smoothing_density.mu[:,None] * self.b[None,:,None]
        AEz_b = numpy.sum(numpy.einsum('ab,cbd->cad', self.A, Ez_b[:-1]), axis=0)
        Az_b2 = numpy.sum(self.ks.smoothing_density.integrate('Ax_aBx_b_outer', A_mat=self.A, a_vec=self.b, B_mat=self.A, b_vec=self.b)[:-1],axis=0)
        mu_b = numpy.sum(Ez_b[1:], axis=0)
        AEzzA = numpy.sum(numpy.einsum('abc,cd->abd', numpy.einsum('ab,cbd->cad', self.A, Ezz[:-1]), self.A.T), axis=0)
        self.Qz = (numpy.sum(Ezz[1:], axis=0) + Az_b2 - mu_b - mu_b.T - AEzz_cross - AEzz_cross.T) / self.T
        
    def update_state_density(self):
        """ Updates the state density.
        """
        self.ks.state_density = conditionals.ConditionalGaussianDensity(numpy.array([self.A]), 
                                                                        numpy.array([self.b]), 
                                                                        numpy.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.ks.state_density.Lambda[0], self.ks.state_density.ln_det_Sigma[0]
        
    def update_Qx(self):
        """ Computes the optimal observation covariance.
        """
        A = -self.C
        a_t = self.ks.X - self.d[None]
        Exx = numpy.zeros((self.Dx, self.Dx))
        for t in range(1,self.T+1):
            cur_smooth_density = self.ks.smoothing_density.slice([t])
            Exx += cur_smooth_density.integrate('Ax_aBx_b_outer', A_mat=A, a_vec=a_t[t-1], B_mat=A, b_vec=a_t[t-1])[0]
        self.Qx = Exx / self.T
        
    def update_C(self):
        """ Computes the optimal observation matrix.
        """
        Ezz = numpy.sum(self.ks.smoothing_density.integrate('xx')[1:], axis=0)
        Ez = self.ks.smoothing_density.integrate('x')[1:]
        zx = numpy.sum(Ez[:,:,None] * (self.ks.X[:,None] - self.d[None,None]), axis=0)
        self.C = numpy.linalg.solve(Ezz, zx).T
        
    def update_d(self):
        """ Computes the optimal observation offset.
        """
        Ez = self.ks.smoothing_density.integrate('x')[1:]
        self.d = numpy.mean(self.ks.X - numpy.dot(self.C, Ez.T).T, axis=0)
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.ks.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([self.C]),
                                                                           numpy.array([self.d]), 
                                                                           numpy.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.ks.emission_density.Lambda[0], self.ks.emission_density.ln_det_Sigma[0]
        
    def update_init_density(self):
        """ Computes the optimal initial state distribution.
        """
        init_smooth_density = self.ks.smoothing_density.slice([0])
        mu0 = init_smooth_density.integrate('x')
        Sigma0 = init_smooth_density.integrate('Ax_aBx_b_outer', A_mat=None, a_vec=-mu0[0], B_mat=None, b_vec=-mu0[0])
        opt_init_density = densities.GaussianDensity(Sigma0, mu0)
        self.ks.filter_density.update([0], opt_init_density)
        
    def compute_log_likelihood(self) -> float:
        """ Computes the log-likelihood of the model.
        """
        return self.ks.compute_log_likelihood()
        