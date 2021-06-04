import numpy, scipy
import sys
sys.path.append('../src/')
import densities, conditionals, factors
from linear_ssm import KalmanFilter, StateSpace_EM


class NonLinearGaussianConditional(conditionals.ConditionalGaussianDensity):
    
    def __init__(self, M: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray, Sigma: numpy.ndarray=None, Lambda: numpy.ndarray=None, ln_det_Sigma: numpy.ndarray=None):
        """ A conditional Gaussian density
            
            p(y|x) = N(mu(x), Sigma)
            
            with the conditional mean function mu(x) = M f(x) + b. 
            phi(x) is a feature vector of the form
            
            f(x) = (1,x_1,...,x_m,k(h_1(x)),...,k(h_n(x))),
            
            with
            
            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.
            
            :param M: numpy.ndarray [1, Dy, Dphi]
                Matrix in the mean function.
            :param b: numpy.ndarray [1, Dy]
                Vector in the conditional mean function.
            :param W: numpy.ndarray [Dphi, Dx + 1]
                Parameters for linear mapping in the nonlinear functions
            :param Sigma: numpy.ndarray [1, Dy, Dy]
                The covariance matrix of the conditional. (Default=None)
            :param Lambda: numpy.ndarray [1, Dy, Dy] or None
                Information (precision) matrix of the Gaussians. (Default=None)
            :param ln_det_Sigma: numpy.ndarray [1] or None
                Log determinant of the covariance matrix. (Default=None)
        """
        super().__init__(M, b, Sigma, Lambda, ln_det_Sigma)
        self.w0 = W[:,0]
        self.W = W[:,1:]
        self.Dx = self.W.shape[1]
        self.Dk = self.W.shape[0]
        self.Dphi = self.Dk + self.Dx
        self._setup_phi()
        
    def _setup_phi(self):
        """ Sets up the non-linear kernel function in phi(x).
        """
        v = self.W
        nu = self.W * self.w0[:,None]
        ln_beta = - .5 * self.w0 ** 2
        self.k_func = factors.OneRankFactor(v=v, nu=nu, ln_beta=ln_beta)
        
    def evaluate_f(self, x: numpy.ndarray):
        """ Evaluates the f
        
        f(x) = (0,x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).
        
        :param x: numpy.ndarray [N, Dx]
            Points where phi should be evaluated.
            
        :return: numpy.ndarray [N, Dphi]
            Deature vector.
        """
        N = x.shape[0]
        f_x = numpy.empty((N, self.Dphi))
        f_x[:,:self.Dx] = x
        f_x[:,self.Dx:] = self.k_func.evaluate(x)
        return f_x
    
    def get_conditional_mu(self, x: numpy.ndarray) -> numpy.ndarray:
        """ Computest the conditional mu function
        
            mu(x) = mu(x) = M f(x) + b
            
        :param x: numpy.ndarray [N, Dx]
            Instances, the mu should be conditioned on.
        
        :return: numpy.ndarray [1, N, Dy]
            Conditional means.
        """
        f_x = self.evaluate_f(x)
        mu_y = numpy.einsum('ab,cb->ca', self.M[0], f_x) + self.b[0][None]
        return mu_y
    
    def get_expected_moments(self, phi_x: 'GaussianDensity') -> numpy.ndarray:
        """ Computes the expected covariance
        
            Sigma_y = E[yy'] - E[y]E[y]'
            
        :param phi_x: GaussianDensity
            The density which we average over.

        :return: numpy.ndarray [phi_R, Dy, Dy]
            Returns the expected mean
        """
        
        #### E[f(x)] ####
        # E[x] [R, Dx]
        Ex = phi_x.integrate('x')
        # E[k(x)] [R, Dphi - Dx]
        phi_k = phi_x.multiply(self.k_func)
        Ekx = phi_k.integrate().reshape((phi_x.R, self.Dphi - self.Dx))
        # E[f(x)]
        Ef = numpy.concatenate([Ex, Ekx], axis=1)
        
        #### E[f(x)f(x)'] ####
        Eff = numpy.empty([phi_x.R, self.Dphi, self.Dphi])
        # Linear terms E[xx']
        Eff[:,:self.Dx,:self.Dx] = phi_x.integrate('xx')
        # Cross terms E[x k(x)']
        Ekx = phi_k.integrate('x').reshape((phi_x.R, self.Dk, self.Dx))
        Eff[:,:self.Dx,self.Dx:] = numpy.swapaxes(Ekx, axis1=1, axis2=2)
        Eff[:,self.Dx:,:self.Dx] = Ekx
        # kernel terms E[k(x)k(x)']
        Ekk = phi_x.multiply(self.k_func).multiply(self.k_func).integrate().reshape((phi_x.R, self.Dk, self.Dk))
        Eff[:,self.Dx:,self.Dx:] = Ekk
        
        ### mu_y = E[mu(x)] = ME[f(x)] + b ###
        mu_y = numpy.einsum('ab,cb->ca', self.M[0], Ef) + self.b[0][None]
        
        ### Sigma_y = E[yy'] - mu_ymu_y' = Sigma + E[mu(x)mu(x)'] - mu_ymu_y'
        #                                = Sigma + ME[f(x)f(x)']M' + bE[f(x)']M' + ME[f(x)]b' + bb' - mu_ymu_y'
        Sigma_y = numpy.tile(self.Sigma, (phi_x.R, 1, 1))
        Sigma_y += numpy.einsum('ab,cbd->cad', self.M[0], numpy.einsum('abc,dc->abd', Eff, self.M[0]))
        MEfb = numpy.einsum('ab,c->abc', numpy.einsum('ab,cb->ca', self.M[0], Ef), self.b[0])
        Sigma_y += MEfb + numpy.swapaxes(MEfb, axis1=1, axis2=2)
        Sigma_y += (self.b[0,None] * self.b[0,:,None])[None]
        Sigma_y -= mu_y[:,None] * mu_y[:,:,None]
        return mu_y, Sigma_y
    
    def get_expected_cross_terms(self, phi_x: 'GaussianDensity') -> numpy.ndarray:
        """ Computes
        
            E[yx'] = \int\int yx' p(y|x)phi(x) dydx = int (M f(x) + b)x' phi(x) dx
            
        :param phi_x: GaussianDensity
            The density which we average over.

        :return: numpy.ndarray [phi_R, Dx, Dy]
            Returns the cross expectations.
        """
        
        # E[xx']
        Exx = phi_x.integrate('xx')
        # E[k(x)x']
        Ekx = phi_x.multiply(self.k_func).integrate('x').reshape((phi_x.R, self.Dk, self.Dx))
        # E[f(x)x']
        Ef_x = numpy.concatenate([Exx, Ekx], axis=1)
        # M E[f(x)x']
        MEf_x = numpy.einsum('ab,cbd->cad', self.M[0], Ef_x)
        # bE[x']
        bEx = self.b[0][None,:,None] * phi_x.integrate('x')[:,None]
        # E[yx']
        Eyx = MEf_x + bEx
        return Eyx
        
    def affine_joint_moment_matching(self, phi_x: 'GaussianDensity') -> 'GaussianDensity':
        """ Gets an approximation of the joint density
        
            p(x,y) ~= N(mu_{xy},Sigma_{xy}),
            
        The mean is given by
            
            mu_{xy} = (mu_x, mu_y)'
            
        with mu_y = E[mu_y(x)]. The covariance is given by
            
            Sigma_{xy} = (Sigma_x            E[xy'] - mu_xmu_y'
                          E[yx'] - mu_ymu_x' E[yy'] - mu_ymu_y').
                          
        :param phi_x: GaussianDensity
            Marginal Gaussian density over x.
        
        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        mu_y, Sigma_y = self.get_expected_moments(phi_x)
        Eyx = self.get_expected_cross_terms(phi_x)
        mu_x = phi_x.mu
        cov_yx = Eyx - mu_y[:,:,None] * mu_x[:,None]
        mu_xy = numpy.concatenate([mu_x, mu_y], axis=1)
        Sigma_xy = numpy.empty((phi_x.R, self.Dy + self.Dx, self.Dy + self.Dx))
        Sigma_xy[:,:self.Dx,:self.Dx] = phi_x.Sigma
        Sigma_xy[:,self.Dx:,:self.Dx] = cov_yx
        Sigma_xy[:,:self.Dx,self.Dx:] = numpy.swapaxes(cov_yx, axis1=1, axis2=2)
        Sigma_xy[:,self.Dx:,self.Dx:] = Sigma_y
        phi_xy = densities.GaussianDensity(Sigma=Sigma_y, mu=mu_y)
        return phi_xy
    
    def affine_conditional_moment_matching(self, phi_x: 'GaussianDensity') -> 'ConditionalGaussianDensity':
        """ Gets an approximation of the joint density via moment matching
        
            p(x|y) ~= N(mu_{x|y},Sigma_{x|y}),
    
        :param phi_x: GaussianDensity
            Marginal Gaussian density over x.
        
        :return: ConditionalDensity
            Returns the conditional density of x given y.
        """
        mu_y, Sigma_y = self.get_expected_moments(phi_x)
        Lambda_y = self.invert_matrix(Sigma_y)[0]
        Eyx = self.get_expected_cross_terms(phi_x)
        mu_x = phi_x.mu
        cov_yx = Eyx - mu_y[:,:,None] * mu_x[:,None]
        M_new = numpy.einsum('abc,abd->acd', cov_yx, Lambda_y)
        b_new = mu_x - numpy.einsum('abc,ac->ab', M_new, mu_y)
        Sigma_new = phi_x.Sigma - numpy.einsum('abc,acd->abd', M_new, cov_yx)
        cond_phi_xy = conditionals.ConditionalGaussianDensity(M=M_new, b=b_new, Sigma=Sigma_new)
        return cond_phi_xy
        
    def affine_marginal_moment_matching(self, phi_x: 'GaussianDensity') -> 'GaussianDensity':
        """ Gets an approximation of the marginal density
        
            p(y) ~= N(mu_y,Sigma_y),
            
        The mean is given by
            
            mu_y = E[mu_y(x)]. 
            
        The covariance is given by
            
            Sigma_y = E[yy'] - mu_ymu_y'.
                          
        :param phi_x: GaussianDensity
            Marginal Gaussian density over x.
        
        :return: GaussianDensity
            Returns the joint distribution of x,y.
        """
        
        mu_y, Sigma_y = self.get_expected_moments(phi_x)
        phi_y = densities.GaussianDensity(Sigma=Sigma_y, mu=mu_y)
        return phi_y

    
class NonLinearKalmanFilter(KalmanFilter):
    
    def __init__(self, X: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray, Qz: numpy.ndarray, 
                 C: numpy.ndarray, d: numpy.ndarray, Qx: numpy.ndarray):
        """ This is a non-linear Kalman filter, where the mean of the transition function is a sum of linear and RBF terms.

        
        :param X: numpy.ndarray [N, Dx]
            The observed data.
        :param A: numpy.ndarray [Dz, Dz]
            The state transition matrix.
        :param b: numpy.ndarray [Dz]
            The state transition offset.
        :param W: numpy.ndarray [Dphi, Dx + 1]
            Parameters for linear mapping in the nonlinear functions
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
        self.state_density = NonLinearGaussianConditional(M=numpy.array([A]), b=numpy.array([b]), W=W, Sigma=numpy.array([Qz]))
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([C]), numpy.array([d]), numpy.array([Qx]))
        self.prediction_density = self._setup_density()
        self.filter_density = self._setup_density()
        
        
    def prediction(self, t: int):
        """ Here the approximate prediction density is calculated, via moment matching
        
        p(z_t|x_{1:t-1}) = N(mu^p_t, Sigma^p_t)
        
        with 
            mu_t^p = \int\int  z_t p(z_t|z_t-1)p(z_t-1|x_1:t-1) d z_t d z_t-1
            Sigma_t^p = \int\int  z_tz_t' p(z_t|z_t-1)p(z_t-1|x_1:t-1) d z_t d z_t-1 - mu_t^pmu_t^p'
        
        :param t: int
            Time index.
        """
        # p(z_t-1|x_{1:t-1})
        pre_filter_density = self.filter_density.slice([t-1])
        # p(z_t|x_{1:t-1})
        cur_prediction_density = self.state_density.affine_marginal_moment_matching(pre_filter_density)
        # Write result into prediction density collection
        self.prediction_density.update([t], cur_prediction_density)
        
        
class NonLinearKalmanSmoother(NonLinearKalmanFilter):
    
    
    def __init__(self, X: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray, Qz: numpy.ndarray, 
                 C: numpy.ndarray, d: numpy.ndarray, Qx: numpy.ndarray):
        
        super().__init__(X, A, b, W, Qz, C, d, Qx)
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
        
        First we appoximate the backward density by moment matching
        
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
        backward_density = self.state_density.affine_conditional_moment_matching(cur_filter_density)
        # p(z_{t+1} | x_{1:T})
        post_smoothing_density = self.smoothing_density.slice([t+1])
        # p(z_{t} | x_{1:T})
        cur_smoothing_density = post_smoothing_density.affine_marginal_transformation(backward_density)
        # p(z_{t}, z_{t+1} | x_{1:T})
        cur_two_step_smoothing_density = post_smoothing_density.affine_joint_transformation(backward_density)
        # Write result into smoothing density collection
        self.smoothing_density.update([t], cur_smoothing_density)
        self.twostep_smoothing_density.update([t], cur_two_step_smoothing_density)
        

class NonLinearStateSpace_EM(StateSpace_EM):
    
    def __init__(self, X: numpy.ndarray, Dz: int, Dk: int, noise_x: float=.1, noise_z: float=.1):
        """ This object implements a non linear state-space model and optimizes it according to the 
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
        self.Dk = Dk
        self.Dphi = Dk + Dz
        self.Qx = noise_x ** 2 * numpy.eye(self.Dx)
        self.Qz = noise_z ** 2 * numpy.eye(self.Dz)
        self.A = numpy.zeros([self.Dz, self.Dphi])
        self.A[:,:self.Dz] = numpy.eye(self.Dz)
        self.b = numpy.zeros((self.Dz,))
        self.d = numpy.mean(X, axis=0)
        X_smoothed = numpy.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = numpy.convolve(X[:,i], numpy.ones(10) / 10., mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(numpy.dot((X_smoothed-self.d[None]).T, X_smoothed-self.d[None]), eigvals=(self.Dx-self.Dz, self.Dx-1))
        self.C =  eig_vecs * eig_vals / self.T
        z_hat = numpy.dot(numpy.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - numpy.dot(z_hat, self.C.T) - self.d
        self.Qx = numpy.dot(delta_X.T, delta_X)
        self.W = numpy.random.randn(self.Dk, self.Dz + 1)
        self.ks = NonLinearKalmanSmoother(X, self.A, self.b, self.W, self.Qz, self.C, self.d, self.Qx)
        self.Qz_inv, self.ln_det_Qz = self.ks.state_density.Lambda[0], self.ks.state_density.ln_det_Sigma[0]
        self.Qx_inv, self.ln_det_Qx = self.ks.emission_density.Lambda[0], self.ks.emission_density.ln_det_Sigma[0]
        
    def mstep(self):
        """ Performs the maximization step, i.e. updates all model parameters.
        """
        if self.iteration % 3 == 0:
            self.update_A()
        elif self.iteration % 3 == 1:
            self.update_b()
        else:
            self.update_Qz()
        self.update_W()
        #
        self.update_state_density()
        self.update_C()
        self.update_d()
        self.update_Qx()
        self.update_emission_density()
        self.update_init_density()
    
    def update_A(self):
        """ Computes the optimal state transition matrix.
        """
        phi = self.ks.smoothing_density.slice(range(self.T))
        
        # E[f(z)f(z)']
        Ekk = phi.multiply(self.ks.state_density.k_func).multiply(self.ks.state_density.k_func).integrate().reshape((self.T, self.Dk, self.Dk))
        Ekz = phi.multiply(self.ks.state_density.k_func).integrate('x').reshape((self.T, self.Dk, self.Dz))
        Eff = numpy.empty((self.Dphi, self.Dphi))
        Eff[:self.Dz,:self.Dz] = numpy.sum(phi.integrate('xx'), axis=0)
        Eff[self.Dz:,self.Dz:] = numpy.sum(Ekk, axis=0)
        Eff[self.Dz:,:self.Dz] = numpy.sum(Ekz, axis=0)
        Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        # E[f(z)] b'
        Ez = numpy.sum(phi.integrate('x'), axis=0)
        Ek = numpy.sum(phi.multiply(self.ks.state_density.k_func).integrate().reshape((self.T,self.Dk)), axis=0)
        Ef = numpy.concatenate([Ez, Ek])
        Ebf = Ef[None] * self.b[:,None]
        # E[z f(z)']
        v_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        v_joint[:,self.Dz:] = self.ks.state_density.k_func.v
        nu_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        nu_joint[:,self.Dz:] = self.ks.state_density.k_func.nu
        ln_beta = self.ks.state_density.k_func.ln_beta
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=ln_beta)
        Ezz_cross = numpy.sum(self.ks.twostep_smoothing_density.integrate('xx')[:,self.Dz:,:self.Dz], axis=0)
        Ezk = numpy.sum(self.ks.twostep_smoothing_density.multiply(joint_k_func).integrate('x').reshape((self.T,self.Dk,(2*self.Dz)))[:,:,:self.Dz], axis=0).T
        Ezf = numpy.concatenate([Ezz_cross, Ezk], axis=1)
        self.A = numpy.linalg.solve(Eff, (Ezf -  Ebf).T).T
        
    def update_b(self):
        """ Computes the optimal state offset.
        """
        Ez = self.ks.smoothing_density.integrate('x')
        Ek = self.ks.smoothing_density.multiply(self.ks.state_density.k_func).integrate().reshape((self.T+1,self.Dk))
        Ef = numpy.concatenate([Ez, Ek], axis=1)
        self.b = numpy.mean(self.ks.smoothing_density.mu[1:] - numpy.dot(self.A, Ef[:-1].T).T, axis=0)
        
    def update_W(self):
        pass
        
    def update_Qz(self):
        """ Computes the optimal state covariance.
        """
        Ezz = self.ks.smoothing_density.integrate('xx')
        # E[zz']
        Ezz_sum = numpy.sum(Ezz[1:], axis=0)
        # E[z f(z)'] A'
        v_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        v_joint[:,self.Dz:] = self.ks.state_density.k_func.v
        nu_joint = numpy.zeros([self.Dk, int(2 * self.Dz)])
        nu_joint[:,self.Dz:] = self.ks.state_density.k_func.nu
        joint_k_func = factors.OneRankFactor(v=v_joint, nu=nu_joint, ln_beta=self.ks.state_density.k_func.ln_beta)
        Ezz_cross = self.ks.twostep_smoothing_density.integrate('xx')[:,self.Dz:,:self.Dz]
        Ekz = self.ks.twostep_smoothing_density.multiply(joint_k_func).integrate('x').reshape((self.T,self.Dk,int(2*self.Dz)))[:,:,:self.Dz]
        Ezf = numpy.concatenate([Ezz_cross, numpy.swapaxes(Ekz,1,2)], axis=2)
        EzfA = numpy.einsum('abc,dc->bd', Ezf, self.A)
        # E[z]b'
        Ezb = numpy.sum(self.ks.smoothing_density.integrate('x')[1:,None] * self.b[None,:,None], axis=0)
        # A E[f(z)] b'
        Ez = numpy.sum(self.ks.smoothing_density.integrate('x')[:-1], axis=0)
        Ek = numpy.sum(self.ks.smoothing_density.multiply(self.ks.state_density.k_func).integrate().reshape((self.T+1,self.Dk))[:-1], axis=0)
        Ef = numpy.concatenate([Ez, Ek])
        AEfb = numpy.dot(self.A, Ef)[:,None] * self.b[None]
        # A E[f(z)f(z)'] A'
        Ekk = self.ks.smoothing_density.multiply(self.ks.state_density.k_func).multiply(self.ks.state_density.k_func).integrate().reshape((self.T+1, self.Dk, self.Dk))
        Ekz = self.ks.smoothing_density.multiply(self.ks.state_density.k_func).integrate('x').reshape((self.T+1, self.Dk, self.Dz))
        Eff = numpy.empty((self.Dphi, self.Dphi))
        Eff[:self.Dz,:self.Dz] = numpy.sum(Ezz[:-1], axis=0)
        Eff[self.Dz:,self.Dz:] = numpy.sum(Ekk[:-1], axis=0)
        Eff[self.Dz:,:self.Dz] = numpy.sum(Ekz[:-1], axis=0)
        Eff[:self.Dz,self.Dz:] = Eff[self.Dz:,:self.Dz].T
        AEffA = numpy.dot(numpy.dot(self.A, Eff), self.A.T)
        self.Qz = (Ezz_sum - EzfA - EzfA.T + AEffA - Ezb - Ezb.T + AEfb + AEfb.T + self.T * self.b[:,None] * self.b[None]) / self.T
        
    def update_state_density(self):
        """ Updates the state density.
        """
        self.ks.state_density = NonLinearGaussianConditional(M=numpy.array([self.A]), b=numpy.array([self.b]), W=self.W, Sigma=numpy.array([self.Qz]))
        self.Qz_inv, self.ln_det_Qz = self.ks.state_density.Lambda[0], self.ks.state_density.ln_det_Sigma[0]
        