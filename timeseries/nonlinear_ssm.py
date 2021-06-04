import numpy
import sys
sys.path.append('../src/')
import densities, conditionals, factors
from linear_ssm import KalmanFilter


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
        
    def _setup_phi(self):
        """ Sets up the non-linear kernel function in phi(x).
        """
        v = self.W
        nu = self.W * self.w0
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
    
    def get_expected_cross_terms(self, phi_x: 'GaussianDensity') -> numpy.narray:
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
    
    def affine_conditional_moment_matching(self, phi: 'GaussianDensity') -> 'ConditionalGaussianDensity':
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
        
    def affine_marginal_moment_matching(self, phi: 'GaussianDensity') -> 'GaussianDensity':
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
        self.state_density = NonLinearGaussianConditional(M=numpy.array([A]), b=numpy.array([b]) W=W, Sigma=Qz)
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
        # Write result into smoothing density collection
        self.smoothing_density.update([t], cur_smoothing_density)
        