import numpy, scipy
import sys
sys.path.append('../src/')
import densities, conditionals

class ObservationModel:
    
    def __init__(self):
        """ This is the template class for observation models in state space models. 
        Basically these classes should contain all functionality for the mapping between 
        the latent variables z, and observations x, i.e. p(x_t|z_t). The object should 
        have an attribute `emission_density`, which is be a `ConditionalDensity`. 
        Furthermore, it should be possible to optimize hyperparameters, when provided 
        with a density over the latent space.
        """
        self.emission_density = None
    
    def filtering(self, prediction_density: 'GaussianDensity', x_t: numpy.ndarray) -> 'GaussianDensity':
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
        
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        raise NotImplementedError('Filtering for observation model not implemented.')
        
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        raise NotImplementedError('Hyperparameter updates for observation model not implemented.')

class LinearObservationModel(ObservationModel):
    
    def __init__(self, Dx: int, Dz: int, noise_x: float=1e-1):
        """ This class implements a linear observation model, where the observations are generated as
        
            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx).
            
        :param Dx: int
            Dimensionality of observations.
        :param Dz: int
            Dimensionality of latent space.
        :param noise_x: float
            Intial isoptropic std. on the observations.
        """
        self.Dx, self.Dz = Dx, Dz
        if Dx == Dz:
            self.C = numpy.eye(Dx)
        else:
            self.C = numpy.random.randn(Dx, Dz)
        self.d = numpy.zeros(Dx)
        self.Qx = noise_x ** 2 * numpy.eye(self.Dx)
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([self.C]), 
                                                                        numpy.array([self.d]), 
                                                                        numpy.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
        
    def pca_init(self, X: numpy.ndarray, smooth_window: int=10):
        self.d = numpy.mean(X, axis=0)
        T = X.shape[0]
        X_smoothed = numpy.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:,i] = numpy.convolve(X[:,i], 
                                             numpy.ones(smooth_window) / smooth_window, 
                                             mode='same')
        eig_vals, eig_vecs = scipy.linalg.eigh(numpy.dot((X_smoothed-self.d[None]).T, 
                                                         X_smoothed-self.d[None]), 
                                               eigvals=(self.Dx-self.Dz, self.Dx-1))
        self.C =  eig_vecs * eig_vals / T
        z_hat = numpy.dot(numpy.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - numpy.dot(z_hat, self.C.T) - self.d
        self.Qx = numpy.dot(delta_X.T, delta_X)
        
    def filtering(self, prediction_density: 'GaussianDensity', x_t: numpy.ndarray) -> 'GaussianDensity':
        """ Here the filtering density is calculated.
        
        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)
        
        :param prediction_density: GaussianDensity
            Prediction density p(z_t|x_{1:t-1}).
        :param x_t: numpy.ndarray [1, Dx]
        
        :return: GaussianDensity
            Filter density p(z_t|x_{1:t}).
        """
        # p(z_t| x_t, x_{1:t-1})
        p_z_given_x = self.emission_density.affine_conditional_transformation(prediction_density)
        # Condition on x_t
        cur_filter_density = p_z_given_x.condition_on_x(x_t)
        return cur_filter_density
    
    def update_hyperparameters(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the hyperparameters of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        self.update_C(smoothing_density, X)
        self.update_d(smoothing_density, X)
        self.update_Qx(smoothing_density, X)
        self.update_emission_density()
           
    def update_Qx(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the covariance of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        T = X.shape[0]
        A = -self.C
        a_t = X - self.d[None]
        Exx = numpy.zeros((self.Dx, self.Dx))
        for t in range(1, T+1):
            cur_smooth_density = smoothing_density.slice([t])
            Exx += cur_smooth_density.integrate('Ax_aBx_b_outer', A_mat=A, 
                                                a_vec=a_t[t-1], B_mat=A, 
                                                b_vec=a_t[t-1])[0]
        self.Qx = Exx / T
        
    def update_C(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the transition matrix of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        Ezz = numpy.sum(smoothing_density.integrate('xx')[1:], axis=0)
        Ez = smoothing_density.integrate('x')[1:]
        zx = numpy.sum(Ez[:,:,None] * (X[:,None] - self.d[None,None]), axis=0)
        self.C = numpy.linalg.solve(Ezz, zx).T
        
    def update_d(self, smoothing_density: 'GaussianDensity', X: numpy.ndarray):
        """ This procedure updates the transition offset of the observation model.
        
        :param smoothing_density: GaussianDensity
            The smoothing density over the latent space.
        :param X: numpy.ndarray [T, Dx]
            The observations.
        """  
        Ez = smoothing_density.integrate('x')[1:]
        self.d = numpy.mean(X - numpy.dot(self.C, Ez.T).T, axis=0)
        
    def update_emission_density(self):
        """ Updates the emission density.
        """
        self.emission_density = conditionals.ConditionalGaussianDensity(numpy.array([self.C]),
                                                                        numpy.array([self.d]),
                                                                        numpy.array([self.Qx]))
        self.Qx_inv, self.ln_det_Qx = self.emission_density.Lambda[0], self.emission_density.ln_det_Sigma[0]
    