import numpy
import sys
sys.path.append('../src/')
import densities, conditionals
from linear_ssm import KalmanFilter

class HeteroscedasticKalmanFilter(KalmanFilter):
    
    def __init__(self, X: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, Qz: numpy.ndarray, 
                 C: numpy.ndarray, d: numpy.ndarray, U: numpy.ndarray, W:  numpy.ndarray, beta: numpy.ndarray, 
                 sigma_x: float):
        """ This is a heteroscedastic Kalman filter, where the observation covariance is
        
        Sigma_x(t) = sigma^2 I + \sum_i U_i D_i(z_t) U_i',
        
        and D_i(z) = 2 * beta_i * cosh(w_i'z + b_i)
        
        
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
            
        :param W: numpy.ndarray [Dz + 1, D_noise]
        
        :param beta: numpy.ndarray [D_noise]
        
        :param sigma_x: float
        """
        pass
    
    