# %%
import sys
sys.path.append('../')
from jax import numpy as jnp
import numpy as np
from src_jax import factors, densities, conditionals
from matplotlib import pyplot as plt
from typing import Tuple

class LinearRegression:

    def __init__(self, X: jnp.ndarray, y: jnp.ndarray, sigma_y: float=.1,
                 mu_prior: float = 0, sigma_prior: float = 1., bias: bool=True):
        """Linear regression class, i.e.
        
            y = Xw + b + \epsilon with \epsilon \sim N(0,\sigma_y)
            
            and prior p(w) = N(\mu_w,\sigma_w^2).

        :param X: Input data [N x D]
        :type X: jnp.ndarray
        :param y: Target variable [N]
        :type y: jnp.ndarray
        :param sigma_y: Standard deviation of the observations, defaults to .1
        :type sigma_y: float, optional
        :param mu_prior: Mean of the prior, defaults to 0
        :type mu_prior: float, optional
        :param sigma_prior: Standard deviation of the prior, defaults to 1.
        :type sigma_prior: float, optional
        :param bias: Whether bias is included in the model, defaults to True.
        :type bias: bool
        """
        self.N, self.D = X.shape
        self.bias = bias
        if self.bias:
            self.D += 1
            self.X = jnp.ones((self.N, self.D))
            self.X = self.X.at[:,1:].set(X)
        else:
            self.X = X
        self.y = y
        self.mu_prior = jnp.array([mu_prior])
        self.sigma_prior = jnp.array([sigma_prior])
        self.sigma_y = jnp.array([sigma_y])
        self._construct_prior()

    def _construct_prior(self):
        """Constructs the prior p(w\vert \mu_w, \sigma_w^2)
        """
        Sigma = jnp.array([self.sigma_prior ** 2. * jnp.eye(self.D)])
        mu = self.mu_prior * jnp.ones((1, self.D))
        self.prior = densities.GaussianDiagDensity(Sigma=Sigma, mu=mu)

    def get_likelihood(self, X: jnp.ndarray, y: jnp.ndarray) -> factors.ConjugateFactor:
        """Computes the likelihood

            L(y|X,w) = \prod_i N(y_i\vert X_iw, \sigma_x^2)

        :param X: Input data [N x D]
        :type X: jnp.ndarray
        :param y: Target variable [N]
        :type y: jnp.ndarray
        :return: Likelihood object for data X, and y.
        :rtype: factors.ConjugateFactor
        """
        Lambda = jnp.array([jnp.dot(X.T, X) / self.sigma_y ** 2.])
        nu = jnp.array([jnp.dot(X.T, y) / self.sigma_y ** 2.])
        ln_beta = jnp.array([- .5 * (jnp.dot(y.T, y) / self.sigma_y ** 2. +
                          self.N * jnp.log(2. * jnp.pi * self.sigma_y ** 2.))])
        likelihood = factors.ConjugateFactor(Lambda=Lambda, nu=nu, ln_beta=ln_beta)
        return likelihood

    def get_posterior(self):
        """ Computes the posterior
        
            p(w\vert X,y) = \frac{L(y|X,w)p(w)}{p(y\vert X)}.
        """
        likelihood = self.get_likelihood(self.X, self.y)
        self.posterior_measure = self.prior.hadamard(likelihood)
        self.posterior = self.posterior_measure.get_density()

    def get_log_marginal_likelihood(self) -> jnp.ndarray:
        """Computes the log marginal likelihood (of training data)
        
           \ln p(y\vert X) = \ln \int L(y|X,w)p(w)dw


        :return: Log marginal likelihood.
        :rtype: jnp.ndarray
        """
        self.get_posterior()
        return self.posterior_measure.log_integral()

    def predict(self, X: jnp.ndarray) -> densities.GaussianDensity:
        """ Predicts y for given X, i.e.
        
            p(y^*\vert X^*) = \int L(y^*\vert X^*, w)p(w\vert X, y).

        :param X: Input data [N x D]
        :type X: jnp.ndarray
        :return: Posterior density for y.
        :rtype: densities.GaussianDensity
        """
        N = X.shape[0]
        if self.bias:
            X_new = jnp.ones((N, self.D))
            X_new = X_new.at[:,1:].set(X)
        else:
            X_new = X
        M = jnp.reshape(X_new, (N, 1, X_new.shape[1]))
        b = jnp.zeros((N, 1))
        Sigma = self.sigma_y ** 2. * jnp.ones((N, 1, 1))
        likelihood_measure = conditionals.ConditionalGaussianDensity(M=M, b=b, Sigma=Sigma)
        return likelihood_measure.affine_marginal_transformation(self.posterior)

def generate_data(N: int, D: int, sigma_noise: float=.1, bias: bool=True) -> Tuple[jnp.ndarray]:
    """Generates data for linear regression.

    :param N: Number of data points
    :type N: int
    :param D: Number of input dimensions.
    :type D: int
    :param sigma_noise: Observation noise, defaults to .1
    :type sigma_noise: float, optional
    :param bias: Whether bias is included in the model, defaults to True.
    :type bias: bool
    :return: Input, target wand weights.
    :rtype: Tuple[jnp.ndarray]
    """
    X = 10 * jnp.array(np.random.rand(N, D))
    w = jnp.array(np.random.randn(D,))
    if bias:
        b = jnp.array(np.random.randn(1,))
    else:
        b = jnp.zeros((1,))
    y = jnp.dot(X, w) + b + sigma_noise * jnp.array(np.random.randn(N,))
    return X, y, w, b

# %%
if __name__=='__main__':
    N = 20
    D = 1
    X, y, w, b = generate_data(N, D)
    X_range = jnp.array([jnp.linspace(0,10,100)]).T
    lreg = LinearRegression(X, y)
    lreg.get_posterior()
    prediction_density = lreg.predict(X_range)

    w_range, b_range = jnp.linspace(w[0] - .2, w[0] + .2, 100), jnp.linspace(b[0] - .2, b[0] + .2,100)
    w_mesh, b_mesh = jnp.meshgrid(w_range, b_range)
    mesh = jnp.vstack([b_mesh.flatten(), w_mesh.flatten()]).T
    posterior_mesh = lreg.posterior.evaluate(mesh)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.pcolor(b_range, w_range, posterior_mesh.reshape(100,100))
    plt.plot(b, w, 'C3o')
    plt.xlabel('Bias')
    plt.ylabel('Weight')
    plt.title('Posterior')
    plt.subplot(122)
    plt.plot(X, y, 'k.')
    mu, std = jnp.squeeze(prediction_density.mu), jnp.sqrt(jnp.squeeze(prediction_density.Sigma))
    plt.fill_between(jnp.squeeze(X_range), mu - std, mu + std, alpha=.5)
    plt.plot(jnp.squeeze(X_range), mu, 'C0')
    plt.plot(jnp.squeeze(X_range), jnp.dot(X_range, w) + b, 'k--')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Data space')
    plt.show()
# %%
