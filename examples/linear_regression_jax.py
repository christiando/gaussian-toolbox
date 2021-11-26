# %%
import sys
sys.path.append('../')
from jax import numpy as jnp, jit
import numpy as np
# import objax
# from scipy.optimize import minimize
from src_jax import densities, measures, conditionals
# from matplotlib import pyplot

class LinearRegression:

    def __init__(self, X: jnp.ndarray, y: jnp.ndarray, sigma_x: float=.1,
                 mu_prior: float = 0, sigma_prior: float = 1.):
        self.N, self.D = X.shape
        self.X = X
        self.y = y
        self.mu_prior = jnp.array([mu_prior])
        self.sigma_prior = jnp.array([sigma_prior])
        self.sigma_x = jnp.array([sigma_x])
        self._construct_prior()

    def _construct_prior(self):
        Sigma = jnp.array([self.sigma_prior ** 2. * jnp.eye(self.D)])
        mu = self.mu_prior * jnp.ones((1, self.D))
        self.prior = densities.GaussianDiagDensity(Sigma=Sigma, mu=mu)

    def get_likelihood(self, X: jnp.ndarray, y: jnp.ndarray) -> measures.GaussianMeasure:
        Lambda = jnp.array([jnp.dot(X.T, X) / self.sigma_x ** 2.])
        nu = jnp.array([jnp.dot(X.T, y) / self.sigma_x ** 2.])
        ln_beta = jnp.array([- .5 * (jnp.dot(y.T, y) / self.sigma_x ** 2. +
                          self.N * jnp.log(2. * jnp.pi * self.sigma_x ** 2.))])
        likelihood = measures.GaussianMeasure(Lambda=Lambda, nu=nu, ln_beta=ln_beta)
        return likelihood

    def get_posterior(self):
        likelihood = self.get_likelihood(self.X, self.y)
        self.posterior_measure = likelihood.hadamard(self.prior)
        self.posterior = self.posterior_measure.get_density()

    def get_log_marginal_likelihood(self) -> jnp.ndarray:
        self.get_posterior()
        return self.posterior_measure.log_integral()

    def predict(self, X: jnp.ndarray):
        N = X.shape[0]
        M = jnp.reshape(X, (N, 1, X.shape[1]))
        b = jnp.zeros((N, 1))
        Sigma = self.sigma_x ** 2. * jnp.ones((N, 1, 1))
        likelihood_measure = conditionals.ConditionalGaussianDensity(M=M, b=b, Sigma=Sigma)
        return likelihood_measure.affine_marginal_transformation(self.posterior)

def generate_data(N: int, D: int, sigma_noise: float=.1):
    X = jnp.array(np.random.rand(N, D))
    w = jnp.array(np.random.randn(D,))
    y = jnp.dot(X, w) + sigma_noise * jnp.array(np.random.randn(N,))
    return X, y, w

# %%
if __name__=='__main__':
    N = 10000
    D = 10
    X, y, w = generate_data(N, D)
    X_range = jnp.array([jnp.linspace(0,1,10)]).T
    # y_range = jnp.dot(X_range, w)
    # pyplot.plot(X, y, '.')
    # pyplot.plot(X_range, y_range, 'k')
    # pyplot.show()

    # %%
    @jit
    def func():
        lreg = LinearRegression(X, y)
        lreg.get_posterior()
        prediction_density = lreg.predict(X_range)

    for i in range(100):
        print(i)
        func()

    # pyplot.plot(X, y, 'k.')
    # pyplot.plot(X_range, y_range, 'k')
# lb, ub = prediction_density.mu[:,0] - jnp.sqrt(prediction_density.Sigma[:,0,0]), prediction_density.mu[:,0] + jnp.sqrt(prediction_density.Sigma[:,0,0])
# pyplot.plot(X_range, prediction_density.mu, 'C3--')
# pyplot.fill_between(X_range[:,0], lb, ub, color='C3', alpha=.5)
# pyplot.show()

# %%

# def func():
#     sigma_range = jnp.logspace(-2,1,100)
#     marginal_likelihood = []
#     for i, sigma_x in enumerate(sigma_range):
#         lreg = LinearRegression(X, y, sigma_x=sigma_x)
#         lreg.get_posterior()
#         marginal_likelihood.append(lreg.get_log_marginal_likelihood()[0,0])
#
#     pyplot.plot(sigma_range, marginal_likelihood)
#     pyplot.plot(sigma_range, -marginal_likelihood2)
#     pyplot.yscale('log')
#     pyplot.xscale('log')
#     pyplot.show()
#
# func()
# # %%
# opt_sigma_x = sigma_range[jnp.argmax(marginal_likelihood)]
#
# lreg = LinearRegression(X, y, sigma_x=opt_sigma_x)
# lreg.get_posterior()
# prediction_density = lreg.predict(X_range)
#
# pyplot.plot(X, y, 'k.')
# pyplot.plot(X_range, y_range, 'k')
# lb, ub = prediction_density.mu[:,0] - jnp.sqrt(prediction_density.Sigma[:,0,0]), \
#          prediction_density.mu[:,0] + jnp.sqrt(prediction_density.Sigma[:,0,0])
# pyplot.plot(X_range, prediction_density.mu, 'C3--')
# pyplot.fill_between(X_range[:,0], lb, ub, color='C3', alpha=.5)
# pyplot.show()
#
# # %% optimize sigma_x
#
# from objax.optimizer import lars
# lreg = LinearRegression(X, y)
#
#
# def train_op(log_sigma_x):
#     lreg.sigma_x = jnp.exp(log_sigma_x)
#     return -lreg.get_log_marginal_likelihood()
#
# sigma_opt_x = jnp.exp(minimize(value_and_grad(train_op), x0=objax.TrainVar(jnp.array([0.])), method='L-BFGS-B', jac=True).x)
#
# lreg = LinearRegression(X, y, sigma_x=opt_sigma_x)
# lreg.get_posterior()
# prediction_density = lreg.predict(X_range)
#
# pyplot.plot(X, y, 'k.')
# pyplot.plot(X_range, y_range, 'k')
# lb, ub = prediction_density.mu[:,0] - jnp.sqrt(prediction_density.Sigma[:,0,0]), \
#          prediction_density.mu[:,0] + jnp.sqrt(prediction_density.Sigma[:,0,0])
# pyplot.plot(X_range, prediction_density.mu, 'C3--')
# pyplot.fill_between(X_range[:,0], lb, ub, color='C3', alpha=.5, zorder=9)
# pyplot.title('$\sigma^* = %.2f$' %sigma_opt_x)
# pyplot.show()
