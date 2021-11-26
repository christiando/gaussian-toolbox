# %%
import sys
sys.path.append('../')
from autograd import numpy, value_and_grad
# from scipy.optimize import minimize
from src import densities, measures, conditionals
# from matplotlib import pyplot

class LinearRegression:

    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, sigma_x: float=.1,
                 mu_prior: float = 0, sigma_prior: float = 1.):
        self.N, self.D = X.shape
        self.X = X
        self.y = y
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.sigma_x = sigma_x
        self._construct_prior()

    def _construct_prior(self):
        Sigma = numpy.array([self.sigma_prior ** 2 * numpy.eye(self.D)])
        mu = self.mu_prior * numpy.ones((1, self.D))
        self.prior = densities.GaussianDiagDensity(Sigma=Sigma, mu=mu)

    def get_likelihood(self, X: numpy.ndarray, y: numpy.ndarray) -> measures.GaussianMeasure:
        Lambda = numpy.array([numpy.dot(X.T, X) / self.sigma_x ** 2])
        nu = numpy.array([numpy.dot(X.T, y) / self.sigma_x ** 2])
        ln_beta = numpy.array([- .5 * (numpy.dot(y.T, y) / self.sigma_x ** 2 +
                          self.N * numpy.log(2. * numpy.pi * self.sigma_x ** 2))])
        likelihood = measures.GaussianMeasure(Lambda=Lambda, nu=nu, ln_beta=ln_beta)
        return likelihood

    def get_posterior(self):
        likelihood = self.get_likelihood(self.X, self.y)
        self.posterior_measure = likelihood.hadamard(self.prior)
        self.posterior = self.posterior_measure.get_density()

    def get_log_marginal_likelihood(self) -> numpy.ndarray:
        self.get_posterior()
        return self.posterior_measure.log_integral()

    def predict(self, X: numpy.ndarray):
        N = X.shape[0]
        M = numpy.reshape(X, (N, 1, X.shape[1]))
        b = numpy.zeros((N, 1))
        Sigma = self.sigma_x ** 2 * numpy.ones((N, 1, 1))
        likelihood_measure = conditionals.ConditionalGaussianDensity(M=M, b=b, Sigma=Sigma)
        return likelihood_measure.affine_marginal_transformation(self.posterior)

def generate_data(N: int, D: int, sigma_noise: float=.1):
    X = numpy.random.rand(N, D)
    w = numpy.random.randn(D)
    y = numpy.dot(X, w) + sigma_noise * numpy.random.randn(N,)
    return X, y, w

# %%
if __name__=='__main__':
    N = 10000
    D = 10
    X, y, w = generate_data(N, D)
    X_range = numpy.array([numpy.linspace(0,1,10)]).T
    # y_range = numpy.dot(X_range, w)
    # pyplot.plot(X, y, '.')
    # pyplot.plot(X_range, y_range, 'k')
    # pyplot.show()

    # %%

    def func():
        lreg = LinearRegression(X, y)
        lreg.get_posterior()
        prediction_density = lreg.predict(X_range)


    for i in range(100):
        func()

# pyplot.plot(X, y, 'k.')
# pyplot.plot(X_range, y_range, 'k')
# lb, ub = prediction_density.mu[:,0] - numpy.sqrt(prediction_density.Sigma[:,0,0]), prediction_density.mu[:,0] + numpy.sqrt(prediction_density.Sigma[:,0,0])
# pyplot.plot(X_range, prediction_density.mu, 'C3--')
# pyplot.fill_between(X_range[:,0], lb, ub, color='C3', alpha=.5)
# pyplot.show()
# %%

# def func():
#     sigma_range = numpy.logspace(-2,1,100)
#     marginal_likelihood = []
#     for i, sigma_x in enumerate(sigma_range):
#         lreg = LinearRegression(X, y, sigma_x=sigma_x)
#         lreg.get_posterior()
#         marginal_likelihood.append(lreg.get_log_marginal_likelihood())

    # pyplot.plot(sigma_range, marginal_likelihood)
    # pyplot.plot(sigma_range, -marginal_likelihood2)
    # pyplot.yscale('log')
    # pyplot.xscale('log')
    # pyplot.show()
#
# func()
# # %%
#
# sigma_range = numpy.logspace(-2,1,1000)
# marginal_likelihood = numpy.empty(1000)
# marginal_likelihood2 = numpy.empty(1000)
# for i, sigma_x in enumerate(sigma_range):
#     lreg = LinearRegression(X, y, sigma_x=sigma_x, sigma_prior=.1)
#     lreg.get_posterior()
#     marginal_likelihood[i] = lreg.get_log_marginal_likelihood()
#     marginal_likelihood2[i] = -.5 * numpy.sum(y ** 2) / (lreg.prior.Sigma + lreg.sigma_x ** 2 * numpy.eye(D)) - .5 * numpy.log(
#         lreg.prior.Sigma + lreg.sigma_x ** 2 * numpy.eye(D)) - .5 * N * numpy.log(2. * numpy.pi)
#
# pyplot.plot(sigma_range, marginal_likelihood)
# # pyplot.plot(sigma_range, -marginal_likelihood2)
# # pyplot.yscale('log')
# pyplot.xscale('log')
# pyplot.show()
#
# # %%
# opt_sigma_x = sigma_range[numpy.argmax(marginal_likelihood)]
#
# lreg = LinearRegression(X, y, sigma_x=opt_sigma_x)
# lreg.get_posterior()
# prediction_density = lreg.predict(X_range)
#
# pyplot.plot(X, y, 'k.')
# pyplot.plot(X_range, y_range, 'k')
# lb, ub = prediction_density.mu[:,0] - numpy.sqrt(prediction_density.Sigma[:,0,0]), \
#          prediction_density.mu[:,0] + numpy.sqrt(prediction_density.Sigma[:,0,0])
# pyplot.plot(X_range, prediction_density.mu, 'C3--')
# pyplot.fill_between(X_range[:,0], lb, ub, color='C3', alpha=.5)
# pyplot.show()
#
# # %% optimize sigma_x
#
# lreg = LinearRegression(X, y)
#
# def train_op(log_sigma_x):
#     lreg.sigma_x = numpy.exp(log_sigma_x)
#     return -lreg.get_log_marginal_likelihood()
#
# sigma_opt_x = numpy.exp(minimize(value_and_grad(train_op), x0=0, method='L-BFGS-B', jac=True).x)
#
# lreg = LinearRegression(X, y, sigma_x=opt_sigma_x)
# lreg.get_posterior()
# prediction_density = lreg.predict(X_range)
#
# pyplot.plot(X, y, 'k.')
# pyplot.plot(X_range, y_range, 'k')
# lb, ub = prediction_density.mu[:,0] - numpy.sqrt(prediction_density.Sigma[:,0,0]), \
#          prediction_density.mu[:,0] + numpy.sqrt(prediction_density.Sigma[:,0,0])
# pyplot.plot(X_range, prediction_density.mu, 'C3--')
# pyplot.fill_between(X_range[:,0], lb, ub, color='C3', alpha=.5, zorder=9)
# pyplot.title('$\sigma^* = %.2f$' %sigma_opt_x)
# pyplot.show()
