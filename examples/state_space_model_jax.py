import sys
sys.path.append('../')
import numpy
import pandas
from matplotlib import pyplot
from jax.config import config
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

################################ Synthetic data #########################################################################################

def _proj(U, v):
    return numpy.dot(numpy.dot(v, U) / numpy.linalg.norm(U, axis=0), U.T)


def _gs(N):
    V = numpy.zeros((N, N))
    A = numpy.random.randn(N, N)
    # A = numpy.eye(N)
    for d in range(N):
        v = A[:, d]
        V[:, d] = v - _proj(V[:, :d], v)
        V[:, d] /= numpy.sqrt(numpy.sum(V[:, d] ** 2))
    return V


def _generate_heteroscedastic_data(T, Dx, Dz, Du, sigma_z=.02, sigma_x=.02):
    params_dict = {'T': T, 'Dx': Dx, 'Dz': Dz, 'Du': Du, 'sigma_z': sigma_z, 'sigma_x': sigma_x}
    C = numpy.random.randn(Dx, Dz)
    C /= numpy.sum(C, axis=0)[None] * .5
    U = _gs(Dx)[:Du].T
    w = 2 * numpy.random.randn(Du, Dz)
    # w /=  numpy.sum(numpy.abs(w), axis=1)[:,None]
    b_w = numpy.random.randn(Du)
    beta = 1e-2 * numpy.random.rand(Du)
    params_dict = {**params_dict, 'C': C, 'U': U, 'w': w, 'b_w': b_w, 'beta': beta}

    # Sample latent space
    z = numpy.zeros([Dz, T])
    noise_z = sigma_z * numpy.random.randn(Dz, T)
    # A = .99 * numpy.eye(Dz)
    # A[1,0] = .05
    # A[0,1] = -.05
    # b = numpy.zeros(Dz)
    # for t in range(1,T):
    #     z[:,t] = numpy.dot(A, z[:,t-1]) + b + noise_z[:,t-1]
    freq = 2 / (1000 * numpy.random.rand(Dz) + 500)
    phase = 2 * numpy.pi * numpy.random.rand(Dz)
    for idz in range(Dz):
        z[idz] = 1 * numpy.cos(2 * numpy.pi * numpy.arange(T) * freq[idz] + phase[idz]) + noise_z[idz]
    D_Sigma = 2 * beta[:, None] * (numpy.cosh(numpy.dot(w, z) + b_w[:, None]))
    x = numpy.zeros((Dx, T))
    mu_x = numpy.dot(C, z)
    noise_x = numpy.random.randn(Dx, T)
    for t in range(T):
        Sigma_x = sigma_x ** 2 * numpy.eye(Dx) + numpy.dot(numpy.dot(U, numpy.diag(D_Sigma[:, t])), U.T)
        L_x = numpy.linalg.cholesky(Sigma_x)
        x[:, t] = mu_x[:, t] + numpy.dot(L_x, noise_x[:, t])
    return x.T, z.T, params_dict


def load_synthetic_data(Dz: int = 2, Dx: int = 7, Du: int = 3, T: int = 10000, sigma_x: float = .01, seed=1):
    numpy.random.seed(seed)
    var_names = ['x_%d' % i for i in range(Dx)]
    X, Z, params_dict = _generate_heteroscedastic_data(T, Dx, Dz, Du, sigma_x=sigma_x)
    return pandas.DataFrame(data=X, columns=var_names), params_dict

# %%
if __name__=='__main__':
    X, params_dict = load_synthetic_data()
    X = X.to_numpy()
    # pyplot.plot(X[:1000,0])
    # pyplot.show()

    # %%
    from timeseries_jax import state_models
    from timeseries_jax import observation_models
    from timeseries_jax.ssm_em import StateSpaceEM
    from jax import numpy as jnp
    llk_list = []
    from jax import value_and_grad, jit
    dz = 2
    dk = 10
    du = 3

    dx = X.shape[1]
    # sm = state_models.LSEMStateModel(dz, dk)
    # for i in range(10):
    #     numpy.random.seed(i)
    #     sm = state_models.LinearStateModel(dz)
    #     om = observation_models.HCCovObservationModel(Dx=dx, Dz=dz, Du=du)
    #     ssm_em_lin = StateSpaceEM(jnp.array(X[:8000]), observation_model=om, state_model=sm, timeit=True)
    #     ssm_em_lin.run()
    #     llk_list.append(ssm_em_lin.llk_list[-1])

    sm = state_models.LinearStateModel(dz)
    # om = observation_models.LinearObservationModel(dx, dz, noise_x=1.)
    om = observation_models.HCCovObservationModel(Dx=dx, Dz=dz, Du=du, noise_x=.1)
    # om.pca_init(X, 50)
    # om.U = jnp.array(params_dict['U'])
    # om.W = om.W.at[:,1:].set(jnp.array(params_dict['w']))
    # om.W = om.W.at[:,0].set(jnp.array(params_dict['b_w']))
    # om.beta = jnp.array(params_dict['beta'])
    # om.sigma_x = jnp.array(params_dict['sigma_x'])
    # om.update_emission_density()
    # om.C = jnp.array(params_dict['C'])
    # om.C = jnp.array(params_dict['d'])
    ssm_em_lin = StateSpaceEM(jnp.array(X[:8000]), observation_model=om, state_model=sm, timeit=True)
    ssm_em_lin.run()
    # om = observation_models.HCCovObservationModel(Dx=dx, Dz=dz, Du=du)
    # om.pca_init(X, 50)
    # om.C = ssm_em_lin.om.C
    # om.d = ssm_em_lin.om.d
    # om.update_emission_density()
    # ssm_em_lin = StateSpaceEM(jnp.array(X[:8000]), observation_model=om, state_model=sm, timeit=True)
    # ssm_em_lin.run()
    # ssm_em_lin.estep()
    # ssm_em_lin.mstep()
    # smoothing_density = ssm_em_lin.smoothing_density
    # two_step_smoothing_density = ssm_em_lin.twostep_smoothing_density
    # self = ssm_em_lin.om
    # ssm_em_lin.estep()
    # ssm_em_lin.mstep()

