# Time-series models

## State-space models

State space models (SSMs) are latent variable models for time-series data, where the latent space is continuous and dictates the dynamics of the observation. Here, we consider some (not exhaustive) SSMs, where the latent variables follow a first order Markov process. A state-space model always consists of a _state model_ determining the stochastic dynamics in the latent space $`\mathbf{z}_t\in \mathbb{R}^{D_z}`$, and an _observation model_, dictating the conditional distribution of the observation the observation $`\mathbf{x}_t`$, i.e. $`p(\mathbf{x}_t\vert\mathbf{z}_t)`$. Implementations can be found in [`state_models.py`](state_models.py) and [`observation_models.py`](observation_models.py), respectively. With one instance of each, the Expectation-Maximization (EM) algorithm in [`ssm_em.py`](ssm_em.py) can be invoked, to infer the model parameters given timeseries data.

### [State models](state_models.py)

The state models that are considered here, have the form

```math
\mathbf{z}_t = f_t(\mathbf{z}_{t-1}) + \zeta_t,
```
where $`\zeta_t \sim N(0,\Sigma_z(t))`$, i.e. the transition probability is normal

```math
p(\mathbf{z}_t\vert \mathbf{z}_{t-1}) = N(f_t(\mathbf{z}_{t-1}),\Sigma_z(t)).
```

#### `StateModel`

This is a dummy class, to show what functions a state model needs to provide, such that the EM procedure can be executed.

#### `LinearStateModel`

This is a linear state transition model   
```math
\mathbf{z}_t = A \mathbf{z}_{t-1} + \mathbf{b} + \zeta_t,
```
with $`\zeta_t \sim N(0,\Sigma_z)`$. The parameters that need to be inferred are $`A, b, \Sigma_z`$.

#### `LSEMStateModel`

This implements a linear+squared exponential mean (LSEM) state model     
```math
\mathbf{z}_t = A f(\mathbf{z}_{t-1}) + b + \zeta_t,
```
with $`zeta_t \sim N(0,\Sigma_z)`$. The feature function is 
```math
f(\mathbf{z}) = (z_0, z_1,...,z_m, k(h_1(\mathbf{z}))),...,k(h_n(\mathbf{z}))).
```
The kernel and linear activation function are given by
$`k(h) = \exp(-h^2 / 2)`$ and $`h_i(x) = w_i'x + w_{i,0}`$.

The parameters that need to be inferred are $`A, b, \Sigma_z, W`$, where $`W`$ are all the kernel weights.

### [Observation models](observation_models.py)

#### `ObservationModel`

This is a dummy class, to show what functions a observation model needs to provide, such that the EM procedure can be executed.

#### `LinearObservationModel`

This class is a linear observation model, where the observations are generated as
  
```math
\mathbf{x}_t = C \mathbf{z}_t + \mathbf{d} + \xi_t 
```
with $`\xi_t \sim N(0,\Sigma_x)`$. Parameters to be inferred are $`C, \mathbf{d}, \Sigma_x`$.

#### `HCCovObservationModel`

This class is a heteroscedastic observation model, where the observations are generated as
   
```math
\mathbf{x}_t = C \mathbf{z}_t + \mathbf{d} + \xi_t.
```
with  $`\xi_t \sim N(0,\Sigma_x(\mathbf{z}_t))`$. The covariance observationsmatrix is given by
```math
\Sigma_x(\mathbf{z}) = \sigma_x^2 I + \sum_i U_i D_i(z) U_i',
```
with $`D_i(z) = 2 * \beta_i * \cosh(h_i(z))`$ and $`h_i(z) = w_i'z + b_i`$. Furthermore, $`U_i^\top U_j=\delta_{ij}`$.
Parameters to be inferred are $`C, \mathbf{d}, \sigma_x, U, \beta, W, b`$.

### Example

```python
import numpy
import observation_models
import state_models
from ssm_em import StateSpaceEM

# Generate or load some timeseries data
T = 1000
trange = numpy.arange(T)
Dx = 3
Dz = 3
X = numpy.empty((T,Dx))
X[:,0] = numpy.sin(trange / 20)
X[:,1] = numpy.sin(trange / 10)
X[:,2] = numpy.sin(trange / 5)
noise_x = .2
noise_z = .1
X += noise_x * numpy.random.randn(*X.shape)

# Instantiate a state model
sm = state_models.LinearStateModel(Dz, noise_z)
# Instantiate an observation model and initialize paramters
om = observation_models.LinearObservationModel(Dx, Dz, noise_x)
om.pca_init(X)
# Create SSM object and run em.
ssm_em = StateSpaceEM(X, observation_model=om, state_model=sm)
ssm_em.run()
```

For an example notebook check [here](../notebooks/timeseries/SSMExamples.ipynb).

## Hidden Markov models

TBD