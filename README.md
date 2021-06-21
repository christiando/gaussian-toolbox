# Gaussian Toolbox

At the heart of this library the is to quickly manipulate Gaussians and solve integrals with respect to Gaussian measures. These operations are used in a wide range of probabilistic models, such as Gaussian process models, state space models, latent variable models, etc. The plan is to slowly incorporate these models one after the other in this library.

## The backbone: Gaussian manipulations

This library primarily provides the functionality to quickly manipulate, integrate and sample from Gaussians. If 

```math
\phi(\mathbf{x}) = N(\mu, \Sigma),
```

is a Gaussian density, this library allows to quickly compute the resulting functional form

```math
u(\mathbf{x}) = \beta\exp\left(-\frac{1}{2}\mathbf{x}^\top\Lambda \mathbf{x} + \nu^\top \mathbf{x}\right)\phi(\mathbf{x}).
```

For the resulting measure certain integrals can then be computed quickly

```math
\int f(\mathbf{x}) {\rm d}u(\mathbf{x}),
```

where $`f`$ is can be up to fourth order of $`\mathbf{x}`$. Furthermore, some functionality for mixture measures and density is provided.

### The code structure

The main code is in `/src/` folder. 

+ `factors.py` contains the main utilities for functions that are conjugate to Gaussian measures, i.e. its product with a Gaussian measure is again a Gaussian measure. The main class is `ConjugateFactor`, which is the most general functional form. Check the documentation for subclasses.
+ `measures.py` contains the functionality of `GaussianMeasure`, i.e. the integration functionality. They can be multiplied with `ConjugateFactor` and the result are again `GaussianMeasure`. In addition `GaussianMixtureMeasure` is provided that is a class for a linear combination of Gaussian measures. Check the documentation for subclasses.
+ `densities.py` has the utilities for probability densities, i.e. it is enforced, that they are normalized. One can sample from instances of `GaussianDensity`, marginalize, condition on dimensions. Furthermore, all affine transformations are implemented. Furthermore, they inherit all the functionality from `GaussianMeasure`. Also here the `GaussianMixtureDensity` is provided, which is the density counter part of `GaussianMixtureMeasure`. The code roughly follows this [note](http://user.it.uu.se/~thosc112/pubpdf/schonl2011.pdf).
+ `conditionals.py` deals with objects that represent conditional densities, i.e. objects that are not densities without defining the conditional dimensions.

__Caution__: This is code under development. Integrals were checked by sampling, but no guarantees. ;)

## Time-series models

A certain number of models of probalistic time-series models is provided. One class are __state-space models (SSMs)__, that have the form

```math
\mathbf{z}_t = f(\mathbf{z}_{t-1}) + \zeta_t, \\
\mathbf{x}_t = g(\mathbf{z}_{t}) + \xi_t, 
```

with $\zeta_t \sim N(0,\Sigma_z(t))$ and $\xi_t \sim N(0,\Sigma_x(t))$. The first equation is the so-called state equation, defining the _state model_, and the second equation  is the observation (aka emission) equation, defining the _observation model_. This library provides various state- and observation models, that can be combined. An __expectation-maximization (EM) algorithm__ is used for inference. For details see [here](timeseries/README_timeseries.md).