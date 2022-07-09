# Gaussian Toolbox

[![tests](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml/badge.svg)](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml)

At the heart of this library the is to quickly manipulate Gaussians and solve integrals with respect to Gaussian measures. These operations are used in a wide range of probabilistic models, such as Gaussian process models, state space models, latent variable models, etc. The plan is to slowly incorporate these models one after the other in this library.

## The backbone: Gaussian manipulations

This library primarily provides the functionality to quickly manipulate, integrate and sample from Gaussians. If 

$$
p(X) = N(\mu, \Sigma),
$$

is a Gaussian density, this library allows to quickly compute the resulting functional form

$$
u(X) = \beta\exp\left(-\frac{1}{2}X^\top\Lambda X + \nu^\top X\right)p(X).
$$

For the resulting measure certain integrals can then be computed quickly

$$
\int f(X) {\rm d}u(X),
$$

where $f$ is can be up to fourth order of $x$. Furthermore, some functionality for mixture measures and density is provided.

### The code structure

The main code is in `gaussian_algebra` folder. 

+ `factor.py` contains the main utilities for functions that are conjugate to Gaussian measures, i.e. its product with a Gaussian measure is again a Gaussian measure. The main class is `ConjugateFactor`, which is the most general functional form. Check the documentation for subclasses.
+ `measure.py` contains the functionality of `GaussianMeasure`, i.e. the integration functionality. They can be multiplied with `ConjugateFactor` and the result are again `GaussianMeasure`. In addition `GaussianMixtureMeasure` is provided that is a class for a linear combination of Gaussian measure. Check the documentation for subclasses.
+ `pdf.py` has the utilities for probability densities, i.e. it is enforced, that they are normalized. One can sample from instances of `GaussianPDF`, marginalize, condition on dimensions. Furthermore, all affine transformations are implemented. Furthermore, they inherit all the functionality from `GaussianMeasure`. Also here the `GaussianMixtureDensity` is provided, which is the density counter part of `GaussianMixtureMeasure`.
+ `conditional.py` deals with objects that represent conditional densities, i.e. objects that are not densities without defining the conditional dimensions.

The code roughly follows this [note](http://users.isy.liu.se/en/rt/schon/Publications/SchonL2011.pdf).

__Caution__: This is code under development. Integrals were checked by sampling, but no guarantees. ;)

## Time-series models

A certain number of models of probalistic time-series models is provided. One class are __state-space models (SSMs)__, that have the form

$$
z_{t} = f(z_{t-1}) + \zeta_t,
$$

$$
x_{t} = g(z_{t}) + \xi_t, 
$$

with $\!\zeta_t \sim {\cal N}(0,\Sigma_z(t))\!$ and $\!\xi_t \sim {\cal N}(0,\Sigma_x(t))\!$. The first equation is the so-called state equation, defining the _state model_, and the second equation  is the observation (aka emission) equation, defining the _observation model_. This library provides various state- and observation models, that can be combined. An __expectation-maximization (EM) algorithm__ is used for inference. For details see [here](timeseries_jax/README_timeseries.md).

# Installation

Clone the repository into a directory and go into the folder. Type `pip install .` for installation or `pip install -e .` for developement installation.