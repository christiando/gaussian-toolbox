# `G`aussian `T`oolbox

[![tests](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml/badge.svg)](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/github/christiando/gaussian-toolbox/branch/main/graph/badge.svg?token=IR47CKMXXD)](https://codecov.io/github/christiando/gaussian-toolbox)
![Docs](https://github.com/christiando/gaussian-toolbox/actions/workflows/docs.yml/badge.svg)

The main motivation of this library is to make Gaussian manipulations as easy as possible. For this certain [object classes](https://christiando.github.io/gaussian-toolbox/notebooks/gaussian_objects.html) are defined, which can be manipulated in the following way. The basic code tries to follow roughly this The code roughly follows this [note](http://users.isy.liu.se/en/rt/schon/Publications/SchonL2011.pdf).

[**Basic Usage**](#basics) | [**Install guide**](#installation) | [**Citing**](#citation) | [**Documentation**](https://christiando.github.io/gaussian-toolbox/)

# Basics

## Elementary Gaussian manipulation

Here, just the some important operations are shown and how they can be performed in `GT`. For the following example assume, that 

$$
p(X) = {\cal N}(\mu, \Sigma),
$$

is a Gaussian density and 

$$
f(X) = \beta\exp\left(-\frac{1}{2}X^\top\Lambda X + \nu^\top X\right).
$$

is a function that is _conjugate_ to a Gaussian. In `GT` we have two classes `GaussianPDF` and `ConjugateFactor` for these class of functions repectively. 
### Multiplication with conjugate factors.

We want to calculate the object

$$
\phi(X) = f(X) * p(X).
$$

In `GT` this is done as follows

```python
p_X = GaussianPDF(Sigma=..., mu=...)
f_X = ConjugateFactor(...)
phi_X = f_X * p_X
```

`phi_X` is the resulting object, which can be used for further operations. It's as simple as that.
### Integration

Some times we would like to integrate certain functions with respect to a Gaussian density. For example, we want to calculate

$$
\int (AX + a)(BX + b)^\top p(X){\rm d}X
$$

In `GT` this can be done as follows:

```python
integral = p_X.integrate("(Ax+a)(Bx+b)'", A_mat=..., a_vec=..., B_mat=..., b_vec=...)
```

`GT` implements the integral of several functions (e.g. polynomials up to fourth order) and frees the user from cumbersome computations.

### Affine transformation

For doing _inference_ it is very important to be able to performing certain operations e.g. 

$$
\text{Given marginal and conditional }p(X),p(Y\vert X)\text{ get the {\it other} conditional }p(X\vert Y). 
$$

In order to do so `GT` provides `ConditionalGaussianPDF`, and the operation above can be then written as

```python
p_Y_given_X = ConditionalGaussianPDF(...)
p_X_given_Y = p_Y_given_X.affine_conditional_transformation(p_X)
```

Other operations that are provided are conditioning, marginalizing, getting the joint or marginal density. For a more exhaustive example see the [docs](https://christiando.github.io/gaussian-toolbox/notebooks/affine_transforms.html)

# And much more

Based upon these operations and extensions thereof, basic models (e.g. [linear regression](https://christiando.github.io/gaussian-toolbox/notebooks/linear_regression.html)), but also more complex models (e.g. for [time-series](https://christiando.github.io/gaussian-toolbox/notebooks/timeseries.html)) can be implemented.
Furthermore, the `GT` is written completely with [JAX](https://github.com/google/jax/tree/main/docs), such that your code can run on GPU/TPU, can be just-in-time compiled, vectorized etc. Furthermore, it can be easily combined with other libraries like [optax](https://github.com/deepmind/optax) and [haiku](https://github.com/deepmind/dm-haiku). Combining Gaussian manipulations with neural networks has never been easier.

Got interested? What can you do with it?
# Installation

`GT` requires `python>=3.10`.
Clone the repository into a directory and go into the folder. Just do the following

```bash
pip install git+https://github.com/christiando/gaussian-toolbox
```

For code development do
```bash
git clone https://github.com/christiando/gaussian-toolbox.git
cd gaussian-toolbox/
pip install -r requirements.txt
pip install -e .
```

# Citation

To cite this repository:

```
@software{gt2023github,
  author = {Christian Donner},
  title = {{Gaussian Toolbox}: A Python package for Gaussian algebra},
  url = {http://github.com/christiando/gaussian-toolbox},
  version = {0.0.1},
  year = {2023},
}
```