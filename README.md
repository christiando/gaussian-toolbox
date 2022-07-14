# `G`aussian `T`oolbox

[![tests](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml/badge.svg)](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml)

[![codecov](https://codecov.io/github/christiando/gaussian-toolbox/branch/main/graph/badge.svg?token=IR47CKMXXD)](https://codecov.io/github/christiando/gaussian-toolbox)

The main motivation of this library is to make Gaussian manipulations as easy as possible. For this certain [object classes](/docs/source/notebooks/gaussian_objects.ipynb) are defined, which can be manipulated in the following way. The basic code tries to follow roughly this The code roughly follows this [note](http://users.isy.liu.se/en/rt/schon/Publications/SchonL2011.pdf).

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
### Intergation

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
p_X_given_Y = p_Y_given_X.affine_conditional_transformation(p_x)
```

Other operations that are provided are conditioning, marginalizing, getting the joint or marginal density. For a more exhaustive example see the [docs](/docs/source/notebooks/affine_transforms.ipynb)

# And much more

Based upon these operations and extensions thereof, basic models (e.g. [linear regression](/docs/source/notebooks/linear_regression.ipynb)), but also more complex models (e.g. for time-series) can be implemented.
Furthermore, the `GT` is written completely with [JAX](https://github.com/google/jax/tree/main/docs) and [OBJAX](https://github.com/google/objax), and hence combining Gaussian manipulations with neural networks has never been easier.

Got interested? What can you do with it.
# Installation

Clone the repository into a directory and go into the folder. Type `pip install .` for installation or `pip install -e .` for developement installation.