# Gaussian Toolbox (`GT`)

[![tests](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml/badge.svg)](https://github.com//christiando/gaussian-toolbox/actions/workflows/python-app.yml)

The main motivation of this library is to make Gaussian manipulations as easy as possible. For this certain [object classes](/docs/source/notebooks/gaussian_objects.ipynb) are defined, which can be manipulated in the following way. The basic code tries to follow roughly this The code roughly follows this [note](http://users.isy.liu.se/en/rt/schon/Publications/SchonL2011.pdf).

## Elementary Gaussian manipulation

Here, just the some important operations are shown and how they can be performed in `GT`. For the following example assume, that 

$$
p(X) = N(\mu, \Sigma),
$$

is a Gaussian density and 

$$
f(X) = \beta\exp\left(-\frac{1}{2}X^\top\Lambda X + \nu^\top X\right)p(X).
$$

is a function that is _conjugate_ to a Gaussian. In `GT` we have two classes `GaussianPDF` and `ConjugateFactor` for these class of functions repectively. 
### Multiplication with conjugate factors.

We want to calculate the object

$$
phi(X) = f(X) * u(X).
$$

In `GT` this is done as follows

```
p_X = GaussianPDF(Sigma=..., mu=...)
f_X = ConjugateFactor(...)
fp_X = f_X * p_X
```

`fp_X` is the resulting object, which can be used for further operations. It's as simple as that.
### Intergation

Some times we would like to integrate certain functions with respect to a Gaussian density. For example, we want to calculate

$$
\int (AX + a)(BX + b)^\top p(X){\rm d}X
$$

In `GT` this can be done as follows:

```
p_X = GaussianPDF(Sigma=..., mu=...)
integral = p_X.integrate("(Ax+a)(Bx+b)", A_mat=..., a_vec=..., B_mat=..., b_vec=...)
```

`GT` implements the integral of several functions (e.g. polynomials up to fourth order) and frees the user from cumbersome computations.

### Affine transformation

For doing _inference_ it is very important to be able to performing certain operations e.g. 

$$
T_{cond}\[p(X),p(Y\vert X)\] \rightarrow p(X\vert Y).
$$

In order to do so `GT` provides `ConditionalGaussianPDF`, and the operation above can be then written as

```
p_X = GaussianPDF(Sigma=..., mu=...)
p_Y_given_X = ConditionalGaussianPDF(...)
p_X_given_Y = p_Y_given_X.affine_conditional_transformation(...)
```

Other operations that are provided are conditioning, marginalizing, getting the joint or marginal density. For a more exhaustive example see the [docs](/docs/source/notebooks/affine_transforms.ipynb)

# And much more

Based upon these operations and extensions thereof, basic models (e.g. [linear regression](/docs/source/notebooks/linear_regression.ipynb)), but also more complex models (e.g. for time-series) can be implemented.

Got interested? What can you do with it.
# Installation

Clone the repository into a directory and go into the folder. Type `pip install .` for installation or `pip install -e .` for developement installation.