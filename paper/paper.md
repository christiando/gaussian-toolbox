---
title: 'GTax: A Python package for Gaussian algebra'
tags:
  - Python
  - Gaussians
  - JAX
  - Machine Learning
  - Statistic
authors:
  - name: Christian Donner
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Swiss Data Science Center, Switzerland
   index: 1
date: 13 August 2022
bibliography: paper.bib
---

# Summary

We provide an object oriented toolbox, that facilitates the user quick manipulations of Gaussian densities. Multiplications, integration, marginalizations, and conitioning can be done in a quasi symbolic way, such that the user does not need to care about deriving always mean and covariance. By defining so called *affine transformations*, that take a conditional density $p({\bf x}\vert {\bf y})$ and a marginal $p({\bf y})$, the user can easily obtain $p({\bf x}, {\bf y})$, $p({\bf x})$, and $p({\bf y}\vert {\bf x})$. This basic operations allow implementation of several models, such as Bayesian Linear regression, GP regression, Kalman filter \& smoother, etc. We also show case, how non-linear conditionals $p({\bf x}\vert {\bf y})$ can be defined, where the affine transformations can only performed approximately. This opens the door for quick prototyping of a manifold of probabilistic models.

# Statement of need

Gaussian densities are ubiquitous in nowadays Machine Learning. Reasons for that are manifold, but it surely among them are, that certain operations are tractable, such as normalization, marginalization, conditioning. Also it is conjugate to itself, which is the reason that e.g. Bayesian Linear regression is efficient and almost the starting point for any data-driven project. Naturally aforementioned operations appear also in the machine learning subfield concerned with Gaussian processes (GPs), and nowadays celebrated diffusion models make heavy use of Gaussian manipulations.

Many open source software libraries concerned with representing distributions, such as `tensorflow probabilities` [@dillon2017tensorlow], `pytorch distribution` [@Paszke_PyTorch_An_Imperative_2019], and `distrax` [@deepmind2020jax], aim at generality and cover a wide range of distribution. However, the focus is mainly, that they can be sampled and used within loss functions, facilitating black box inference with Hamiltonian Monte Carlo sampling or black box variational inference as in `pyro`[@bingham2019pyro] or `stan`[@stan2022stan].

However, because these libraries are concerned about generality, they do not leverage the tractability of Gaussian densities to its fullest. Here we present `GTax` which has an orthogonal focus: It just focuses on Gaussian densities and models, that can represented with it. It simplifies Gaussian manipulation (multiplication with conjugate factors, integration, marginalization, conditioning, etc.) and implements *affine transformations* for Gaussians. 

We showcase the usage of `GTax` using it for implementation of state-space models, and Gaussian processes (GPs). We would like to emphasize, that our aim is not to subsitute timeseries, or GP libraries, such as `dynamax`[@dynamax], `gpflow`[@GPFlow2017], or `gpjax`[@Pinder2022], but just want to show, how rather complex models can be expressed much simpler in terms og Gaussian operations.

`GTax` is fully based on `jax`[@jax2018github] enabling just in time compilation and GPU compatibility.

***
The most simple example is
$$
	f({\bf x}) = h({\bf x})g({\bf x}),
$$
where $g$ is a Gaussian measure, and $h$ is a function, that is conjugate. Then by definition $f$ is a Gaussian measure as well. The \gt  attempts, that the user does not have to think about updating the parameters of $g$ with those of $h$ to get $f$, but that she can purely program on the functional level. In the case above the toolbox allows to write

```python
	f_x = h_x * g_x
```
and the resulting object is the new Gaussian measure. Subsequently, we see that by defining a certain hierarchy of functions, that can simplify Gaussian computations for the user, which allows her to focus on the modeling.

# Acknowledgements

We acknowledge contributions from Maurizio DiLucente, and Hideaki Shimazaki.

# References