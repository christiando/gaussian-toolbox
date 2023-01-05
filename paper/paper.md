---
title: 'Gaussian Toolbox: A Python package for Gaussian algebra'
tags:
  - Python
  - Gaussians
  - JAX
  - Machine Learning
  - Statistic
authors:
  - name: Christian Donner
    orcid: 0000-0002-4499-2895
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Swiss Data Science Center, Switzerland
   index: 1
date: 13 August 2022
bibliography: paper.bib
---

# Summary

We provide an object oriented toolbox, that facilitates the user quick manipulations of Gaussian densities. Multiplications, integration, marginalization, and conditioning can be done in a quasi symbolic way, such that the user does not need to care about deriving always mean and covariance. By defining so called *affine transformations*, that take a conditional density $p({\bf x}\vert {\bf y})$ and a marginal $p({\bf y})$, the user can easily obtain $p({\bf x}, {\bf y})$, $p({\bf x})$, and $p({\bf y}\vert {\bf x})$. This basic operations allow implementation of many models, such as Bayesian Linear regression, GP regression, Kalman filter \& smoother, etc. We also show case, how non-linear conditionals $p({\bf x}\vert {\bf y})$ can be defined, where the affine transformations can only performed approximately. This opens the door for quick prototyping of a manifold of probabilistic models.

# Statement of need

Gaussian densities are ubiquitous in nowadays Machine Learning. Reasons for that are manifold, but it surely an important one is, that certain operations are tractable, such as normalization, marginalization, conditioning. Also it is conjugate to itself, which is the reason that e.g. Bayesian Linear regression is efficient and almost the starting point for any data-driven project. Naturally aforementioned operations appear also in the machine learning subfield concerned with Gaussian processes (GPs) [@williams2006gaussian], and nowadays celebrated diffusion models[@ho2020denoising] make heavy use of Gaussian manipulations.

Many open source software libraries concerned with representing distributions, such as `tensorflow probabilities` [@dillon2017tensorflow], `pytorch distribution` [@Paszke_PyTorch_An_Imperative_2019], and `distrax` [@deepmind2020jax], aim at generality and cover a wide range of distribution. However, the focus is mainly, that they can be sampled and used within loss functions, facilitating black box inference with Hamiltonian Monte Carlo sampling or black box variational inference as in `pyro`[@bingham2019pyro] or `stan`[@stan2022stan].

However, because these libraries are concerned about generality, they do not leverage the tractability of Gaussian densities as much as one could do. Here we present `Gaussian Toolbox (GT)` which has an orthogonal focus: It just focuses on Gaussian densities and models, that can be represented through a combination of these. It simplifies Gaussian manipulation (multiplication with conjugate factors, integration, marginalization, conditioning, etc.) and implements *affine transformations* for Gaussians. 

We showcase the usage of `GT` using it for implementation of the Bayesian regression, Kalman filter, and a simple latent variable model. To note, there are many related libraries, that address higher level needs such as probabilistic timeseries (e.g. `dynamax`[@dynamax]), or Gaussian Processes (e.g. `gpflow`[@GPFlow2017], `gpjax`[@Pinder2022]). `GT` aims at expressing the underlying Gaussian operations of these models with minimal effort, and can fertilize future research in that area. 

`GT` is fully based on `jax`[@jax2018github] enabling just in time compilation and GPU compatibility.

# Acknowledgements

We acknowledge contributions from Maurizio DiLucente, and Hideaki Shimazaki. We acknowledge the _Sense Dynamics_ (C19-06) project to provide the framework and collaborations that made this work possible.

# References