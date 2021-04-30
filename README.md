# Gaussian Toolbox

This library's aim is to quickly manipulate Gaussians and solve integrals with respect to Gaussian measures.

## Introduction

This library provides the functionality to quickly manipulate, integrate and sample from Gaussians. If 

$`
\phi(\mathbf{x}) = {\cal N}(\boldsymbol{\mu}, \Sigma),
`$

is a Gaussian density, this library allows to quickly compute the resulting functional form

$`\u(\mathbf{x}) = \beta\exp(-\frac{1}{2}\mathbf{x}^\top\Lambda \mathbf{x} + \bolsymbol{\nu}^\top \mathbf{x})\phi(\mathbf{x}).`$

For the resulting measure certain integrals can then be computed quickly


$`\int f(\mathbf{x}) {\rm d}u((\mathbf{x})),`$

where $`f`$ is can be up to fourth order of $`\mathbf{x}`$. Furthermore, some functionality for mixture measures and density is provided.

## The code structure

The main code is in `/src/` folder. 

+ In `factors.py` contains the main utilities for functions that are conjugate to Gaussian measures, i.e. its product with a Gaussian measure is again a Gaussian measure. The main class is `ConjugateFactor`, which is the most general functional form. Check the documentation for subclasses.
+ `measures.py` contains the functionality of `GaussianMeasure`, i.e. the integration functionality. They can be multiplied with `ConjugateFactors` and the result ar again `GaussianMeasure`. In addition `GaussianMixtureMeasure` is provided that is a class for a linear combination of Gaussian measures. Check the documentation for subclasses.
+ `densities.py` has the utilities for probability densities, i.e. it is enforced, that they are normalized. One can sample from instances of `GaussianDensity`. Furthermore, they inherit all the functionality from `GaussianMeasure`. Also here the `GaussianMixtureDensity` is provided, which is the density counter part of `GaussianMixtureMeasure`.

## Caution

This is code under development. Inegrals were checked by sampling, but no guarantees. ;)