from jax import numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Integer
from typing import Union
from jax.scipy.stats import norm

def binom(k: Union[int, Integer[Array, ""]], i: Union[int, Integer[Array, ""]]) -> Union[int, Integer[Array, ""]]:
    return jnp.array(jnp.round(jnp.exp(gammaln(k + 1) - gammaln(i + 1) - gammaln(k - i + 1))), dtype=int)

def normal_pdf(x: Float[Array, ""]) -> Float[Array, ""]:
    return norm.pdf(x)
    #return norm.pdf(jnp.where(jnp.isfinite(x), x, 1e15))

def normal_cdf(x: Float[Array, ""]) -> Float[Array, ""]:
    y = norm.cdf(x)
    return jnp.where(y < 1., y, 1. + norm.logcdf(x))
    #y = norm.cdf(jnp.where(jnp.isfinite(x), x, jnp.sign(x) * 1e15))
    