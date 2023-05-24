from jax import numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Integer
from typing import Union

def binom(k: Union[int, Integer[Array, ""]], i: Union[int, Integer[Array, ""]]) -> Union[int, Integer[Array, ""]]:
  return jnp.array(jnp.round(jnp.exp(gammaln(k + 1) - gammaln(i + 1) - gammaln(k - i + 1))), dtype=int)