# Install

Just use the following commands

```
git clone https://github.com/christiando/gaussian-toolbox.git
cd gaussian-toolbox/
pip install .
```

# Getting started

Then you can already get started:

```
from matplotlib import pyplot as plt
from jax import numpy as jnp
from gaussian_toolbox import gaussian_algebra as ga

R, D = 10, 1
mu = jnp.zeros((R, D))
Sigma = jnp.ones((R, D, D))
Sigma = Sigma.at[:,0,0].set(jnp.linspace(.1,1,R))


p_X = ga.pdf.GaussianPDF(Sigma=Sigma, mu=mu)

x = jnp.linspace(-5,5,1000)[:,None]

plt.plot(x[:,0], p_X(x).T)
plt.xlabel('X')
plt.ylabel('p(X)')
plt.show()
```

For more details find the tutorials below.