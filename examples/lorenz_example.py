# %%
import numpy as np
from matplotlib import pyplot as plt
import sys

# sys.path.append('../../timeseries/')
sys.path.append("../")
from jax import numpy as jnp
from jax import config

config.update("jax_debug_nans", True)
from timeseries_jax import observation_model, state_model
from timeseries_jax.ssm_em import StateSpaceEM
import pandas

# %%
np.random.seed(1)
X = jnp.asarray(pandas.read_csv("../data/hideaki/y_train_ts-2.csv"))[:, 1:]
X = jnp.asarray((X - jnp.mean(X)) / jnp.std(X))
X = observation_model.augment_taken(X, 5, 3)
# %%
Dx, Dz, Dk = X.shape[1], 3, 5
# Below is the only line, that changes!
sm = state_model.LSEMStateModel(Dz, Dk, noise_z=1.0)
om = observation_model.LinearObservationModel(Dx, Dz, noise_x=1.0)
ssm_em = StateSpaceEM(X, observation_model=om, state_model=sm)
ssm_em.run()

# %%
X_test = X
fully_observed_till = 450
num_samples = 50
p_z_fo, mu_x_fo, std_x_fo = ssm_em.predict_static(
    jnp.array(X_test[:fully_observed_till]), observed_dims=jnp.arange(X_test.shape[1])
)
p0 = p_z_fo.slice(jnp.array([-1]))
p_z_predict, mu_x_predict, std_x_predict = ssm_em.predict_static(
    jnp.array(X_test[fully_observed_till:]), p0=p0
)
sample_z_predict, sample_x_predict = ssm_em.sample_trajectory_static(
    jnp.array(X_test[fully_observed_till:]), p0=p0, num_samples=num_samples
)

mu_predict = jnp.concatenate([X_test[:fully_observed_till], mu_x_predict])
sample_x = jnp.empty((X_test.shape[0], num_samples, Dx))
sample_x = sample_x.at[:fully_observed_till].set(X_test[:fully_observed_till, None])
sample_x = sample_x.at[fully_observed_till:].set(sample_x_predict)
std_predict = jnp.concatenate([jnp.zeros(X[:fully_observed_till].shape), std_x_predict])

plt.figure(figsize=(10, 6))
for dx in range(Dx):
    plt.subplot(Dx, 1, dx + 1)
    plt.plot(mu_predict[:, dx], zorder=9)
    plt.plot(X_test[:, dx], "k", zorder=9)
    plt.plot(sample_x[:, :3, dx], "C3", alpha=0.5)
    plt.plot(np.mean(sample_x[:, :, dx], axis=1), "C3--", zorder=9)
    plt.fill_between(
        range(mu_predict.shape[0]),
        mu_predict[:, dx] - 1.68 * std_predict[:, dx],
        mu_predict[:, dx] + 1.68 * std_predict[:, dx],
        alpha=0.5,
        zorder=9,
    )
    plt.vlines(fully_observed_till, -2, 2, color="k")
# %%
