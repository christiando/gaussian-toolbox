from gaussian_toolbox.timeseries import state_model, observation_model
from gaussian_toolbox.timeseries.ssm import StateSpaceModel
from objax import random
from jax import numpy as jnp
import pytest


@pytest.mark.parametrize("T, Dx, Dz, Dk", [(10, 2, 1, 3)])
def test_ssm_linear(T, Dx, Dz, Dk):
    X = jnp.cosh(jnp.arange(0, T) / 2)[:, None] + 0.1 * random.normal((T, Dx))
    sm = state_model.LinearStateModel(Dz)
    om = observation_model.LinearObservationModel(Dx, Dz)
    ssm = StateSpaceModel(X, om, sm, max_iter=2)
    ssm.fit()
    assert len(ssm.llk_list) == 2
    assert ssm.llk_list[0] < ssm.llk_list[1]
    ssm.predict(X, p0=ssm.filter_density.slice(jnp.array([0])))


@pytest.mark.parametrize("T, Dx, Dz, Dk", [(10, 2, 1, 3)])
def test_ssm_se_state(T, Dx, Dz, Dk):
    X = jnp.cosh(jnp.arange(0, T) / 2)[:, None] + 0.1 * random.normal((T, Dx))
    sm = state_model.LSEMStateModel(Dz, Dk)
    om = observation_model.LinearObservationModel(Dx, Dz)
    ssm = StateSpaceModel(X, om, sm, max_iter=2)
    ssm.fit()
    assert len(ssm.llk_list) == 2
    assert ssm.llk_list[0] < ssm.llk_list[1]
    ssm.predict(X, p0=ssm.filter_density.slice(jnp.array([0])))


@pytest.mark.parametrize("T, Dx, Dz, Dk", [(10, 2, 1, 3)])
def test_ssm_rbf_state(T, Dx, Dz, Dk):
    X = jnp.cosh(jnp.arange(0, T) / 2)[:, None] + 0.1 * random.normal((T, Dx))
    sm = state_model.LRBFMStateModel(Dz, Dk)
    om = observation_model.LinearObservationModel(Dx, Dz)
    ssm = StateSpaceModel(X, om, sm, max_iter=2)
    ssm.fit()
    assert len(ssm.llk_list) == 2
    assert ssm.llk_list[0] < ssm.llk_list[1]
    ssm.predict(X, p0=ssm.filter_density.slice(jnp.array([0])))


@pytest.mark.parametrize("T, Dx, Dz, Dk", [(10, 2, 1, 3)])
def test_ssm_se_observation(T, Dx, Dz, Dk):
    X = jnp.cosh(jnp.arange(0, T) / 2)[:, None] + 0.1 * random.normal((T, Dx))
    sm = state_model.LinearStateModel(Dz)
    om = observation_model.LSEMObservationModel(Dx, Dz, Dk)
    ssm = StateSpaceModel(X, om, sm, max_iter=2)
    ssm.fit()
    assert len(ssm.llk_list) == 2
    assert ssm.llk_list[0] < ssm.llk_list[1]
    ssm.predict(X, p0=ssm.filter_density.slice(jnp.array([0])))


@pytest.mark.parametrize("T, Dx, Dz, Dk", [(10, 2, 1, 3)])
def test_ssm_rbf_observation(T, Dx, Dz, Dk):
    X = jnp.cosh(jnp.arange(0, T) / 2)[:, None] + 0.1 * random.normal((T, Dx))
    sm = state_model.LinearStateModel(Dz)
    om = observation_model.LRBFMObservationModel(Dx, Dz, Dk)
    ssm = StateSpaceModel(X, om, sm, max_iter=2)
    ssm.fit()
    assert len(ssm.llk_list) == 2
    assert ssm.llk_list[0] < ssm.llk_list[1]
    ssm.predict(X, p0=ssm.filter_density.slice(jnp.array([0])))
