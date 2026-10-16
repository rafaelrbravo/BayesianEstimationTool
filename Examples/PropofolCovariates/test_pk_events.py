"""Regression checks for same-time events and padded PK trajectories."""
import numpy as np
import jax.numpy as jnp

from pk_model import concentrations


PARAMS = jnp.array([10., 20., 30., 1., 2., 3.])


def test_same_time_measurements_do_not_redose():
    time = jnp.array([0., 0., 60.])
    rate = jnp.zeros(3)
    bolus = jnp.array([10000., 0., 0.])
    predicted = np.asarray(concentrations(PARAMS, time, rate, bolus))
    np.testing.assert_allclose(predicted[:2], [1., 1.], rtol=1e-6)
    assert 0. < predicted[2] < 1.


def test_padding_does_not_change_previous_concentrations():
    time = jnp.array([0., 60., 120.])
    rate = jnp.array([100., 100., 0.])
    bolus = jnp.zeros(3)
    original = np.asarray(concentrations(PARAMS, time, rate, bolus))
    padded = np.asarray(concentrations(PARAMS, jnp.array([0., 60., 120., 120., 120.]),
                                       jnp.array([100., 100., 0., 0., 0.]),
                                       jnp.zeros(5)))
    np.testing.assert_allclose(padded[:3], original, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(padded[3:], original[-1], rtol=1e-6, atol=1e-7)
