"""Cross-check independent SciPy and JAX solvers on unpadded event histories."""
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest

from pk_model import concentrations
from scipy_reference import predict_patient


@pytest.mark.parametrize("rates,boluses", [
    ([0., 0., 0., 0.], [0., 0., 0., 0.]),
    ([0., 0., 0., 0.], [10000., 0., 0., 0.]),
    ([100., 100., 0., 0.], [0., 0., 0., 0.]),
    ([100., 0., 80., 0.], [5000., 0., 2000., 0.]),
])
def test_solvers_agree(rates, boluses):
    parameters = np.array([10., 20., 30., 1., 2., 3.])
    times = np.array([0., 10., 70., 120.])
    events = pd.DataFrame({"TIME": times, "INFUSIONRATE": rates, "BOLUS": boluses,
                           "ISMEASUREMENT": [1, 1, 1, 1]})
    independent = predict_patient(parameters, events)
    jax = np.asarray(concentrations(jnp.asarray(parameters, dtype=jnp.float32),
                                    jnp.asarray(times, dtype=jnp.float32),
                                    jnp.asarray(rates, dtype=jnp.float32),
                                    jnp.asarray(boluses, dtype=jnp.float32)))
    np.testing.assert_allclose(jax, independent, rtol=2e-4, atol=2e-5)


def test_repeated_measurements_and_padding():
    parameters = np.array([10., 20., 30., 1., 2., 3.])
    events = pd.DataFrame({"TIME": [0., 60.], "INFUSIONRATE": [100., 0.],
                           "BOLUS": [10000., 0.], "ISMEASUREMENT": [2, 1]})
    independent = predict_patient(parameters, events)
    # Expand repeated measurement into separate same-time JAX rows, then pad.
    jax = np.asarray(concentrations(jnp.asarray(parameters, dtype=jnp.float32),
                                    jnp.asarray([0., 0., 60., 60.], dtype=jnp.float32),
                                    jnp.asarray([100., 100., 0., 0.], dtype=jnp.float32),
                                    jnp.asarray([10000., 0., 0., 0.], dtype=jnp.float32)))
    np.testing.assert_allclose(jax[:3], independent, rtol=2e-4, atol=2e-5)
