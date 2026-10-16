"""Randomized SciPy/JAX three-compartment solver parity without patient data or NumPyro."""
import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm

from pk_model import concentrations


def scipy_concentrations(parameters, time, rate, bolus):
    v1, v2, v3, cl, q2, q3 = parameters
    cl, q2, q3 = cl / 60., q2 / 60., q3 / 60.
    a = np.array([[-(cl + q2 + q3) / v1, q2 / v2, q3 / v3],
                  [q2 / v1, -q2 / v2, 0.], [q3 / v1, 0., -q3 / v3]])
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = a
    augmented[0, 3] = 1.
    amount = np.zeros(3)
    prediction = []
    for i in range(len(time)):
        amount[0] += bolus[i]
        prediction.append(amount[0] / (v1 * 1000.))
        if i + 1 < len(time) and time[i + 1] > time[i]:
            amount = (expm(augmented * (time[i + 1] - time[i])) @ np.r_[amount, rate[i]])[:3]
    return np.asarray(prediction)


def test_randomized_jax_scipy_pk_parity():
    rng = np.random.default_rng(20261016)
    low = np.array([5., 10., 30., .2, .2, .2])
    high = np.array([30., 70., 300., 4., 5., 5.])
    for _ in range(30):
        parameters = np.exp(rng.uniform(np.log(low), np.log(high)))
        time = np.cumsum(rng.choice([0, 1, 5, 30, 120, 300], size=18)).astype(float)
        rate = rng.uniform(0, 200, size=len(time))
        bolus = rng.choice([0, 0, 0, 1000, 5000], size=len(time)).astype(float)
        expected = scipy_concentrations(parameters, time, rate, bolus)
        actual = np.asarray(concentrations(jnp.asarray(parameters), jnp.asarray(time),
                                           jnp.asarray(rate), jnp.asarray(bolus)))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
