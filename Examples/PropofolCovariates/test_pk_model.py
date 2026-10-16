"""Independent numerical and differentiation checks for the propofol solver.

Run: python Examples/PropofolCovariates/test_pk_model.py
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm

from pk_model import concentrations, rates_to_parameters


def test_against_scipy():
    params = np.array([10., 20., 30., 1., 2., 3.])
    time = np.array([0., 60., 120., 180.])
    rate = np.array([100., 50., 0., 0.])
    bolus = np.array([10000., 0., 2000., 0.])
    v1, v2, v3, cl, q2, q3 = params
    cl, q2, q3 = cl / 60, q2 / 60, q3 / 60
    a = np.array([
        [-(cl + q2 + q3) / v1, q2 / v2, q3 / v3],
        [q2 / v1, -q2 / v2, 0.],
        [q3 / v1, 0., -q3 / v3],
    ])
    aug = np.zeros((4, 4))
    aug[:3, :3] = a
    aug[0, 3] = 1.
    amount = np.zeros(3)
    expected = []
    for i in range(len(time)):
        amount[0] += bolus[i]
        expected.append(amount[0] / (v1 * 1000))
        if i + 1 < len(time):
            amount = (expm(aug * (time[i + 1] - time[i])) @ np.r_[amount, rate[i]])[:3]
    predicted = concentrations(jnp.asarray(params), jnp.asarray(time), jnp.asarray(rate), jnp.asarray(bolus))
    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-7)
    gradient = jax.grad(lambda p: jnp.sum(concentrations(p, jnp.asarray(time), jnp.asarray(rate), jnp.asarray(bolus))))(jnp.asarray(params))
    assert np.all(np.isfinite(gradient))
    assert np.any(np.abs(gradient) > 0)



def test_duplicate_event_time_does_not_repeat_bolus():
    params = jnp.array([10., 20., 30., 1., 2., 3.])
    # Two measurements at t=0 must see the same bolus exactly once.
    times = jnp.array([0., 0., 60.])
    rate = jnp.array([100., 100., 0.])
    bolus = jnp.array([10000., 0., 0.])
    prediction = concentrations(params, times, rate, bolus)
    np.testing.assert_allclose(prediction[:2], [1., 1.], rtol=1e-6)
    reference = concentrations(params, jnp.array([0., 60.]), jnp.array([100., 0.]), jnp.array([10000., 0.]))
    np.testing.assert_allclose(prediction[2], reference[1], rtol=1e-6)

def test_rate_constant_parameter_conversion():
    params = jnp.array([10., 20., 100., 1.5, 2., 1.])
    v1, v2, v3, cl, q2, q3 = params
    rates = jnp.array([cl / (60. * v1), q2 / (60. * v1), q3 / (60. * v1),
                       q2 / (60. * v2), q3 / (60. * v3), v1])
    np.testing.assert_allclose(rates_to_parameters(rates), params, rtol=1e-6)
    time = jnp.array([0., 60., 120.])
    infusion = jnp.array([100., 50., 0.])
    bolus = jnp.array([10000., 0., 0.])
    np.testing.assert_allclose(concentrations(rates_to_parameters(rates), time, infusion, bolus),
                               concentrations(params, time, infusion, bolus), rtol=1e-6)


if __name__ == "__main__":
    test_against_scipy()
    test_duplicate_event_time_does_not_repeat_bolus()
    test_rate_constant_parameter_conversion()
    print("Independent PK prediction and gradient checks passed")
