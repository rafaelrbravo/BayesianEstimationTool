"""Differentiable three-compartment propofol PK simulator.

The state is drug amount in micrograms in each compartment. A constant
infusion is propagated exactly over each event interval using a 4x4 matrix
exponential, avoiding adaptive ODE solver iterations. Output is central
concentration in micrograms/mL at every event, *after* its bolus and
*before* the following infusion interval.
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm



def rates_to_parameters(rates):
    """Map [k10,k12,k13,k21,k31,V1] (1/s, L) to [V1,V2,V3,CL,Q2,Q3] (L, L/min).

    The mapping assumes positive rate constants and reversible peripheral compartments.
    """
    k10, k12, k13, k21, k31, v1 = rates
    return jnp.stack((v1, v1 * k12 / k21, v1 * k13 / k31,
                      60. * v1 * k10, 60. * v1 * k12, 60. * v1 * k13))

def published_rates_to_parameters(rates):
    """Map published [k10,k12,k13,k21,k31,V1] (1/s, mL) into solver units."""
    return rates_to_parameters(rates.at[5].multiply(0.001))


def concentrations(parameters, time, rate, bolus):
    """parameters = [V1, V2, V3, CL, Q2, Q3] in L and L/min."""
    v1, v2, v3, cl, q2, q3 = parameters
    # Time is in seconds. The upstream data README documents infusion rates
    # as ug/s; preserve them unchanged. The Julia reader has a conflicting ug/min
    # comment, but ug/s is consistent with the simulator's time and volume units.
    cl, q2, q3 = cl / 60, q2 / 60, q3 / 60
    a = jnp.array([
        [-(cl + q2 + q3) / v1, q2 / v2, q3 / v3],
        [q2 / v1, -q2 / v2, 0.0],
        [q3 / v1, 0.0, -q3 / v3],
    ])
    # Augmented linear system integrates constant infusion exactly, including
    # the singular zero eigenvalue of the infusion source.
    aug = jnp.zeros((4, 4)).at[:3, :3].set(a).at[0, 3].set(1.0)
    dt = jnp.diff(time, append=time[-1])

    def step(amount, event):
        duration, infusion, dose = event
        amount = amount.at[0].add(dose)
        concentration = amount[0] / (v1 * 1000.0)
        def propagate(_):
            transition = expm(aug * duration)
            return (transition @ jnp.concatenate((amount, jnp.array([infusion]))))[:3]

        # The data loader validates nondecreasing times before tracing. The
        # positive-duration branch skips padded and repeated-measurement rows;
        # this traced kernel does not itself raise on negative durations.
        next_amount = jax.lax.cond(duration > 0, propagate, lambda _: amount, operand=None)
        return next_amount, concentration

    _, result = jax.lax.scan(step, jnp.zeros(3), (dt, rate, bolus))
    return result


if __name__ == "__main__":
    import numpy as np

    p = jnp.array([10., 20., 30., 1., 2., 3.])
    t = jnp.array([0., 60., 120., 180.])
    zeros = jnp.zeros(4)
    # No dosing produces no concentration.
    np.testing.assert_allclose(concentrations(p, t, zeros, zeros), 0, atol=1e-7)
    # A bolus at t=0 produces the exact immediate concentration.
    dose = jnp.array([10000., 0., 0., 0.])
    prediction = concentrations(p, t, zeros, dose)
    np.testing.assert_allclose(float(prediction[0]), 1.0, rtol=1e-6)
    assert np.all(np.isfinite(prediction)) and np.all(np.asarray(prediction) >= 0)
    assert float(prediction[-1]) < float(prediction[0])
    # A constant infusion starts at zero concentration and increases.
    infusion = jnp.array([100., 100., 100., 0.])
    prediction = concentrations(p, t, infusion, zeros)
    assert float(prediction[0]) == 0
    assert float(prediction[1]) > 0
    # Repeated timestamps accumulate boluses without advancing time.
    # With zero clearance/exchange, infusion adds rate * elapsed seconds.
    simple = jnp.array([10., 20., 30., 0., 0., 0.])
    repeated_time = jnp.array([0., 0., 10., 20.])
    repeated_rate = jnp.array([0., 100., 100., 0.])
    repeated_bolus = jnp.array([1000., 2000., 0., 0.])
    np.testing.assert_allclose(
        concentrations(simple, repeated_time, repeated_rate, repeated_bolus),
        [0.1, 0.3, 0.4, 0.5], rtol=1e-5, atol=1e-6)
    # Isolated central compartment must reproduce analytic exponential elimination.
    # CL=0.6 L/min, V1=10 L implies an elimination constant of 0.001/s.
    elimination = jnp.array([10., 20., 30., 0.6, 0., 0.])
    elimination_time = jnp.array([0., 60., 120.])
    elimination_bolus = jnp.array([10000., 0., 0.])
    np.testing.assert_allclose(
        concentrations(elimination, elimination_time, jnp.zeros(3), elimination_bolus),
        jnp.exp(-0.001 * elimination_time), rtol=1e-5, atol=1e-6)
    print("PK solver smoke tests passed")
