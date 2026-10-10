"""Longitudinal held-out conditioning with variable observation counts.

This test isolates a known local intercept, avoiding confounding with a
global parameter while exercising BET's actual masked likelihood interface.
"""
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from BayesianEstimationTool import Estimator, Svi


def model_fn(globals_, local, data):
    return {"prediction": local["intercept"] + 0.1 * data["time"]}


def likelihood(globals_, local, data, out):
    mask = data["observed"]
    prediction = out["prediction"]
    safe_y = jnp.where(mask, data["y"], prediction)
    return {"measurements": (safe_y, dist.Normal(prediction, 0.15).mask(mask))}


def score(out, data):
    # Evaluate all actual observations, including those withheld from inference.
    return jnp.mean((out["prediction"] - data["y"]) ** 2)


def method():
    return Svi(numSteps=150, numSamples=64, progressBar=False, medianSamples=2)


def test_more_held_out_observations_improve_local_prediction():
    times = jnp.tile(jnp.arange(6, dtype=float), (12, 1))
    offsets = jnp.linspace(-0.9, 0.9, 12)
    observations = offsets[:, None] + 0.1 * times
    data = {
        "time": times,
        "y": observations,
        "observed": jnp.ones_like(observations, dtype=bool),
    }
    est = Estimator(data, model_fn, ["intercept"], rngSeed=27)
    est.Train(list(range(10)), likelihood, method())

    def error_with_count(n):
        # Replace the estimator's test-only observation mask, preserving the
        # original complete data for ScoreFn's held-out comparison.
        new_mask = jnp.arange(6) < n
        est._data = {**data, "observed": data["observed"].at[10:].set(new_mask)}
        est.Test([10, 11], likelihood, method(), rngSeed=33)
        values = np.asarray(est.ScoreTestSamples(score))
        assert values.shape == (1, 64, 2)
        assert np.isfinite(values).all()
        return float(np.mean(np.median(values, axis=1)))

    one = error_with_count(1)
    six = error_with_count(6)
    # The six repeated independent measurements constrain each individual's
    # intercept more tightly; a generous tolerance avoids stochastic flakes.
    assert six < one + 0.12
