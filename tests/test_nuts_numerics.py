"""Numerical regression tests for NUTS on a conjugate normal model.

Small sample counts and loose tolerances test broad posterior behavior,
not precise Monte Carlo convergence.
"""
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from BayesianEstimationTool import Estimator, Nuts


def model_fn(globals_, local_zs, data):
    return {"mean": globals_["mu"] + data["x"] * 0.0}


def global_prior(shapes):
    return {"mu": dist.Normal(0.0, 1.0)}


def likelihood_fn(globals_, local_zs, data, out):
    return {"y": (data["y"], dist.Normal(out["mean"], 1.0))}


def test_conjugate_global_posterior_mean_and_variance():
    # y_i ~ Normal(mu, 1), mu ~ Normal(0, 1).
    # With n=4 and all y=2, posterior mean=8/5 and variance=1/5.
    data = {"x": jnp.zeros(4), "y": jnp.full((4,), 2.0)}
    est = Estimator(data, model_fn, GlobalPriorFn=global_prior, rngSeed=31)
    est.Train([0, 1, 2, 3], likelihood_fn,
              Nuts(numWarmup=150, numSamples=300, numChains=1, medianSamples=10))
    samples = np.asarray(est._trainGlobals["mu"])
    assert samples.shape == (300,)
    assert abs(samples.mean() - 1.6) < 0.25
    assert abs(samples.var() - 0.2) < 0.14
    assert np.isfinite(samples).all()


def test_nuts_seed_reproducibility():
    data = {"x": jnp.zeros(2), "y": jnp.ones(2)}
    def fit():
        est = Estimator(data, model_fn, GlobalPriorFn=global_prior, rngSeed=5)
        est.Train([0, 1], likelihood_fn,
                  Nuts(numWarmup=20, numSamples=30, numChains=1, medianSamples=5))
        return np.asarray(est._trainGlobals["mu"])
    np.testing.assert_array_equal(fit(), fit())
