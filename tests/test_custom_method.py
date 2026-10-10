"""Exercise BET's custom Method protocol without an optimizer.

This Method draws from the model's prior predictive distribution. It tests
the plumbing contract, not posterior inference quality.
"""
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import Predictive

from BayesianEstimationTool import Estimator, ValidateMethod


class PriorPredictiveMethod:
    _printMode = None

    def __init__(self, numSamples=5):
        self.numSamples = numSamples

    def _Fit(self, ModelFn, rngKey, samplePrefixes):
        raw = Predictive(ModelFn, num_samples=self.numSamples)(rngKey)
        samples = {name: value for name, value in raw.items()
                   if any(name.startswith(prefix) for prefix in samplePrefixes)}
        return {"output": None, "samples": samples}

    def _SaveArgs(self):
        return {"numSamples": self.numSamples}


def model_fn(globals_, local_zs, data):
    return {"pred": globals_["bias"] + local_zs["z"] + data["x"]}


def global_prior(shapes):
    return {"bias": dist.Normal(0, 1)}


def likelihood_fn(globals_, local_zs, data, out):
    return {"y": (data["y"], dist.Normal(out["pred"], 1))}


def score_fn(out, data):
    return out["pred"] - data["y"]


def test_custom_method_train_test_and_scores():
    method = PriorPredictiveMethod(5)
    assert ValidateMethod(method)
    x = jnp.arange(4.)
    est = Estimator({"x": x, "y": x}, model_fn, ["z"], global_prior, rngSeed=21)
    est.Train([0, 1], likelihood_fn, method)
    assert est._GetNTrainSamples() == 5
    assert np.asarray(est.ScoreTrainSamples(score_fn)).shape == (5, 2)
    outputs = est.Test([2, 3], likelihood_fn, method, nGlobalSamples=2)
    assert len(outputs) == 2
    assert np.asarray(est.ScoreTestSamples(score_fn)).shape == (2, 5, 2)
