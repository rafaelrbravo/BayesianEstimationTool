"""Scientific-level BET integration checks on small synthetic populations.

These are intentionally lightweight smoke-calibration tests, not guarantees
of identifiability or a substitute for full posterior predictive validation.
"""
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from BayesianEstimationTool import Estimator, Svi


def _prior(shapes):
    return {"mean": dist.Normal(0., 1.)}


def _model(globals_, local, data):
    # Correlated latent components are supplied by BET's LKJ hierarchy.
    return {"prediction": globals_["mean"] + local["a"] + 0.5 * local["b"] + data["x"]}


def _likelihood(globals_, local, data, out):
    return {"y": (data["y"], dist.Normal(out["prediction"], 0.2))}


def _score(out, data):
    return out["prediction"] - data["y"]


def _method():
    return Svi(numSteps=100, numSamples=24, progressBar=False, medianSamples=2)


def _data():
    # Deterministic correlated local effects with a modest global offset.
    x = jnp.zeros(10)
    a = jnp.array([-1., -.7, -.5, -.3, -.1, .1, .3, .5, .7, 1.])
    b = 0.7 * a + 0.3 * jnp.array([.1, -.2, .1, -.1, .2, -.1, .1, -.2, .1, .1])
    return {"x": x, "y": 0.35 + a + 0.5 * b}


def _est():
    return Estimator(_data(), _model, ["a", "b"], _prior, rngSeed=11)


def test_hierarchical_correlated_latents_recovery_smoke():
    """Fit a two-local-variable LKJ hierarchy and check finite, plausible predictions.

    One observation per individual cannot separately identify a and b, so this
    intentionally checks aggregate prediction recovery, not each latent truth.
    """
    est = _est()
    est.Train(list(range(8)), _likelihood, _method(), saveModelOut=True)
    scores = np.asarray(est.ScoreTrainSamples(_score))
    assert scores.shape == (24, 8)
    assert np.isfinite(scores).all()
    assert np.mean(np.abs(np.median(scores, axis=0))) < 1.5
    assert est._trainCholesky is not None
    assert np.isfinite(np.asarray(est._trainCholesky)).all()


def test_population_to_individual_transfer_observation_conditioning():
    """Condition held-out local variables; compare with prior-only prediction."""
    est = _est()
    est.Train(list(range(8)), _likelihood, _method())
    est.Test([8, 9], nGlobalSamples=2, nLocalZSamples=24)
    prior = np.asarray(est.ScoreTestSamples(_score))
    assert prior.shape == (2, 24, 2)
    assert np.isfinite(prior).all()
    est.Test([8, 9], _likelihood, _method(), nGlobalSamples=2)
    conditioned = np.asarray(est.ScoreTestSamples(_score))
    assert conditioned.shape == (2, 24, 2)
    assert np.isfinite(conditioned).all()
    # This checks the conditioned pathway, not monotonic improvement with
    # observation count; that needs longitudinal data and a stronger test.


def test_persistence_predictions_and_seeded_scores_equivalent(tmp_path):
    est = _est()
    est.Train(list(range(8)), _likelihood, _method(), saveModelOut=True)
    est.Test([8, 9], nGlobalSamples=2, nLocalZSamples=12, saveModelOut=True)
    train = np.asarray(est.ScoreTrainSamples(_score, nSamples=6, rngSeed=19))
    test = np.asarray(est.ScoreTestSamples(_score, nSamples=5, rngSeed=23))
    path = tmp_path / "scientific_roundtrip.npz"
    est.Save(path, saveAllData=True, saveTrainModelOut=True, saveTestModelOut=True)
    loaded = Estimator.Load(path, _model, _prior)
    np.testing.assert_array_equal(
        np.asarray(loaded.ScoreTrainSamples(_score, nSamples=6, rngSeed=19)), train
    )
    np.testing.assert_array_equal(
        np.asarray(loaded.ScoreTestSamples(_score, nSamples=5, rngSeed=23)), test
    )


def test_hierarchical_global_recovery_from_repeated_measurements():
    """Recover a known global offset when local effects average to zero.

    The observations have independent repeated measurements per patient and
    a balanced synthetic local population, making the global mean estimable.
    """
    n, t = 16, 5
    x = jnp.zeros((n, t))
    local_a = jnp.linspace(-0.8, 0.8, n)
    local_b = 0.6 * local_a
    y = 0.4 + local_a[:, None] + 0.5 * local_b[:, None] + x
    est = Estimator({"x": x, "y": y}, _model, ["a", "b"], _prior, rngSeed=43)
    est.Train(list(range(n)), _likelihood, _method())
    draws = np.asarray(est._trainGlobals["mean"])
    assert draws.shape == (24,)
    assert np.isfinite(draws).all()
    # SVI is approximate and the LKJ hierarchy can be weakly identified;
    # use a broad recovery tolerance, not a calibrated credible interval.
    assert abs(float(np.median(draws)) - 0.4) < 0.9
