"""Small end-to-end BET tests. Run with NumPyro installed.

The tests use tiny SVI fits to keep CI runtime bounded; these test interfaces
and shapes, not posterior convergence.
"""
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from BayesianEstimationTool import Estimator, Svi


def model_fn(globals_, local_zs, data):
    return {"prediction": globals_["offset"] + local_zs["z"] + data["x"]}


def global_prior(data_shapes):
    return {"offset": dist.Normal(0, 1)}


def likelihood_fn(globals_, local_zs, data, out):
    return {"observed": (data["y"], dist.Normal(out["prediction"], 0.5))}


def score_fn(out, data):
    return out["prediction"] - data["y"]


def estimator():
    x = jnp.arange(4.0)
    y = x + jnp.array([0.1, -0.2, 0.3, -0.1])
    return Estimator({"x": x, "y": y}, model_fn, ["z"], global_prior, rngSeed=17)


def method():
    return Svi(numSteps=20, numSamples=8, progressBar=False, medianSamples=2)


def test_train_test_score_and_roundtrip(tmp_path):
    est = estimator()
    est.Train([0, 1, 2], likelihood_fn, method(), saveModelOut=True)
    train_scores = np.asarray(est.ScoreTrainSamples(score_fn))
    assert train_scores.shape == (8, 3)
    assert np.isfinite(train_scores).all()

    # No individual observations: propagate the learned population posterior.
    est.Test([3], nGlobalSamples=2, nLocalZSamples=5, saveModelOut=True)
    test_scores = np.asarray(est.ScoreTestSamples(score_fn))
    assert test_scores.shape == (2, 5, 1)
    assert np.isfinite(test_scores).all()

    filename = tmp_path / "bet_roundtrip.npz"
    est.Save(filename, saveTrainModelOut=True, saveTestModelOut=True, saveAllData=True)
    loaded = Estimator.Load(filename, model_fn, global_prior)
    np.testing.assert_allclose(np.asarray(loaded.ScoreTrainSamples(score_fn)), train_scores)
    np.testing.assert_allclose(np.asarray(loaded.ScoreTestSamples(score_fn)), test_scores)


def test_test_with_individual_likelihood():
    est = estimator()
    est.Train([0, 1, 2], likelihood_fn, method())
    outputs = est.Test([3], likelihood_fn, method(), nGlobalSamples=2)
    assert len(outputs) == 2
    scores = np.asarray(est.ScoreTestSamples(score_fn))
    assert scores.shape == (2, 8, 1)
    assert np.isfinite(scores).all()


def test_seeded_subsample_is_repeatable():
    est = estimator()
    est.Train([0, 1, 2], likelihood_fn, method(), saveModelOut=True)
    a = np.asarray(est.ScoreTrainSamples(score_fn, nSamples=4, rngSeed=123))
    b = np.asarray(est.ScoreTrainSamples(score_fn, nSamples=4, rngSeed=123))
    np.testing.assert_array_equal(a, b)


def test_local_only_sample_count():
    x = jnp.arange(4.0)
    def local_model(g, z, d):
        return {"prediction": z["z"] + d["x"]}
    est = Estimator({"x": x, "y": x}, local_model, ["z"], rngSeed=19)
    est.Train([0, 1, 2], likelihood_fn, method())
    assert est._GetNTrainSamples() == 8
    est.Test([3], nGlobalSamples=2, nLocalZSamples=5)
    assert np.asarray(est.ScoreTestSamples(score_fn)).shape == (2, 5, 1)


def test_saved_local_zs_are_required_for_recomputed_train_scores(tmp_path):
    est = estimator()
    est.Train([0, 1, 2], likelihood_fn, method(), saveModelOut=False)
    path = tmp_path / "without_local_zs.npz"
    est.Save(path, saveTrainLocalZs=False, saveAllData=True)
    restored = Estimator.Load(path, model_fn, global_prior)
    with pytest.raises(RuntimeError, match="local Zs"):
        restored.ScoreTrainSamples(score_fn)


def test_saved_model_outputs_allow_scoring_without_local_zs(tmp_path):
    est = estimator()
    est.Train([0, 1, 2], likelihood_fn, method(), saveModelOut=True)
    expected = np.asarray(est.ScoreTrainSamples(score_fn))
    path = tmp_path / "outputs_only.npz"
    est.Save(path, saveTrainLocalZs=False, saveTrainModelOut=True, saveAllData=True)
    restored = Estimator.Load(path, model_fn, global_prior)
    np.testing.assert_allclose(np.asarray(restored.ScoreTrainSamples(score_fn)), expected)


def test_test_sample_alignment_with_multiple_individuals():
    x = jnp.arange(5.0)
    y = x + jnp.array([0.1, -0.2, 0.3, -0.1, 0.2])
    est = Estimator({"x": x, "y": y}, model_fn, ["z"], global_prior, rngSeed=23)
    est.Train([0, 1, 2], likelihood_fn, method())
    est.Test([3, 4], nGlobalSamples=3, nLocalZSamples=6, saveModelOut=True)
    assert np.asarray(est.ScoreTestSamples(score_fn)).shape == (3, 6, 2)
    assert np.asarray(est.ScoreTestSamples(score_fn, nSamples=2, rngSeed=44)).shape == (3, 2, 2)
