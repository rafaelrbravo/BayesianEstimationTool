"""Fast estimator API validation tests; no MCMC runs required."""
import numpy as np
import pytest
from BayesianEstimationTool import Estimator, Nuts, Svi


def model_fn(globals_, local_zs, data):
    return {"y": data["x"]}


def likelihood_fn(globals_, local_zs, data, out):
    return {}


def score_fn(out, data):
    return {"error": out["y"] - data["x"]}


def make_estimator(local_zs=("z",)):
    return Estimator({"x": np.arange(5.0)}, model_fn, localZNames=list(local_zs), rngSeed=42)


def test_training_requires_valid_indices():
    est = make_estimator()
    with pytest.raises((TypeError, ValueError, IndexError)):
        est.Train([99], likelihood_fn, Nuts(numWarmup=0, numSamples=2, numChains=1))


def test_test_requires_training():
    est = make_estimator()
    with pytest.raises(RuntimeError, match="Train"):
        est.Test([0])


def test_score_requires_training():
    est = make_estimator()
    with pytest.raises(RuntimeError, match="Train"):
        est.ScoreTrainSamples(score_fn)


def test_test_requires_local_z():
    est = make_estimator(local_zs=())
    with pytest.raises(RuntimeError, match="Train"):
        est.Test([0])


def test_method_constructor_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        Nuts(numSamples=0)
    with pytest.raises(TypeError):
        Nuts(numChains=True)
    with pytest.raises(ValueError):
        Svi(stepSize=0)
    with pytest.raises(TypeError):
        Svi(numSteps=True)


def test_save_record_before_training():
    record = make_estimator().Save(None)
    assert isinstance(record, dict)
    assert "meta" in record
    assert "state" in record


def test_train_rejects_bad_method_before_fit():
    est = make_estimator()
    with pytest.raises(TypeError, match="_Fit"):
        est.Train([0, 1], likelihood_fn, object())


def test_untrained_save_load_roundtrip(tmp_path):
    original = make_estimator()
    path = tmp_path / "untrained.npz"
    original.Save(path, saveAllData=True)
    restored = Estimator.Load(path, model_fn)
    assert restored._trainParams is None
    assert restored._testParams is None
    np.testing.assert_array_equal(np.asarray(restored._data["x"]), np.arange(5.0))
    assert restored._localZNames == ["z"]


def test_save_without_data_requires_explicit_data_on_load(tmp_path):
    original = make_estimator()
    path = tmp_path / "no_data.npz"
    original.Save(path)
    with pytest.raises(ValueError, match="data must be provided"):
        Estimator.Load(path, model_fn)
    restored = Estimator.Load(path, model_fn, data={"x": np.arange(5.0)})
    assert restored._localZNames == ["z"]


def test_missing_posterior_draws_error_is_actionable():
    est = make_estimator()
    with pytest.raises(RuntimeError, match="No saved training posterior draws"):
        est._SetTestGlobals(2, None)
