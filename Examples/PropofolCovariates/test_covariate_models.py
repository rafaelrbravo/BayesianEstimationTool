"""Synthetic checks that covariate effects change only the covariate model."""
import csv

import numpy as np
import pytest
import jax.numpy as jnp

from compare_covariates import make_model, write_patient_scores, validate_scores, standardize_covariates
from bet_baseline import PARAMETERS


def test_covariates_affect_only_covariate_model():
    global_params = {
        "logTypical": jnp.log(jnp.array([10., 20., 100., 1.5, 2., 1.])),
        "spread": jnp.zeros(6),
        "covariateEffects": jnp.zeros((6, 5)).at[0, 0].set(0.5),
    }
    local_z = {name: jnp.array(0.) for name in PARAMETERS}
    event = {"time": jnp.array([0., 60.]),
             "rate": jnp.array([0., 0.]),
             "bolus": jnp.array([10000., 0.])}
    low = dict(event, covariates=jnp.array([0., 0., 0., 0., 0.]))
    high = dict(event, covariates=jnp.array([1., 0., 0., 0., 0.]))
    baseline = make_model(False)
    covariate = make_model(True)
    np.testing.assert_allclose(baseline(global_params, local_z, low)["concentration"],
                               baseline(global_params, local_z, high)["concentration"], rtol=1e-6)
    predicted_low = covariate(global_params, local_z, low)["concentration"]
    predicted_high = covariate(global_params, local_z, high)["concentration"]
    np.testing.assert_allclose(predicted_low[0] / predicted_high[0], np.exp(0.5), rtol=1e-5)
    assert np.all(np.isfinite(np.asarray(predicted_high)))



def test_paired_score_export_preserves_patient_order(tmp_path):
    scores = {
        "no covariates": (jnp.array([1., 2.]), jnp.array([3., 4.])),
        "covariates": (jnp.array([0.5, 1.5]), jnp.array([2.5, 3.5])),
    }
    path = tmp_path / "scores.csv"
    write_patient_scores(path, [101, 99], scores)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert rows[0] == ["patient_id", "baseline_unconditioned", "covariates_unconditioned",
                       "baseline_conditioned", "covariates_conditioned"]
    assert rows[1] == ["101", "1.0", "0.5", "3.0", "2.5"]
    assert rows[2] == ["99", "2.0", "1.5", "4.0", "3.5"]


if __name__ == "__main__":
    test_covariates_affect_only_covariate_model()
    print("Covariate model smoke checks passed")


def test_reject_nonfinite_scores():
    scores = {
        "no covariates": (np.array([1., np.nan]), np.array([2., 3.])),
        "covariates": (np.array([1., 2.]), np.array([2., 3.])),
    }
    with pytest.raises(ValueError, match="Invalid patient scores"):
        validate_scores([10, 20], scores)


def test_reject_misaligned_score_lengths():
    scores = {
        "no covariates": (np.array([1.]), np.array([2., 3.])),
        "covariates": (np.array([1., 2.]), np.array([2., 3.])),
    }
    with pytest.raises(ValueError, match="Invalid patient scores"):
        validate_scores([10, 20], scores)


def test_reject_negative_absolute_log_errors():
    scores = {
        "no covariates": (np.array([-0.1, 1.]), np.array([1., 2.])),
        "covariates": (np.array([1., 2.]), np.array([1., 2.])),
    }
    with pytest.raises(ValueError, match="Invalid patient scores"):
        validate_scores([10, 20], scores)


def test_reject_duplicate_held_out_patient_ids():
    scores = {
        "no covariates": (np.array([1., 2.]), np.array([2., 3.])),
        "covariates": (np.array([1., 2.]), np.array([2., 3.])),
    }
    with pytest.raises(ValueError, match="nonempty and unique"):
        validate_scores([10, 10], scores)


def test_reject_empty_held_out_cohort():
    scores = {
        "no covariates": (np.array([]), np.array([])),
        "covariates": (np.array([]), np.array([])),
    }
    with pytest.raises(ValueError, match="nonempty and unique"):
        validate_scores([], scores)


def test_standardization_does_not_use_held_out_patients():
    covariates = np.array([[1., 5.], [3., 5.], [100., 900.]])
    transformed = standardize_covariates(covariates, [0, 1])
    np.testing.assert_allclose(transformed[:2], [[-1., 0.], [1., 0.]])
    np.testing.assert_allclose(transformed[2], [98., 895.])
    changed_test = covariates.copy()
    changed_test[2] = [-1000., -9000.]
    np.testing.assert_allclose(standardize_covariates(changed_test, [0, 1])[:2], transformed[:2])


def test_standardization_rejects_nonfinite_covariates():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite two-dimensional"):
            standardize_covariates(np.array([[1., 2.], [bad, 3.]]), [0])


def test_standardization_rejects_empty_training_cohort():
    with pytest.raises(ValueError, match="Training positions"):
        standardize_covariates(np.array([[1., 2.]]), [])


def test_standardization_rejects_nonmatrix_input():
    with pytest.raises(ValueError, match="finite two-dimensional"):
        standardize_covariates(np.array([1., 2.]), [0])
