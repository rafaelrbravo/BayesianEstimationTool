"""Synthetic checks for leakage-free covariate scaling and paired score exports."""
import csv

import numpy as np
import pytest

from compare_covariates import standardize_covariates, validate_scores, write_patient_scores


def test_scaler_uses_only_training_patients():
    values = np.array([[10., 2.], [20., 4.], [1000., 500.]])
    standardized = standardize_covariates(values, [0, 1])
    np.testing.assert_allclose(standardized[:2], [[-1., -1.], [1., 1.]])
    np.testing.assert_allclose(standardized[2], [197., 497.])
    assert not np.allclose(standardized.mean(axis=0), 0.)


def test_constant_training_covariate_stays_finite():
    values = np.array([[1., 7.], [3., 7.], [9., 8.]])
    standardized = standardize_covariates(values, [0, 1])
    np.testing.assert_allclose(standardized[:, 1], [0., 0., 1.])
    assert np.isfinite(standardized).all()


def test_invalid_training_indices_rejected():
    with pytest.raises(ValueError, match="Training positions"):
        standardize_covariates(np.ones((2, 2)), [])
    with pytest.raises(ValueError, match="finite"):
        standardize_covariates([[1., np.nan]], [0])


def test_patient_score_export_preserves_model_and_conditioning_columns(tmp_path):
    scores = {"no covariates": (np.array([1., 2.]), np.array([3., 4.])),
              "covariates": (np.array([5., 6.]), np.array([7., 8.]))}
    path = tmp_path / "scores.csv"
    write_patient_scores(path, [21, 10], scores)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert rows == [
        ["patient_id", "baseline_unconditioned", "covariates_unconditioned",
         "baseline_conditioned", "covariates_conditioned"],
        ["21", "1.0", "5.0", "3.0", "7.0"],
        ["10", "2.0", "6.0", "4.0", "8.0"],
    ]


def test_duplicate_ids_and_mismatched_scores_rejected():
    scores = {"no covariates": ([1., 2.], [3., 4.]),
              "covariates": ([5., 6.], [7., 8.])}
    with pytest.raises(ValueError, match="unique"):
        validate_scores([10, 10], scores)
    scores["covariates"] = ([5.], [7., 8.])
    with pytest.raises(ValueError, match="Invalid patient scores"):
        validate_scores([10, 11], scores)


def test_duplicate_or_invalid_training_positions_rejected():
    values = np.ones((3, 2))
    for positions in ([0, 0], [-1, 1], [0, 3], [0.5, 1.5], [[0, 1]]):
        with pytest.raises(ValueError, match="Training positions"):
            standardize_covariates(values, positions)
