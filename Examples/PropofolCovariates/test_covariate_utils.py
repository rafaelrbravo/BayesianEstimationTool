"""Offline tests for covariate preprocessing and paired-score safeguards."""
import numpy as np
import pytest

from compare_covariates import standardize_covariates, validate_scores, write_patient_scores


def test_standardization_uses_training_patients_only():
    covariates = np.array([[1., 10.], [3., 20.], [1000., -900.]])
    result = standardize_covariates(covariates, [0, 1])
    np.testing.assert_allclose(result[:2].mean(axis=0), [0., 0.], atol=1e-12)
    np.testing.assert_allclose(result[:2].std(axis=0), [1., 1.], atol=1e-12)
    expected_test = (covariates[2] - covariates[:2].mean(axis=0)) / covariates[:2].std(axis=0)
    np.testing.assert_allclose(result[2], expected_test)
    # If held-out data leaked into scaling, this assertion would fail.
    assert np.max(np.abs(result[2])) > 10.


def test_constant_training_covariate_has_finite_scaling():
    covariates = np.array([[2., 1.], [2., 3.], [20., 5.]])
    result = standardize_covariates(covariates, [0, 1])
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:2, 0], [0., 0.])
    np.testing.assert_allclose(result[2, 0], 18.)


@pytest.mark.parametrize("positions", [[], [0, 0], [-1], [3], [0.5]])
def test_invalid_training_positions_rejected(positions):
    with pytest.raises(ValueError):
        standardize_covariates(np.ones((3, 2)), positions)


def test_nonfinite_covariates_rejected():
    with pytest.raises(ValueError, match="finite"):
        standardize_covariates(np.array([[1., np.nan], [2., 3.]]), [0])


def test_paired_scores_require_same_patient_vector_lengths():
    valid = {
        "no covariates": (np.array([1., 2.]), np.array([0.8, 1.5])),
        "covariates": (np.array([0.9, 1.8]), np.array([0.7, 1.4])),
    }
    validate_scores([101, 205], valid)
    invalid = dict(valid)
    invalid["covariates"] = (np.array([0.9]), np.array([0.7, 1.4]))
    with pytest.raises(ValueError, match="Invalid patient scores"):
        validate_scores([101, 205], invalid)


def test_patient_score_export_preserves_pairing_and_column_order(tmp_path):
    path = tmp_path / "scores.csv"
    ids = [101, 205]
    scores = {
        "no covariates": (np.array([1., 2.]), np.array([3., 4.])),
        "covariates": (np.array([10., 20.]), np.array([30., 40.])),
    }
    write_patient_scores(path, ids, scores)
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == (
        "patient_id,baseline_unconditioned,covariates_unconditioned,"
        "baseline_conditioned,covariates_conditioned"
    )
    assert rows[1] == "101,1.0,10.0,3.0,30.0"
    assert rows[2] == "205,2.0,20.0,4.0,40.0"
