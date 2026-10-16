"""Synthetic regression tests for patient-aligned independent reference comparison."""
import pandas as pd
import pytest
from compare_references import compare


def write(path, ids, errors):
    pd.DataFrame({"patient_id": ids, "eleveld_future_log_error": errors}).to_csv(path, index=False)
    return path


def test_reordered_patients_match(tmp_path):
    jax = write(tmp_path / "jax.csv", [10, 20], [0.2, 0.4])
    scipy = write(tmp_path / "scipy.csv", [20, 10], [0.4, 0.2])
    assert compare(jax, scipy).max() == pytest.approx(0)


def test_missing_patient_fails(tmp_path):
    jax = write(tmp_path / "jax.csv", [10, 20], [0.2, 0.4])
    scipy = write(tmp_path / "scipy.csv", [10, 30], [0.2, 0.4])
    with pytest.raises(ValueError, match="patient sets"):
        compare(jax, scipy)


def test_duplicate_patient_fails(tmp_path):
    jax = write(tmp_path / "jax.csv", [10, 10], [0.2, 0.4])
    scipy = write(tmp_path / "scipy.csv", [10, 20], [0.2, 0.4])
    with pytest.raises(ValueError, match="duplicate"):
        compare(jax, scipy)


def test_numerical_mismatch_identifies_patient(tmp_path):
    jax = write(tmp_path / "jax.csv", [10, 20], [0.2, 0.9])
    scipy = write(tmp_path / "scipy.csv", [10, 20], [0.2, 0.4])
    with pytest.raises(AssertionError, match="20"):
        compare(jax, scipy)


def test_nonfinite_error_fails(tmp_path):
    jax = write(tmp_path / "jax.csv", [10], [float("nan")])
    scipy = write(tmp_path / "scipy.csv", [10], [0.2])
    with pytest.raises(ValueError, match="Nonfinite"):
        compare(jax, scipy)


def test_negative_absolute_log_error_fails(tmp_path):
    jax = write(tmp_path / "jax.csv", [10], [-0.2])
    scipy = write(tmp_path / "scipy.csv", [10], [-0.2])
    with pytest.raises(ValueError, match="nonnegative"):
        compare(jax, scipy)


def test_expected_reference_patient_count(tmp_path):
    jax = write(tmp_path / "jax.csv", [10, 20], [0.2, 0.4])
    scipy = write(tmp_path / "scipy.csv", [20, 10], [0.4, 0.2])
    assert compare(jax, scipy, expected_patients=2).max() == pytest.approx(0)
    with pytest.raises(ValueError, match="Expected 3 reference patients, found 2"):
        compare(jax, scipy, expected_patients=3)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_invalid_expected_reference_count(tmp_path, count):
    jax = write(tmp_path / "jax.csv", [10], [0.2])
    scipy = write(tmp_path / "scipy.csv", [10], [0.2])
    with pytest.raises(ValueError, match="positive integer"):
        compare(jax, scipy, expected_patients=count)
