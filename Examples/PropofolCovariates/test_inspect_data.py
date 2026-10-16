"""Synthetic tests for per-patient propofol measurement-count consistency."""
import pandas as pd
import pytest

from inspect_data import validate_measurement_counts


def test_matching_counts():
    tables = {
        "observations": pd.DataFrame({"ID": [10, 10, 20], "CP": [1., 2., 3.]}),
        "events": pd.DataFrame({"ID": [10, 10, 20], "ISMEASUREMENT": [2, 0, 1]}),
    }
    assert validate_measurement_counts(tables, [10, 20]) == 3


def test_mismatch_identifies_patient():
    tables = {
        "observations": pd.DataFrame({"ID": [10, 10, 20], "CP": [1., 2., 3.]}),
        "events": pd.DataFrame({"ID": [10, 20], "ISMEASUREMENT": [1, 1]}),
    }
    with pytest.raises(ValueError, match="Measurement counts disagree"):
        validate_measurement_counts(tables, [10, 20])


def test_fractional_count_rejected():
    tables = {
        "observations": pd.DataFrame({"ID": [10], "CP": [1.]}),
        "events": pd.DataFrame({"ID": [10], "ISMEASUREMENT": [1.5]}),
    }
    with pytest.raises(ValueError, match="Invalid individual event measurement counts"):
        validate_measurement_counts(tables, [10])

def test_nonfinite_count_rejected():
    tables = {
        "observations": pd.DataFrame({"ID": [10], "CP": [1.]}),
        "events": pd.DataFrame({"ID": [10], "ISMEASUREMENT": [float("inf")]}),
    }
    with pytest.raises(ValueError, match="Invalid individual event measurement counts"):
        validate_measurement_counts(tables, [10])


def test_invalid_event_counts_cannot_cancel_in_aggregate():
    tables = {
        "observations": pd.DataFrame({"ID": [10], "CP": [1.]}),
        "events": pd.DataFrame({"ID": [10, 10], "ISMEASUREMENT": [-1, 2]}),
    }
    with pytest.raises(ValueError, match="Invalid individual event measurement counts"):
        validate_measurement_counts(tables, [10])


def test_fractional_event_counts_cannot_cancel_in_aggregate():
    tables = {
        "observations": pd.DataFrame({"ID": [10], "CP": [1.]}),
        "events": pd.DataFrame({"ID": [10, 10], "ISMEASUREMENT": [0.5, 0.5]}),
    }
    with pytest.raises(ValueError, match="Invalid individual event measurement counts"):
        validate_measurement_counts(tables, [10])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.])
def test_invalid_observed_concentration_rejected(invalid):
    tables = {
        "observations": pd.DataFrame({"ID": [10, 10], "CP": [1., invalid]}),
        "events": pd.DataFrame({"ID": [10], "ISMEASUREMENT": [1]}),
    }
    with pytest.raises(ValueError, match="Invalid observed concentration values"):
        validate_measurement_counts(tables, [10])
