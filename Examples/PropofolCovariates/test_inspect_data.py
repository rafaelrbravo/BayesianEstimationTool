"""Synthetic tests for per-patient propofol measurement-count consistency."""
import pandas as pd
import pytest

from inspect_data import validate_measurement_counts, validate_measurement_times


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


def test_nonpositive_and_missing_concentrations_are_not_measurements():
    tables = {
        "observations": pd.DataFrame({"ID": [10, 10, 10, 10], "CP": [1., 0., -1., float("nan")]}),
        "events": pd.DataFrame({"ID": [10], "ISMEASUREMENT": [1]}),
    }
    assert validate_measurement_counts(tables, [10]) == 1


def test_positive_infinite_concentration_rejected():
    tables = {
        "observations": pd.DataFrame({"ID": [10, 10], "CP": [1., float("inf")]}),
        "events": pd.DataFrame({"ID": [10], "ISMEASUREMENT": [2]}),
    }
    with pytest.raises(ValueError, match="Invalid positive observed concentration values"):
        validate_measurement_counts(tables, [10])


def test_excluded_patient_invalid_event_counts_do_not_block_validation():
    tables = {
        "observations": pd.DataFrame({"ID": [10], "CP": [1.]}),
        "events": pd.DataFrame({"ID": [10, 893], "ISMEASUREMENT": [1, float("nan")]}),
    }
    assert validate_measurement_counts(tables, [10]) == 1


def test_excluded_patient_invalid_concentration_does_not_block_validation():
    tables = {
        "observations": pd.DataFrame({"ID": [10, 893], "CP": [1., float("inf")]}),
        "events": pd.DataFrame({"ID": [10], "ISMEASUREMENT": [1]}),
    }
    assert validate_measurement_counts(tables, [10]) == 1


def test_measurement_times_match_after_minute_to_second_conversion():
    tables = {
        "observations": pd.DataFrame({"ID": [10, 10, 20], "CP": [1., 2., 3.], "TIME": [1., 1., 2.]}),
        "events": pd.DataFrame({"ID": [10, 20], "TIME": [60.5, 120.], "ISMEASUREMENT": [2, 1]}),
    }
    validate_measurement_times(tables, [10, 20])


def test_measurement_time_mismatch_identifies_patient():
    tables = {
        "observations": pd.DataFrame({"ID": [10], "CP": [1.], "TIME": [1.]}),
        "events": pd.DataFrame({"ID": [10], "TIME": [100.], "ISMEASUREMENT": [1]}),
    }
    with pytest.raises(ValueError, match="times disagree for ID 10"):
        validate_measurement_times(tables, [10])
