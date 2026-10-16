"""Synthetic checks for the propofol loader, without downloading study data.

Run: python Examples/PropofolCovariates/test_load_data.py
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

from load_data import load_data


def test_padded_events_and_covariates():
    observations = pd.DataFrame({
        "ID": [10, 10, 20],
        "CP": [1.5, 0.8, 2.],
        "AGE": [40, 40, 60],
        "WGT": [80, 80, 90],
        "HT": [200, 200, 180],
        "M1F2": [1, 1, 2],
        "A1V2": [2, 2, 1],
    })
    events = pd.DataFrame({
        "ID": [10, 10, 10, 20, 20],
        "TIME": [0., 30., 60., 0., 60.],
        "INFUSIONRATE": [10., 0., 0., 20., 0.],
        "BOLUS": [100., 0., 0., 0., 0.],
        "ISMEASUREMENT": [1, 0, 1, 1, 0],
    })
    # Patient 10 has two measurements; patient 20 has one.
    with patch("load_data.read_tables", return_value={"observations": observations, "events": events, "parameters": pd.DataFrame({"ID": [10, 20]})}), patch("load_data.patient_ids", return_value=[10, 20]):
        data = load_data()
    assert data["time"].shape == (2, 3)
    np.testing.assert_array_equal(data["observed"], [[1, 0, 1], [1, 0, 0]])
    np.testing.assert_allclose(data["concentration"], [[1.5, 0., 0.8], [2., 0., 0.]])
    np.testing.assert_allclose(data["rate"], [[10. / 60., 0., 0.], [20. / 60., 0., 0.]])
    np.testing.assert_allclose(data["time"][1], [0., 0., 0.])
    np.testing.assert_allclose(data["valid"][1], [1, 0, 0])
    np.testing.assert_allclose(data["covariates"], [[40., 80., 20., -0.5, 0.5], [60., 90., 90. / 1.8 ** 2, 0.5, -0.5]], rtol=1e-6)


def test_rejects_mismatched_measurements():
    observations = pd.DataFrame({"ID": [10], "CP": [1.5], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10, 10], "TIME": [0., 30.], "INFUSIONRATE": [10., 0.], "BOLUS": [100., 0.], "ISMEASUREMENT": [1, 1]})
    with patch("load_data.read_tables", return_value={"observations": observations, "events": events}), patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data()
        except ValueError as error:
            assert "Measurement count mismatch" in str(error)
        else:
            raise AssertionError("Expected measurement-count mismatch to fail")


def test_repeated_measurements_at_one_event():
    observations = pd.DataFrame({"ID": [10, 10], "CP": [1.5, 1.6], "AGE": [40, 40], "WGT": [80, 80], "HT": [200, 200], "M1F2": [1, 1], "A1V2": [2, 2]})
    events = pd.DataFrame({"ID": [10, 10], "TIME": [0., 30.], "INFUSIONRATE": [10., 0.], "BOLUS": [100., 0.], "ISMEASUREMENT": [2, 0]})
    with patch("load_data.read_tables", return_value={"observations": observations, "events": events}), patch("load_data.patient_ids", return_value=[10]):
        data = load_data()
    np.testing.assert_array_equal(data["observed"][0], [1, 1])
    np.testing.assert_allclose(data["time"][0], [0., 0.])
    np.testing.assert_allclose(data["bolus"][0], [100., 0.])
    np.testing.assert_allclose(data["concentration"][0], [1.5, 1.6])



def test_rejects_fractional_measurement_counts():
    observations = pd.DataFrame({"ID": [10], "CP": [1.5], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10], "TIME": [0.], "INFUSIONRATE": [10.], "BOLUS": [100.], "ISMEASUREMENT": [1.5]})
    with patch("load_data.read_tables", return_value={"observations": observations, "events": events}), patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data()
        except ValueError as error:
            assert "Invalid measurement counts" in str(error)
        else:
            raise AssertionError("Fractional measurement counts must fail")


def test_select_patient_before_padding():
    observations = pd.DataFrame({"ID": [10, 20], "CP": [1., 2.], "AGE": [40, 60], "WGT": [80, 90], "HT": [200, 180], "M1F2": [1, 2], "A1V2": [2, 1]})
    events = pd.DataFrame({"ID": [10, 10, 10, 20], "TIME": [0., 30., 60., 0.], "INFUSIONRATE": [10., 0., 0., 20.], "BOLUS": [100., 0., 0., 0.], "ISMEASUREMENT": [1, 0, 0, 1]})
    with patch("load_data.read_tables", return_value={"observations": observations, "events": events}), patch("load_data.patient_ids", return_value=[10, 20]):
        data = load_data(selected_ids=[20])
    np.testing.assert_array_equal(data["patient_ids"], [20])
    assert data["time"].shape == (1, 1)


def test_selected_ids_preserve_sorted_order():
    observations = pd.DataFrame({"ID": [10, 20], "CP": [1., 2.], "AGE": [40, 60], "WGT": [80, 90], "HT": [200, 180], "M1F2": [1, 2], "A1V2": [2, 1]})
    events = pd.DataFrame({"ID": [10, 20], "TIME": [0., 0.], "INFUSIONRATE": [10., 20.], "BOLUS": [100., 0.], "ISMEASUREMENT": [1, 1]})
    tables = {"observations": observations, "events": events}
    with patch("load_data.patient_ids", return_value=[10, 20]):
        data = load_data(selected_ids=[20, 10], tables=tables)
    np.testing.assert_array_equal(data["patient_ids"], [10, 20])


def test_rejects_nonfinite_infusion():
    observations = pd.DataFrame({"ID": [10], "CP": [1.5], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10], "TIME": [0.], "INFUSIONRATE": [np.nan], "BOLUS": [100.], "ISMEASUREMENT": [1]})
    with patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data(tables={"observations": observations, "events": events})
        except ValueError as error:
            assert "Invalid dosing" in str(error)
        else:
            raise AssertionError("Nonfinite infusion rate must fail")


def test_rejects_empty_selection():
    with patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data(selected_ids=[], tables={"observations": pd.DataFrame(), "events": pd.DataFrame()})
        except ValueError as error:
            assert "No patients selected" in str(error)
        else:
            raise AssertionError("Empty selection must fail")


def test_multiple_repeated_measurement_events():
    observations = pd.DataFrame({"ID": [10] * 4, "CP": [1., 1.1, 2., 2.1], "AGE": [40] * 4, "WGT": [80] * 4, "HT": [200] * 4, "M1F2": [1] * 4, "A1V2": [2] * 4})
    events = pd.DataFrame({"ID": [10, 10, 10], "TIME": [0., 30., 60.], "INFUSIONRATE": [10., 20., 0.], "BOLUS": [100., 200., 300.], "ISMEASUREMENT": [2, 0, 2]})
    with patch("load_data.patient_ids", return_value=[10]):
        data = load_data(tables={"observations": observations, "events": events})
    np.testing.assert_allclose(data["time"][0], [0., 0., 30., 60., 60.])
    np.testing.assert_allclose(data["bolus"][0], [100., 0., 200., 300., 0.])
    np.testing.assert_array_equal(data["observed"][0], [1, 1, 0, 1, 1])
    np.testing.assert_allclose(data["concentration"][0], [1., 1.1, 0., 2., 2.1])


def test_rejects_unknown_patient_selection():
    with patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data(selected_ids=[99], tables={"observations": pd.DataFrame(), "events": pd.DataFrame()})
        except ValueError as error:
            assert "unknown or excluded" in str(error)
        else:
            raise AssertionError("Unknown patient IDs must fail")


def test_rejects_nonfinite_bolus():
    observations = pd.DataFrame({"ID": [10], "CP": [1.5], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10], "TIME": [0.], "INFUSIONRATE": [10.], "BOLUS": [np.inf], "ISMEASUREMENT": [1]})
    with patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data(tables={"observations": observations, "events": events})
        except ValueError as error:
            assert "Invalid dosing" in str(error)
        else:
            raise AssertionError("Infinite bolus must fail")


def test_rejects_inconsistent_covariates_across_measurements():
    observations = pd.DataFrame({"ID": [10, 10], "CP": [1.5, 1.6], "AGE": [40, 41], "WGT": [80, 80], "HT": [200, 200], "M1F2": [1, 1], "A1V2": [2, 2]})
    events = pd.DataFrame({"ID": [10, 10], "TIME": [0., 30.], "INFUSIONRATE": [10., 0.], "BOLUS": [100., 0.], "ISMEASUREMENT": [1, 1]})
    with patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data(tables={"observations": observations, "events": events})
        except ValueError as error:
            assert "Inconsistent patient covariates" in str(error)
        else:
            raise AssertionError("Conflicting age for one patient must fail")


def test_upstream_infusion_rate_is_converted_from_micrograms_per_minute():
    """The upstream rate is ug/min: 60 ug/min for one minute delivers 60 ug."""
    observations = pd.DataFrame({"ID": [10], "CP": [1.], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10, 10], "TIME": [0., 60.], "INFUSIONRATE": [60., 0.], "BOLUS": [0., 0.], "ISMEASUREMENT": [0, 1]})
    with patch("load_data.patient_ids", return_value=[10]):
        data = load_data(tables={"observations": observations, "events": events})
    np.testing.assert_allclose(data["rate"][0], [1., 0.])
    np.testing.assert_allclose(np.diff(data["time"][0])[0] * data["rate"][0, 0], 60.)


if __name__ == "__main__":
    test_padded_events_and_covariates()
    test_upstream_infusion_rate_is_converted_from_micrograms_per_minute()
    test_rejects_mismatched_measurements()
    test_repeated_measurements_at_one_event()
    test_rejects_fractional_measurement_counts()
    test_select_patient_before_padding()
    test_selected_ids_preserve_sorted_order()
    test_rejects_nonfinite_infusion()
    test_rejects_empty_selection()
    test_multiple_repeated_measurement_events()
    test_rejects_unknown_patient_selection()
    test_rejects_nonfinite_bolus()
    test_rejects_inconsistent_covariates_across_measurements()
    print("Synthetic loader checks passed")


def test_missing_covariate_rows_gives_clear_error():
    observations = pd.DataFrame({"ID": [20], "CP": [1.], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10], "TIME": [0.], "INFUSIONRATE": [0.], "BOLUS": [0.], "ISMEASUREMENT": [0]})
    with patch("load_data.patient_ids", return_value=[10]):
        with np.testing.assert_raises_regex(ValueError, "Missing covariate rows"):
            load_data(tables={"observations": observations, "events": events})


def test_rejects_duplicate_selected_patient_ids():
    with patch("load_data.patient_ids", return_value=[10]):
        try:
            load_data(selected_ids=[10, 10], tables={"observations": pd.DataFrame(), "events": pd.DataFrame()})
        except ValueError as error:
            assert "contain duplicates" in str(error)
        else:
            raise AssertionError("Duplicate selected IDs must not be silently deduplicated")


def test_upstream_infusion_rate_matches_administered_dose():
    """Verify 3939 ug/min is converted to 65.65 ug/s for 10.02 s."""
    observations = pd.DataFrame({"ID": [1], "CP": [6.932], "AGE": [59], "WGT": [64], "HT": [166], "M1F2": [2], "A1V2": [1]})
    events = pd.DataFrame({"ID": [1, 1], "TIME": [0., 10.02],
                           "INFUSIONRATE": [3939., 0.], "BOLUS": [0., 0.], "ISMEASUREMENT": [0, 1]})
    with patch("load_data.patient_ids", return_value=[1]):
        data = load_data(tables={"observations": observations, "events": events})
    delivered_micrograms = (data["time"][0, 1] - data["time"][0, 0]) * data["rate"][0, 0]
    np.testing.assert_allclose(delivered_micrograms, 3939. * 10.02 / 60., rtol=0.002)


def test_events_after_final_measurement_are_omitted():
    """Late dosing cannot change earlier concentrations or their scores."""
    observations = pd.DataFrame({"ID": [10], "CP": [1.5], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10, 10, 10], "TIME": [0., 30., 60.],
                           "INFUSIONRATE": [10., 20., 30.], "BOLUS": [100., 200., 300.],
                           "ISMEASUREMENT": [0, 1, 0]})
    with patch("load_data.patient_ids", return_value=[10]):
        data = load_data(tables={"observations": observations, "events": events})
    np.testing.assert_allclose(data["time"][0], [0., 30.])
    np.testing.assert_allclose(data["rate"][0], [10. / 60., 20. / 60.])
    np.testing.assert_allclose(data["bolus"][0], [100., 200.])
    np.testing.assert_array_equal(data["observed"][0], [0, 1])


def test_no_measurement_event_fails_clearly():
    observations = pd.DataFrame({"ID": [10], "CP": [0.], "AGE": [40], "WGT": [80], "HT": [200], "M1F2": [1], "A1V2": [2]})
    events = pd.DataFrame({"ID": [10], "TIME": [0.], "INFUSIONRATE": [10.],
                           "BOLUS": [0.], "ISMEASUREMENT": [0]})
    with patch("load_data.patient_ids", return_value=[10]):
        with np.testing.assert_raises_regex(ValueError, "No measured concentrations for ID 10"):
            load_data(tables={"observations": observations, "events": events})
