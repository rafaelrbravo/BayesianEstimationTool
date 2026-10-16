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
    np.testing.assert_allclose(data["time"][1], [0., 60., 60.])
    np.testing.assert_allclose(data["valid"][1], [1, 1, 0])
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


if __name__ == "__main__":
    test_padded_events_and_covariates()
    test_rejects_mismatched_measurements()
    print("Synthetic loader checks passed")
