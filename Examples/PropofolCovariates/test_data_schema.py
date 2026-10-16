"""Regression tests for upstream propofol data schema normalization."""
import pandas as pd
import pytest

from inspect_data import read_tables


def write_tables(directory, observations):
    observations.to_csv(directory / "dataframe.csv", index=False)
    pd.DataFrame({"ID": [1], "TIME": [0.], "INFUSIONRATE": [0.],
                  "BOLUS": [0.], "ISMEASUREMENT": [1]}).to_csv(
                      directory / "eleveld_infusiondata.csv", index=False)
    pd.DataFrame({"ID": [1], "V1": [10.], "V2": [20.], "V3": [30.],
                  "CL": [1.], "Q2": [2.], "Q3": [3.]}).to_csv(
                      directory / "eleveld_modelparams.csv", index=False)


@pytest.mark.parametrize("height_column", ["HT", "HGT", " hgt "])
def test_read_tables_normalizes_height_column(tmp_path, height_column):
    observations = pd.DataFrame({"ID": [1], "CP": [1.], height_column: [170.]})
    write_tables(tmp_path, observations)

    loaded = read_tables(tmp_path)["observations"]

    assert "HT" in loaded.columns
    assert "HGT" not in loaded.columns
    assert loaded.loc[0, "HT"] == 170.


def test_read_tables_rejects_conflicting_height_columns(tmp_path):
    observations = pd.DataFrame({"ID": [1], "CP": [1.], "HT": [170.], "HGT": [180.]})
    write_tables(tmp_path, observations)

    with pytest.raises(ValueError, match="conflicting HT and HGT"):
        read_tables(tmp_path)


def test_read_tables_accepts_matching_duplicate_height_columns(tmp_path):
    observations = pd.DataFrame({"ID": [1], "CP": [1.], "HT": [170.], "HGT": [170.]})
    write_tables(tmp_path, observations)

    loaded = read_tables(tmp_path)["observations"]

    assert list(loaded.columns).count("HT") == 1
    assert "HGT" not in loaded.columns
