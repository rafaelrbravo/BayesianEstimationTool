"""Synthetic checks for Eleveld reference parameter alignment."""
import numpy as np
import pandas as pd
import pytest

from eleveld_reference import reference_parameters


def example_table():
    return pd.DataFrame({
        "ID": [20, 10],
        "V1": [20., 10.], "V2": [21., 11.], "V3": [22., 12.],
        "CL": [2., 1.], "Q2": [3., 1.5], "Q3": [4., 2.],
    })


def test_reference_alignment_uses_ids_not_csv_row_order():
    result = reference_parameters(example_table(), [10, 20])
    np.testing.assert_allclose(result[0], [10., 11., 12., 1., 1.5, 2.])
    np.testing.assert_allclose(result[1], [20., 21., 22., 2., 3., 4.])


def test_reference_rejects_missing_id():
    with pytest.raises(ValueError, match="Missing Eleveld parameters"):
        reference_parameters(example_table(), [30])


def test_reference_rejects_nonpositive_parameter():
    table = example_table()
    table.loc[0, "V1"] = 0.
    with pytest.raises(ValueError, match="positive and finite"):
        reference_parameters(table, [20])


def test_reference_rejects_nonfinite_parameter():
    table = example_table()
    table.loc[0, "CL"] = np.nan
    with pytest.raises(ValueError, match="positive and finite"):
        reference_parameters(table, [20])


def test_reference_rejects_duplicate_patient_id():
    table = pd.concat([example_table(), example_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate Eleveld parameter rows"):
        reference_parameters(table, [20])
