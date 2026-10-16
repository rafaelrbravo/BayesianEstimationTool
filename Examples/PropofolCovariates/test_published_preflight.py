"""Regression tests for raw-covariate and physical-output preflight."""
import numpy as np
import pytest

from compare_published import validate_published_patients
from published_equations import load_equations


@pytest.fixture
def equations(tmp_path):
    path = tmp_path / "equations.toml"
    path.write_text("\n".join([
        'output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
        "[equations.common]",
        'k10 = "0.002"', 'k12 = "0.003"', 'k13 = "0.001"',
        'k21 = "0.002"', 'k31 = "0.0003"', 'V1 = "10000"',
        "[equations.male]", 'k13 = "0.001"', 'k31 = "0.0003"',
        "[equations.female]", 'k13 = "0.002"', 'k31 = "0.0006"',
    ]))
    return load_equations(path)


def test_valid_raw_covariates(equations):
    validate_published_patients(equations, np.array([
        [40., 70., 25., -0.5, -0.5],
        [65., 80., 30., 0.5, 0.5],
    ]))


@pytest.mark.parametrize("row", [
    [40., 70., 25., 0., -0.5],
    [40., 70., 25., -0.5, 1.],
    [40., -70., 25., -0.5, -0.5],
    [np.nan, 70., 25., -0.5, -0.5],
])
def test_reject_invalid_covariates(equations, row):
    with pytest.raises(ValueError):
        validate_published_patients(equations, np.array([row]))


def test_reject_nonphysical_symbolic_output(equations):
    equations["common"]["V1"] = __import__("ast").parse("-10000", mode="eval")
    with pytest.raises(ValueError, match="nonfinite or nonpositive"):
        validate_published_patients(equations, np.array([[40., 70., 25., -0.5, -0.5]]))
