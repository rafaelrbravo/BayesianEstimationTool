"""Regression tests for raw-covariate and physical-output preflight."""
import ast

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
    equations["common"]["V1"] = ast.parse("-10000", mode="eval")
    with pytest.raises(ValueError, match="nonfinite or nonpositive"):
        validate_published_patients(equations, np.array([[40., 70., 25., -0.5, -0.5]]))


def test_reject_nonfinite_symbolic_output(equations):
    # A syntactically valid symbolic expression can be singular for real patients.
    equations["common"]["V1"] = ast.parse("1 / (x_age - x_age)", mode="eval")
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="nonfinite or nonpositive"):
            validate_published_patients(equations, np.array([[40., 70., 25., -0.5, -0.5]]))


def test_reject_wrong_covariate_shape(equations):
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_published_patients(equations, np.array([40., 70., 25., -0.5, -0.5]))


def test_reject_empty_cohort(equations):
    with pytest.raises(ValueError, match="At least one patient"):
        validate_published_patients(equations, np.empty((0, 5)))


def test_reject_nonphysical_female_specific_equation(equations):
    # Common equations can be valid while a sex-specific branch is nonphysical.
    equations["female"]["k31"] = ast.parse("-0.0006", mode="eval")
    with pytest.raises(ValueError, match="nonfinite or nonpositive"):
        validate_published_patients(equations, np.array([[65., 80., 30., 0.5, 0.5]]))


def test_reject_nonphysical_male_specific_equation(equations):
    equations["male"]["k13"] = ast.parse("0", mode="eval")
    with pytest.raises(ValueError, match="nonfinite or nonpositive"):
        validate_published_patients(equations, np.array([[40., 70., 25., -0.5, -0.5]]))
