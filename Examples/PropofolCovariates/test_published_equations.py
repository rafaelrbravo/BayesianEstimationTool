"""Synthetic tests for the published-equation TOML consumer (no BSON needed)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from published_equations import load_equations, published_parameters, _evaluate
import ast


def test_load_and_evaluate_synthetic_equations(tmp_path):
    common = {"k10": "0.002", "k12": "0.003", "k13": "0.001",
              "k21": "0.002", "k31": "0.0003", "V1": "10000.0"}
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]', "", "[equations.common]"]
    lines += [f'{key} = "{value}"' for key, value in common.items()]
    lines += ["", "[equations.male]", 'k13 = "0.001"', 'k31 = "0.0003"',
              "", "[equations.female]", 'k13 = "0.002"', 'k31 = "0.0006"']
    path = tmp_path / "symbolic_equations.toml"
    path.write_text("\n".join(lines))
    equations = load_equations(path)
    male = published_parameters(equations, jnp.array([40., 70., 25., -0.5, -0.5]))
    female = published_parameters(equations, jnp.array([40., 70., 25., 0.5, -0.5]))
    np.testing.assert_allclose(male, [10., 15., 100./3., 1.2, 1.8, 0.6], rtol=1e-5)
    np.testing.assert_allclose(female, [10., 15., 100./3., 1.2, 1.8, 1.2], rtol=1e-5)
    assert np.isfinite(jax.jacrev(lambda x: published_parameters(equations, x))(jnp.array([40., 70., 25., -0.5, -0.5]))).all()


def test_julia_powers_and_abs():
    expr = ast.parse("abs(x_age - 50)^2".replace("^", "**"), mode="eval")
    np.testing.assert_allclose(_evaluate(expr, {"x_age": jnp.array(48.)}), 4.)


@pytest.mark.parametrize("expression", ["__import__('os')", "x_age.real", "min(1, 2)", "x_age[0]", "x_age // 2"])
def test_reject_unsafe_syntax(expression):
    with pytest.raises(ValueError):
        _evaluate(ast.parse(expression, mode="eval"), {"x_age": jnp.array(1.)})


def test_loading_does_not_evaluate_at_arbitrary_inputs(tmp_path):
    """A valid equation may be singular at x_age=1, used by old syntax checks."""
    path = tmp_path / "equations.toml"
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]",
             'k10 = "1 / (x_age - 1)"',
             'k12 = "0.003"', 'k13 = "0.001"', 'k21 = "0.002"',
             'k31 = "0.0003"', 'V1 = "10000"',
             "[equations.male]", 'k13 = "0.001"', 'k31 = "0.0003"',
             "[equations.female]", 'k13 = "0.002"', 'k31 = "0.0006"']
    path.write_text("\\n".join(lines).replace("\\n", "\n"))
    equations = load_equations(path)
    result = published_parameters(equations, jnp.array([40., 70., 25., -0.5, -0.5]))
    assert np.isfinite(result).all()
