"""Synthetic tests for the published-equation TOML consumer (no BSON needed)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from published_equations import load_equations, published_parameters, published_rates, _evaluate
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
    male_cov = jnp.array([40., 70., 25., -0.5, -0.5])
    female_cov = jnp.array([40., 70., 25., 0.5, -0.5])
    male = published_parameters(equations, male_cov)
    female = published_parameters(equations, female_cov)
    np.testing.assert_allclose(published_rates(equations, male_cov), [0.002, 0.003, 0.001, 0.002, 0.0003, 10000.], rtol=1e-6)
    np.testing.assert_allclose(published_rates(equations, female_cov), [0.002, 0.003, 0.002, 0.002, 0.0006, 10000.], rtol=1e-6)
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


def test_sex_specific_expressions_override_unused_common_equations(tmp_path):
    """Generic k13/k31 must not be evaluated for sex-specific predictions."""
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]", 'k10 = "0.002"', 'k12 = "0.003"',
             'k13 = "1 / (x_age - x_age)"', 'k21 = "0.002"',
             'k31 = "1 / (x_age - x_age)"', 'V1 = "10000"',
             "[equations.male]", 'k13 = "0.001"', 'k31 = "0.0003"',
             "[equations.female]", 'k13 = "0.002"', 'k31 = "0.0006"']
    path = tmp_path / "equations.toml"
    path.write_text("\n".join(lines))
    params = published_parameters(load_equations(path), jnp.array([40., 70., 25., -0.5, -0.5]))
    assert np.isfinite(params).all()


def test_unused_sex_branch_singularity_does_not_poison_gradient(tmp_path):
    """A singular unused female branch must not affect male derivatives."""
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]", 'k10 = "0.002"', 'k12 = "0.003"',
             'k13 = "0.001"', 'k21 = "0.002"', 'k31 = "0.0003"',
             'V1 = "10000"', "[equations.male]",
             'k13 = "0.001 * x_age / 40"', 'k31 = "0.0003"',
             "[equations.female]",
             'k13 = "1 / (x_age - x_age)"', 'k31 = "0.0006"']
    path = tmp_path / "equations.toml"
    path.write_text("\n".join(lines))
    equations = load_equations(path)
    covariates = jnp.array([40., 70., 25., -0.5, -0.5])
    result = published_parameters(equations, covariates)
    gradient = jax.jacrev(lambda x: published_parameters(equations, x))(covariates)
    assert np.isfinite(result).all()
    assert np.isfinite(gradient).all()
    assert np.any(np.abs(np.asarray(gradient[:, 0])) > 0)


def test_batched_unused_sex_branch_singularity_does_not_poison_gradients(tmp_path):
    """A mixed-sex batch must retain finite gradients at an inactive singularity."""
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]", 'k10 = "0.002"', 'k12 = "0.003"',
             'k13 = "0.001"', 'k21 = "0.002"', 'k31 = "0.0003"',
             'V1 = "10000"', "[equations.male]",
             'k13 = "0.001 * x_age / 40"', 'k31 = "0.0003"',
             "[equations.female]",
             'k13 = "1 / (x_age - 40)"', 'k31 = "0.0006"']
    path = tmp_path / "symbolic_equations.toml"
    path.write_text("\n".join(lines))
    equations = load_equations(path)
    patients = jnp.array([[40., 70., 25., -0.5, -0.5],
                          [50., 80., 27., 0.5, 0.5]])
    def summed_outputs(covariates):
        return jnp.sum(jax.vmap(lambda x: published_rates(equations, x))(covariates))
    rates = jax.vmap(lambda x: published_rates(equations, x))(patients)
    np.testing.assert_allclose(np.asarray(rates[:, 2]), [0.001, 0.1], rtol=1e-6)
    # Exercise the compiled batched path used in JAX-based inference as well.
    compiled_rates = jax.jit(jax.vmap(lambda x: published_rates(equations, x)))(patients)
    np.testing.assert_allclose(np.asarray(compiled_rates), np.asarray(rates), rtol=1e-6)
    compiled_gradient = jax.jit(jax.grad(summed_outputs))(patients)
    gradient = jax.grad(summed_outputs)(patients)
    np.testing.assert_allclose(np.asarray(compiled_gradient), np.asarray(gradient), rtol=1e-5, atol=1e-7)
    assert np.isfinite(np.asarray(gradient)).all(), "Inactive female branch poisoned batched male gradients"
    np.testing.assert_allclose(np.asarray(gradient[:, 0]), [0.001 / 40, -0.01], rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(gradient[:, 1:]), 0., atol=1e-7)
