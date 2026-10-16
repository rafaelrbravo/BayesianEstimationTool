"""Synthetic BET integration tests for fixed published covariate functions."""
import numpy as np
import jax
import jax.numpy as jnp

from bet_baseline import PARAMETERS
from published_bet_model import make_published_model, GlobalPriorFn
from published_equations import load_equations


def test_published_bet_model_predicts_and_differentiates(tmp_path):
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]", 'k10 = "0.002"', 'k12 = "0.003"',
             'k13 = "0.001"', 'k21 = "0.002"', 'k31 = "0.0003"',
             'V1 = "10000 + 10*x_wgt"',
             "[equations.male]", 'k13 = "0.001"', 'k31 = "0.0003"',
             "[equations.female]", 'k13 = "0.002"', 'k31 = "0.0006"']
    path = tmp_path / "equations.toml"
    path.write_text("\n".join(lines))
    model = make_published_model(load_equations(path))
    data = {"covariates": jnp.array([40., 70., 25., -0.5, -0.5]),
            "time": jnp.array([0., 60., 120.]),
            "rate": jnp.array([100., 100., 0.]),
            "bolus": jnp.array([10000., 0., 0.])}
    z = {name: jnp.array(0.) for name in PARAMETERS}
    global_params = {"spread": jnp.ones(6) * 0.2, "sigma": jnp.array(0.1)}
    result = model(global_params, z, data)["concentration"]
    assert result.shape == (3,)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0], 10000. / 10700., rtol=1e-5)
    gradient = jax.grad(lambda spread: model({**global_params, "spread": spread},
                                               {**z, "V1": jnp.array(1.)}, data)["concentration"].sum())(global_params["spread"])
    assert np.isfinite(gradient).all()


def test_published_prior_only_infers_variability_and_noise():
    assert set(GlobalPriorFn({})) == {"spread", "sigma"}


def test_published_model_local_effect_changes_initial_concentration(tmp_path):
    """A positive V1 random effect must lower the initial bolus concentration."""
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]", 'k10 = "0.002"', 'k12 = "0.003"',
             'k13 = "0.001"', 'k21 = "0.002"', 'k31 = "0.0003"',
             'V1 = "10000"',
             "[equations.male]", 'k13 = "0.001"', 'k31 = "0.0003"',
             "[equations.female]", 'k13 = "0.002"', 'k31 = "0.0006"']
    path = tmp_path / "equations.toml"
    path.write_text("\n".join(lines))
    model = make_published_model(load_equations(path))
    data = {"covariates": jnp.array([40., 70., 25., -0.5, -0.5]),
            "time": jnp.array([0., 60.]), "rate": jnp.zeros(2),
            "bolus": jnp.array([10000., 0.])}
    global_params = {"spread": jnp.ones(6) * 0.2, "sigma": jnp.array(0.1)}
    zero = {name: jnp.array(0.) for name in PARAMETERS}
    increased = {**zero, "V1": jnp.array(1.)}
    base = model(global_params, zero, data)["concentration"][0]
    shifted = model(global_params, increased, data)["concentration"][0]
    np.testing.assert_allclose(base, 1., rtol=1e-5)
    np.testing.assert_allclose(shifted / base, np.exp(-0.2), rtol=1e-5)


def test_published_model_gradient_with_singular_unused_sex_branch(tmp_path):
    """Full PK predictions remain differentiable when the other sex branch is singular."""
    lines = ['output_order = ["k10", "k12", "k13", "k21", "k31", "V1"]',
             "[equations.common]", 'k10 = "0.002"', 'k12 = "0.003"',
             'k13 = "0.001"', 'k21 = "0.002"', 'k31 = "0.0003"',
             'V1 = "10000"', "[equations.male]",
             'k13 = "0.001 * x_age / 40"', 'k31 = "0.0003"',
             "[equations.female]",
             'k13 = "1 / (x_age - x_age)"', 'k31 = "0.0006"']
    path = tmp_path / "equations.toml"
    path.write_text("\n".join(lines))
    model = make_published_model(load_equations(path))
    data = {"covariates": jnp.array([40., 70., 25., -0.5, -0.5]),
            "time": jnp.array([0., 60., 120.]),
            "rate": jnp.array([100., 100., 0.]),
            "bolus": jnp.array([10000., 0., 0.])}
    z = {name: jnp.array(0.) for name in PARAMETERS}
    global_params = {"spread": jnp.ones(6) * 0.2, "sigma": jnp.array(0.1)}
    def prediction(age):
        covariates = data["covariates"].at[0].set(age)
        return model(global_params, z, {**data, "covariates": covariates})["concentration"][-1]
    assert np.isfinite(prediction(40.))
    assert np.isfinite(jax.grad(prediction)(40.))
