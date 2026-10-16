"""BET model with fixed published symbolic covariate functions.

This is a reproduction/compatibility model, NOT an independent held-out
evaluation: the upstream symbolic network was fitted on overlapping patients.
Pass raw covariates, not the exploratory model's standardized covariates.
"""
import jax.numpy as jnp
import numpyro.distributions as dist

from bet_baseline import PARAMETERS
from pk_model import concentrations
from published_equations import load_equations, published_parameters


def make_published_model(equations):
    """Build a BET ModelFn with patient-specific published typical PK parameters."""
    def ModelFn(globalParams, localZs, data):
        typical = published_parameters(equations, data["covariates"])
        local = jnp.stack([localZs[name] for name in PARAMETERS])
        params = typical * jnp.exp(globalParams["spread"] * local)
        return {"concentration": concentrations(params, data["time"], data["rate"], data["bolus"])}
    return ModelFn


def GlobalPriorFn(dataShapes):
    """Infer only variability and observation noise, not published coefficients."""
    return {
        "spread": dist.HalfNormal(0.5).expand((6,)).to_event(1),
        "sigma": dist.HalfNormal(0.5),
    }


def build_published_model(path=None):
    """Load validated equations from TOML and construct a BET-compatible model."""
    return make_published_model(load_equations(path))
