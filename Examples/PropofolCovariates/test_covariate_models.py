"""Synthetic checks that covariate effects change only the covariate model."""
import numpy as np
import jax.numpy as jnp

from compare_covariates import make_model
from bet_baseline import PARAMETERS


def test_covariates_affect_only_covariate_model():
    global_params = {
        "logTypical": jnp.log(jnp.array([10., 20., 100., 1.5, 2., 1.])),
        "spread": jnp.zeros(6),
        "covariateEffects": jnp.zeros((6, 5)).at[0, 0].set(0.5),
    }
    local_z = {name: jnp.array(0.) for name in PARAMETERS}
    event = {"time": jnp.array([0., 60.]),
             "rate": jnp.array([0., 0.]),
             "bolus": jnp.array([10000., 0.])}
    low = dict(event, covariates=jnp.array([0., 0., 0., 0., 0.]))
    high = dict(event, covariates=jnp.array([1., 0., 0., 0., 0.]))
    baseline = make_model(False)
    covariate = make_model(True)
    np.testing.assert_allclose(baseline(global_params, local_z, low)["concentration"],
                               baseline(global_params, local_z, high)["concentration"], rtol=1e-6)
    predicted_low = covariate(global_params, local_z, low)["concentration"]
    predicted_high = covariate(global_params, local_z, high)["concentration"]
    np.testing.assert_allclose(predicted_low[0] / predicted_high[0], np.exp(0.5), rtol=1e-5)
    assert np.all(np.isfinite(np.asarray(predicted_high)))


if __name__ == "__main__":
    test_covariates_affect_only_covariate_model()
    print("Covariate model smoke checks passed")
