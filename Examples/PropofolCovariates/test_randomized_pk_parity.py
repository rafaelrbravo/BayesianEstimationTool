"""Randomized real-reference SciPy/JAX parity without patient data or NumPyro."""
import numpy as np
import pandas as pd
import jax.numpy as jnp

from pk_model import concentrations
from scipy_reference import predict_patient


def test_randomized_jax_scipy_pk_parity():
    rng = np.random.default_rng(20261016)
    low = np.array([5., 10., 30., .2, .2, .2])
    high = np.array([30., 70., 300., 4., 5., 5.])
    for _ in range(30):
        parameters = np.exp(rng.uniform(np.log(low), np.log(high)))
        time = np.cumsum(rng.choice([0, 1, 5, 30, 120, 300], size=18)).astype(float)
        rate = rng.uniform(0, 200, size=len(time))
        bolus = rng.choice([0, 0, 0, 1000, 5000], size=len(time)).astype(float)
        events = pd.DataFrame({"TIME": time, "INFUSIONRATE": rate, "BOLUS": bolus,
                               "ISMEASUREMENT": np.ones(len(time), dtype=int)})
        expected = predict_patient(parameters, events)
        actual = np.asarray(concentrations(jnp.asarray(parameters), jnp.asarray(time),
                                           jnp.asarray(rate), jnp.asarray(bolus)))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)


def test_repeated_measurement_counts_match_expanded_jax_events():
    parameters = np.array([10., 20., 30., 1., 2., 3.])
    events = pd.DataFrame({"TIME": [0., 60., 60., 120.],
                           "INFUSIONRATE": [100., 100., 0., 0.],
                           "BOLUS": [1000., 0., 2000., 0.],
                           "ISMEASUREMENT": [2, 0, 1, 2]})
    expected = predict_patient(parameters, events)
    time = np.array([0., 0., 60., 60., 120., 120.])
    rate = np.array([100., 100., 100., 0., 0., 0.])
    bolus = np.array([1000., 0., 0., 2000., 0., 0.])
    actual = np.asarray(concentrations(jnp.asarray(parameters), jnp.asarray(time),
                                           jnp.asarray(rate), jnp.asarray(bolus)))
    np.testing.assert_allclose(actual[[0, 1, 3, 4, 5]], expected, rtol=2e-4, atol=2e-6)


def test_loader_expansion_matches_production_scipy_reference(monkeypatch):
    import load_data as loader

    monkeypatch.setattr(loader, "patient_ids", lambda tables: [1])
    parameters = np.array([10., 20., 30., 1., 2., 3.])
    events = pd.DataFrame({"ID": [1] * 4, "TIME": [0., 60., 60., 120.],
                           "INFUSIONRATE": [100., 100., 0., 0.],
                           "BOLUS": [1000., 0., 2000., 0.],
                           "ISMEASUREMENT": [2, 0, 1, 2]})
    observations = pd.DataFrame({"ID": [1] * 5, "CP": [1.] * 5,
                                 "AGE": [40.] * 5, "WGT": [70.] * 5,
                                 "HT": [170.] * 5, "M1F2": [1] * 5,
                                 "A1V2": [1] * 5})
    data = loader.load_data(tables={"events": events, "observations": observations})
    actual = np.asarray(concentrations(jnp.asarray(parameters), jnp.asarray(data["time"][0]),
                                           jnp.asarray(data["rate"][0]), jnp.asarray(data["bolus"][0])))
    expected = predict_patient(parameters, events)
    np.testing.assert_allclose(actual[data["observed"][0].astype(bool)], expected, rtol=2e-4, atol=2e-6)


def test_padding_and_unmeasured_events_do_not_change_measured_predictions(monkeypatch):
    import load_data as loader

    monkeypatch.setattr(loader, "patient_ids", lambda tables: [1, 2])
    events = pd.DataFrame({"ID": [1, 1, 1, 2, 2, 2, 2, 2],
                           "TIME": [0., 60., 120., 0., 30., 60., 90., 120.],
                           "INFUSIONRATE": [100., 50., 0., 100., 80., 50., 20., 0.],
                           "BOLUS": [1000., 0., 0., 1000., 0., 0., 0., 0.],
                           "ISMEASUREMENT": [1, 0, 1, 1, 0, 0, 0, 1]})
    observations = pd.DataFrame({"ID": [1, 1, 2, 2], "CP": [1.] * 4,
                                 "AGE": [40.] * 4, "WGT": [70.] * 4,
                                 "HT": [170.] * 4, "M1F2": [1] * 4,
                                 "A1V2": [1] * 4})
    data = loader.load_data(tables={"events": events, "observations": observations})
    parameters = jnp.asarray([10., 20., 30., 1., 2., 3.])
    for i, patient_id in enumerate(data["patient_ids"]):
        actual = np.asarray(concentrations(parameters, jnp.asarray(data["time"][i]),
                                           jnp.asarray(data["rate"][i]), jnp.asarray(data["bolus"][i])))
        reference_events = events.loc[events.ID == patient_id]
        expected = predict_patient(np.asarray(parameters), reference_events)
        np.testing.assert_allclose(actual[data["observed"][i].astype(bool)], expected,
                                   rtol=2e-4, atol=2e-6)
    assert np.any(data["valid"][0] == 0)
