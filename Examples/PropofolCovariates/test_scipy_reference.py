"""Independent SciPy solver tests with analytically checkable dosing cases."""
import numpy as np
from scipy_reference import predict_patient
import pandas as pd


def events(times, rates, boluses, measurements):
    return pd.DataFrame({"TIME": times, "INFUSIONRATE": rates, "BOLUS": boluses,
                         "ISMEASUREMENT": measurements})


def test_no_dose():
    p = [10., 20., 30., 1., 2., 3.]
    e = events([0., 60., 120.], [0., 0., 0.], [0., 0., 0.], [1, 1, 1])
    np.testing.assert_allclose(predict_patient(p, e), 0., atol=1e-12)


def test_immediate_bolus_and_decay():
    p = [10., 20., 30., 1., 2., 3.]
    e = events([0., 60., 120.], [0., 0., 0.], [10000., 0., 0.], [1, 1, 1])
    result = predict_patient(p, e)
    np.testing.assert_allclose(result[0], 1., rtol=1e-12)
    assert np.all(np.isfinite(result)) and np.all(result >= 0.)
    assert result[-1] < result[1] < result[0]


def test_constant_infusion_no_elimination():
    # With zero clearance and zero intercompartment flow, C(t) = rate*t/(V1*1000).
    # A small positive value is used for the unused peripheral volumes.
    p = [10., 20., 30., 0., 0., 0.]
    e = events([0., 10., 20.], [6000., 6000., 0.], [0., 0., 0.], [1, 1, 1])
    np.testing.assert_allclose(predict_patient(p, e), [0., 0.1, 0.2], atol=1e-12)


def test_repeated_measurements_at_same_event():
    p = [10., 20., 30., 1., 2., 3.]
    e = events([0., 10.], [0., 0.], [10000., 0.], [2, 1])
    result = predict_patient(p, e)
    assert len(result) == 3
    np.testing.assert_allclose(result[:2], [1., 1.], rtol=1e-12)


def test_unsorted_events_rejected():
    p = [10., 20., 30., 1., 2., 3.]
    e = events([10., 0.], [0., 0.], [0., 0.], [1, 1])
    import pytest
    with pytest.raises(ValueError, match="Unsorted"):
        predict_patient(p, e)


def test_invalid_parameters_rejected():
    import pytest
    e = events([0., 10.], [0., 0.], [0., 0.], [1, 1])
    for parameters in ([0., 20., 30., 1., 2., 3.], [10., 20., 30., -1., 2., 3.],
                       [10., 20., 30., float("nan"), 2., 3.]):
        with pytest.raises(ValueError, match="PK volumes"):
            predict_patient(parameters, e)


def test_invalid_dosing_and_counts_rejected():
    import pytest
    p = [10., 20., 30., 1., 2., 3.]
    for rates, boluses, counts in [([-1., 0.], [0., 0.], [1, 1]),
                                   ([0., 0.], [-1., 0.], [1, 1]),
                                   ([0., 0.], [0., 0.], [1.5, 1]),
                                   ([float("nan"), 0.], [0., 0.], [1, 1])]:
        with pytest.raises(ValueError, match="Invalid dosing"):
            predict_patient(p, events([0., 10.], rates, boluses, counts))
