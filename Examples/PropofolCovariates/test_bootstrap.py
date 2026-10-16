"""Offline regression tests for paired patient-level bootstrap inference."""
import numpy as np
import pytest

from paired_stats import paired_bootstrap_interval


def test_reproducible_interval():
    delta = np.array([-0.5, -0.2, 0.1, 0.4, 0.8])
    np.testing.assert_array_equal(paired_bootstrap_interval(delta), paired_bootstrap_interval(delta))


def test_constant_effect():
    np.testing.assert_allclose(paired_bootstrap_interval(np.full(12, -0.3)), [-0.3, -0.3])


def test_single_patient():
    np.testing.assert_allclose(paired_bootstrap_interval([0.7]), [0.7, 0.7])


@pytest.mark.parametrize("delta", [[], [np.nan], [np.inf], [[1., 2.]]])
def test_invalid_differences(delta):
    with pytest.raises(ValueError):
        paired_bootstrap_interval(delta)


def test_invalid_bootstrap_count():
    with pytest.raises(ValueError):
        paired_bootstrap_interval([1., 2.], n_boot=0)
