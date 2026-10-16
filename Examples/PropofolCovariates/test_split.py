"""Patient-level split invariants for the propofol comparison."""
import numpy as np
import pytest

from split import make_split, N_PATIENTS, N_TEST, SEED


def test_split_is_reproducible_and_disjoint():
    train, test = make_split()
    train_again, test_again = make_split()
    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(test, test_again)
    assert len(train) == N_PATIENTS - N_TEST
    assert len(test) == N_TEST
    assert len(np.unique(np.concatenate([train, test]))) == N_PATIENTS
    assert len(np.intersect1d(train, test)) == 0


def test_development_subset_is_nested_in_full_split():
    train, test = make_split()
    assert len(train[:100]) == 100
    assert len(test[:25]) == 25
    assert len(np.intersect1d(train[:100], test[:25])) == 0


@pytest.mark.parametrize("n_test", [0, 10, -1])
def test_invalid_split_size(n_test):
    with pytest.raises(ValueError, match="n_test"):
        make_split(n_patients=10, n_test=n_test)


def test_custom_split_uses_requested_seed():
    train, test = make_split(n_patients=20, n_test=5, seed=SEED + 1)
    assert len(train) == 15
    assert len(test) == 5
    assert not np.array_equal(test, make_split(n_patients=20, n_test=5, seed=SEED)[1])
