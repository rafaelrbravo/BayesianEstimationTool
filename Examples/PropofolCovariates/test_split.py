"""Patient-level split invariants for the propofol comparison."""
import numpy as np
import pytest

from split import make_split, eligible_test_indices, N_PATIENTS, N_TEST, SEED


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


def test_eligibility_precedes_development_limit():
    ids = [10, 11, 12, 13, 14]
    counts = {10: 2, 11: 4, 12: 3, 13: 5, 14: 6}
    test = np.array([0, 1, 2, 3, 4])
    assert eligible_test_indices(test, ids, counts, limit=2) == [1, 3]
    assert eligible_test_indices(test, ids, counts) == [1, 3, 4]
    assert eligible_test_indices(test, ids, counts, n_fit=4) == [3, 4]


@pytest.mark.parametrize("n_patients,n_test", [(True, 1), (10, True), (np.bool_(False), 1)])
def test_boolean_patient_counts_rejected(n_patients, n_test):
    with pytest.raises(ValueError, match="integers"):
        make_split(n_patients=n_patients, n_test=n_test)


@pytest.mark.parametrize("seed", [True, np.bool_(False), -1, 1.5, "20261015"])
def test_invalid_split_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        make_split(n_patients=20, n_test=5, seed=seed)
