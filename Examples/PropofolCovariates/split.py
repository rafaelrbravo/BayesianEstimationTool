"""Reproducible patient split for the propofol covariate experiment.

Run from the BET repository root:
    python Examples/PropofolCovariates/split.py

The published source contains 1,031 individuals. Split *patients*, not
measurements, before fitting scalers, selecting covariates, or estimating
model parameters.
"""
import numpy as np

N_PATIENTS = 1031
N_TEST = 200
SEED = 20261015


def make_split(n_patients=N_PATIENTS, n_test=N_TEST, seed=SEED):
    if (isinstance(n_patients, (bool, np.bool_)) or isinstance(n_test, (bool, np.bool_)) or
            not isinstance(n_patients, (int, np.integer)) or not isinstance(n_test, (int, np.integer))):
        raise ValueError("Patient and test counts must be integers")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < n_test < n_patients:
        raise ValueError("n_test must be between 1 and n_patients - 1")
    indices = np.random.default_rng(seed).permutation(n_patients)
    train, test = indices[n_test:], indices[:n_test]
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.union1d(train, test)) == n_patients
    return train, test


def eligible_test_indices(test_indices, patient_ids, measurement_counts, n_fit=3, limit=None):
    """Keep held-out patients with future observations, preserving split order."""
    eligible = [i for i in test_indices if measurement_counts.get(patient_ids[i], 0) > n_fit]
    return eligible if limit is None else eligible[:limit]


if __name__ == "__main__":
    train, test = make_split()
    print(f"Full split: {len(train)} train, {len(test)} test")
    print(f"Development candidates: {len(train[:100])} train, {len(test)} held-out; select first 25 eligible held-out patients after checking measurement counts")
