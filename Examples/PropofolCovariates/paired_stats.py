"""Patient-level paired uncertainty summaries, independent of BET/JAX."""
import numpy as np


def paired_bootstrap_interval(delta, seed=0, n_boot=2000):
    """Percentile interval for mean paired differences across patients."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or not len(delta) or not np.all(np.isfinite(delta)):
        raise ValueError("Expected nonempty finite patient-level differences")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    return np.quantile(means, [0.025, 0.975])
