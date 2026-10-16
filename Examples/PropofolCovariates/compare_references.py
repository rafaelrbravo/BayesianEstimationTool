"""Check patient-level agreement between independent SciPy and JAX PK references."""
import argparse
import numpy as np
import pandas as pd


def compare(jax_csv, scipy_csv, atol=0.02, rtol=0.02):
    left = pd.read_csv(jax_csv)
    right = pd.read_csv(scipy_csv)
    required = {"patient_id", "eleveld_future_log_error"}
    for name, table in ((jax_csv, left), (scipy_csv, right)):
        if not required.issubset(table.columns):
            raise ValueError(f"{name} is missing required columns")
        if table["patient_id"].isna().any() or table["patient_id"].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate patient IDs")
    if set(left.patient_id) != set(right.patient_id) or len(left) == 0:
        raise ValueError("Reference patient sets differ or are empty")
    aligned = left.merge(right, on="patient_id", validate="one_to_one", suffixes=("_jax", "_scipy"))
    jax = aligned["eleveld_future_log_error_jax"].to_numpy(dtype=float)
    scipy = aligned["eleveld_future_log_error_scipy"].to_numpy(dtype=float)
    if not np.isfinite(jax).all() or not np.isfinite(scipy).all():
        raise ValueError("Nonfinite reference errors")
    if np.any(jax < 0) or np.any(scipy < 0):
        raise ValueError("Reference absolute log errors must be nonnegative")
    differences = np.abs(jax - scipy)
    print(f"Compared {len(aligned)} held-out patients; maximum absolute score difference: {differences.max():.6g}")
    if not np.allclose(jax, scipy, atol=atol, rtol=rtol):
        worst = np.argsort(differences)[-5:][::-1]
        details = [(int(aligned.patient_id.iloc[i]), float(jax[i]), float(scipy[i])) for i in worst]
        raise AssertionError(f"JAX/SciPy reference mismatch (patient_id, JAX, SciPy): {details}")
    return differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jax_csv")
    parser.add_argument("scipy_csv")
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    compare(args.jax_csv, args.scipy_csv, atol=args.atol, rtol=args.rtol)
