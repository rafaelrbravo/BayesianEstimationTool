"""Evaluate the upstream Eleveld parameter predictions without Bayesian fitting.

Run after download_data.py:
    python Examples/PropofolCovariates/eleveld_reference.py

This reference is not a learned covariate model and is not a substitute
for the paired BET comparison. It checks solver/data consistency against
the supplied patient-specific Eleveld model parameters.
"""
import argparse
import csv
import numpy as np
import jax.numpy as jnp

from bet_baseline import N_FIT, ScoreFn
from inspect_data import patient_ids, read_tables
from load_data import load_data
from pk_model import concentrations
from split import make_split


def reference_parameters(parameters, patient_ids):
    """Align Eleveld parameters by patient ID, never by CSV row position."""
    columns = ["V1", "V2", "V3", "CL", "Q2", "Q3"]
    if parameters["ID"].duplicated().any():
        raise ValueError("Duplicate Eleveld parameter rows for a patient ID")
    indexed = parameters.set_index("ID")
    missing = set(patient_ids) - set(indexed.index)
    if missing:
        raise ValueError(f"Missing Eleveld parameters for IDs: {sorted(missing)[:5]}")
    values = indexed.loc[patient_ids, columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Eleveld parameters must be positive and finite")
    return values


def evaluate(development=False, output_csv=None):
    tables = read_tables()
    ids = patient_ids(tables)
    _, test = make_split(len(ids))
    if development:
        test = test[:25]
    test_ids = [ids[i] for i in test]
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    test_ids = [patient_id for patient_id in test_ids if counts.get(patient_id, 0) > N_FIT]
    data = load_data(selected_ids=test_ids, tables=tables)
    params = reference_parameters(tables["parameters"], data["patient_ids"])
    scores = []
    for i, parameter in enumerate(params):
        prediction = concentrations(jnp.asarray(parameter), jnp.asarray(data["time"][i]),
                                    jnp.asarray(data["rate"][i]), jnp.asarray(data["bolus"][i]))
        patient = {name: jnp.asarray(data[name][i]) for name in ("observed", "concentration")}
        scores.append(float(ScoreFn({"concentration": prediction}, patient)))
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite Eleveld reference errors")
    if output_csv is not None:
        with open(output_csv, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["patient_id", "eleveld_future_log_error"])
            writer.writerows((int(patient_id), float(score)) for patient_id, score in zip(data["patient_ids"], scores))
    print(f"Eleveld reference, {len(scores)} held-out patients with >{N_FIT} measurements")
    print(f"Future-only mean absolute log error: {scores.mean():.4f}")
    print(f"Future-only median patient error: {np.median(scores):.4f}")
    return data["patient_ids"], scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score supplied Eleveld PK parameters on held-out patients")
    parser.add_argument("--development", action="store_true", help="Use first 25 held-out patients, matching BET development comparison")
    parser.add_argument("--output-csv", help="Optional patient-level Eleveld reference errors")
    args = parser.parse_args()
    evaluate(development=args.development, output_csv=args.output_csv)
