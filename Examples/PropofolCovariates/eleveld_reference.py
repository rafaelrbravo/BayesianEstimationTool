"""Evaluate the upstream Eleveld parameter predictions without Bayesian fitting.

Run after download_data.py:
    python Examples/PropofolCovariates/eleveld_reference.py

This reference is not a learned covariate model and is not a substitute
for the paired BET comparison. It checks solver/data consistency against
the supplied patient-specific Eleveld model parameters.
"""
import numpy as np
import jax.numpy as jnp

from bet_baseline import N_FIT, ScoreFn
from inspect_data import patient_ids, read_tables
from load_data import load_data
from pk_model import concentrations
from split import make_split


def evaluate():
    tables = read_tables()
    ids = patient_ids(tables)
    _, test = make_split(len(ids))
    test_ids = [ids[i] for i in test]
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    test_ids = [patient_id for patient_id in test_ids if counts.get(patient_id, 0) > N_FIT]
    data = load_data(selected_ids=test_ids, tables=tables)
    params = tables["parameters"].set_index("ID").loc[data["patient_ids"],
        ["V1", "V2", "V3", "CL", "Q2", "Q3"]].to_numpy(dtype=np.float32)
    scores = []
    for i, parameter in enumerate(params):
        prediction = concentrations(jnp.asarray(parameter), jnp.asarray(data["time"][i]),
                                    jnp.asarray(data["rate"][i]), jnp.asarray(data["bolus"][i]))
        patient = {name: jnp.asarray(data[name][i]) for name in ("observed", "concentration")}
        scores.append(float(ScoreFn({"concentration": prediction}, patient)))
    scores = np.asarray(scores)
    print(f"Eleveld reference, {len(scores)} held-out patients with >{N_FIT} measurements")
    print(f"Future-only mean absolute log error: {scores.mean():.4f}")
    print(f"Future-only median patient error: {np.median(scores):.4f}")
    return data["patient_ids"], scores


if __name__ == "__main__":
    evaluate()
