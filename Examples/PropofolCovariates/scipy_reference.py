"""Independent SciPy Eleveld reference (no BET, JAX, or fitting)."""
import argparse
import numpy as np
from scipy.linalg import expm
from inspect_data import patient_ids, read_tables
from split import make_split, eligible_test_indices


def predict_patient(parameters, events):
    v1, v2, v3, cl, q2, q3 = map(float, parameters)
    cl, q2, q3 = cl / 60., q2 / 60., q3 / 60.
    a = np.array([[-(cl + q2 + q3) / v1, q2 / v2, q3 / v3],
                  [q2 / v1, -q2 / v2, 0.], [q3 / v1, 0., -q3 / v3]])
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = a
    augmented[0, 3] = 1.
    amount = np.zeros(3)
    predictions = []
    rows = events[["TIME", "INFUSIONRATE", "BOLUS", "ISMEASUREMENT"]].to_numpy(dtype=float)
    for i, (time, rate, bolus, measured) in enumerate(rows):
        amount[0] += bolus
        predictions.extend([amount[0] / (v1 * 1000.)] * int(measured))
        if i + 1 < len(rows):
            duration = rows[i + 1, 0] - time
            if duration < 0:
                raise ValueError("Unsorted event times")
            if duration > 0:
                amount = (expm(augmented * duration) @ np.r_[amount, rate])[:3]
    return np.asarray(predictions)


def evaluate(development=True, n_fit=3):
    tables = read_tables()
    ids = patient_ids(tables)
    _, test = make_split(len(ids))
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    selected = eligible_test_indices(test, ids, counts, n_fit=n_fit, limit=25 if development else None)
    parameters = tables["parameters"].set_index("ID")
    errors = []
    for index in selected:
        patient_id = ids[index]
        events = tables["events"].loc[tables["events"].ID == patient_id]
        observed = tables["observations"].loc[(tables["observations"].ID == patient_id) &
                                               (tables["observations"].CP > 0), "CP"].to_numpy(dtype=float)
        prediction = predict_patient(parameters.loc[patient_id, ["V1", "V2", "V3", "CL", "Q2", "Q3"]], events)
        if len(prediction) != len(observed) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid reference prediction for patient {patient_id}")
        errors.append(np.mean(np.abs(np.log(np.maximum(prediction[n_fit:], 1e-6)) -
                                     np.log(observed[n_fit:]))))
    print(f"SciPy Eleveld reference: {len(errors)} held-out patients")
    print(f"Mean future absolute log error: {np.mean(errors):.4f}")
    print(f"Median patient future absolute log error: {np.median(errors):.4f}")
    return np.asarray(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    evaluate(development=not args.full)
