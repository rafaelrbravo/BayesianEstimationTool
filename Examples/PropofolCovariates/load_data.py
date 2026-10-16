"""Build padded patient arrays for the published propofol PK experiment.

Event times are seconds, source infusion rates are micrograms/second (as
documented in the upstream symbreg/data/README.md), boluses are micrograms, and
plasma concentrations are micrograms/mL. The upstream Julia loader has a conflicting
ug/min comment, but the data documentation and simulator's seconds-based dynamics
support preserving the raw ug/s values. Event i records
a bolus and concentration at its time, then applies its infusion rate until
event i+1. Padding repeats the last time and has zero dosing and no mask.
"""
import numpy as np

from inspect_data import patient_ids, read_tables


def load_data(directory=None, selected_ids=None, tables=None):
    if tables is None:
        tables = read_tables() if directory is None else read_tables(directory)
    obs, events = tables["observations"], tables["events"]
    ids = patient_ids(tables)
    if selected_ids is not None:
        selected_ids = list(selected_ids)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Selected patient IDs contain duplicates")
        selected_ids = set(selected_ids)
        if not selected_ids.issubset(ids):
            raise ValueError("Selected patient IDs include unknown or excluded patients")
        ids = [patient_id for patient_id in ids if patient_id in selected_ids]
    if not ids:
        raise ValueError("No patients selected")
    required = {"TIME", "INFUSIONRATE", "BOLUS", "ISMEASUREMENT"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
    covariates = ["AGE", "WGT", "HT", "M1F2", "A1V2"]
    # The original Julia loader reads height from dataframe column HT; derive BMI and recode sex/route below.
    if not set(covariates).issubset(obs.columns):
        raise ValueError("Missing covariate columns")
    records = []
    for patient_id in ids:
        e = events.loc[events.ID == patient_id]
        o = obs.loc[(obs.ID == patient_id) & (obs.CP > 0)]
        if not (obs.ID == patient_id).any():
            raise ValueError(f"Missing covariate rows for ID {patient_id}")
        times = e.TIME.to_numpy(dtype=np.float64)
        # Upstream symbreg/data/README.md documents these values as ug/s.
        # Preserve them unchanged; FastPKSim integrates over seconds and outputs
        # ug/mL, so ug/s is dimensionally consistent with the model's units.
        rate = e.INFUSIONRATE.to_numpy(dtype=np.float64)
        bolus = e.BOLUS.to_numpy(dtype=np.float64)
        raw_counts = e.ISMEASUREMENT.to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(raw_counts)) or np.any(raw_counts < 0) or np.any(raw_counts != np.floor(raw_counts)):
            raise ValueError(f"Invalid measurement counts for ID {patient_id}")
        if np.any(raw_counts > np.iinfo(np.int32).max):
            raise ValueError(f"Measurement count overflow for ID {patient_id}")
        counts = raw_counts.astype(np.int32)
        # Reject inconsistent counts before expanding repeated measurements:
        # even a single int32-valid count could otherwise allocate billions of rows.
        n_measurements = int(counts.astype(np.int64).sum())
        if n_measurements != len(o):
            raise ValueError(f"Measurement count mismatch for ID {patient_id}: {len(o)} vs {n_measurements}")
        if not np.any(counts > 0):
            raise ValueError(f"No measured concentrations for ID {patient_id}")
        # Events after the last measured concentration cannot affect any
        # likelihood or prediction score; omit them before padding.
        last_measured_event = np.flatnonzero(counts > 0)[-1] + 1
        times, rate, bolus, counts = (values[:last_measured_event] for values in (times, rate, bolus, counts))
        # The upstream Julia get_indmeas repeats an event index when its
        # IsMeasurement count exceeds one. Expand the event into same-time
        # rows so every observation has a separate likelihood contribution.
        repeats = np.maximum(counts.astype(np.int64), 1)
        if repeats.sum() > np.iinfo(np.int32).max:
            raise ValueError(f"Expanded event count overflow for ID {patient_id}")
        first = np.repeat(np.cumsum(repeats) - repeats, repeats)
        times = np.repeat(times, repeats)
        rate = np.repeat(rate, repeats)
        bolus = np.repeat(bolus, repeats)
        mask = np.repeat(counts > 0, repeats).astype(np.int32)
        bolus[np.arange(len(bolus)) != first] = 0.
        concentration = o.CP.to_numpy(dtype=np.float64)
        # Enforce time ordering here, before JAX tracing and inference.
        if len(times) == 0 or np.any(np.diff(times) < 0):
            raise ValueError(f"Unsorted or missing event times for ID {patient_id}")
        if len(concentration) != int(counts.sum()):
            raise ValueError(f"Measurement count mismatch for ID {patient_id}: {len(concentration)} vs {mask.sum()}")
        if np.any(~np.isfinite(concentration)) or np.any(concentration <= 0):
            raise ValueError(f"Invalid measured concentration for ID {patient_id}")
        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(rate)) or np.any(~np.isfinite(bolus)) or np.any(rate < 0) or np.any(bolus < 0):
            raise ValueError(f"Invalid dosing for ID {patient_id}")
        cov_rows = obs.loc[obs.ID == patient_id, covariates].to_numpy(dtype=np.float64)
        if len(cov_rows) == 0:
            raise ValueError(f"Missing covariate rows for ID {patient_id}")
        if not np.all(np.isfinite(cov_rows)):
            raise ValueError(f"Missing covariate for ID {patient_id}")
        if not np.allclose(cov_rows, cov_rows[0], rtol=0, atol=1e-6):
            raise ValueError(f"Inconsistent patient covariates for ID {patient_id}")
        patient_cov = cov_rows[0]
        y = np.zeros(len(times))
        y[mask.astype(bool)] = concentration
        records.append((times, rate, bolus, mask, y, patient_cov))
    width = max(len(x[0]) for x in records)
    arrays = {key: np.zeros((len(ids), width), dtype=np.float32)
              for key in ("time", "rate", "bolus", "observed", "concentration", "valid")}
    cov = np.empty((len(ids), len(covariates)), dtype=np.float32)
    for i, (times, rate, bolus, mask, y, patient_cov) in enumerate(records):
        n = len(times)
        arrays["time"][i, :n] = times
        arrays["time"][i, n:] = times[-1]
        arrays["rate"][i, :n] = rate
        arrays["bolus"][i, :n] = bolus
        arrays["observed"][i, :n] = mask
        arrays["concentration"][i, :n] = y
        arrays["valid"][i, :n] = 1
        age, weight, height, sex, av = patient_cov
        if age <= 0 or height <= 0 or weight <= 0:
            raise ValueError(f"Invalid age/height/weight for ID {ids[i]}")
        if sex not in (1, 2) or av not in (1, 2):
            raise ValueError(f"Invalid sex/route encoding for ID {ids[i]}")
        bmi = weight / (height / 100.) ** 2
        if not np.isfinite(bmi) or bmi <= 0:
            raise ValueError(f"Invalid BMI for ID {ids[i]}")
        cov[i] = [age, weight, bmi, sex - 1.5, av - 1.5]
    arrays["covariates"] = cov
    arrays["patient_ids"] = np.asarray(ids)
    return arrays


if __name__ == "__main__":
    data = load_data()
    print("Patients:", len(data["patient_ids"]))
    print("Event array shape:", data["time"].shape)
    print("Measured concentrations:", int(data["observed"].sum()))
