"""Build padded patient arrays for the published propofol PK experiment.

All event times are seconds, infusion rates are micrograms/second, boluses
are micrograms, and plasma concentrations are micrograms/mL. Event i records
a bolus and concentration at its time, then applies its infusion rate until
event i+1. Padding repeats the last time and has zero dosing and no mask.
"""
import numpy as np

from inspect_data import patient_ids, read_tables


def load_data(directory=None):
    tables = read_tables() if directory is None else read_tables(directory)
    obs, events = tables["observations"], tables["events"]
    ids = patient_ids(tables)
    required = {"TIME", "INFUSIONRATE", "BOLUS", "ISMEASUREMENT"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
    covariates = ["AGE", "WGT", "HT", "M1F2", "A1V2"] 
    # The source symbolic-regression network uses AGE, WGT, BMI, sex, AV.
    if not set(covariates).issubset(obs.columns):
        raise ValueError("Missing covariate columns")
    records = []
    for patient_id in ids:
        e = events.loc[events.ID == patient_id]
        o = obs.loc[(obs.ID == patient_id) & (obs.CP > 0)]
        times = e.TIME.to_numpy(dtype=np.float64)
        rate = e.INFUSIONRATE.to_numpy(dtype=np.float64)
        bolus = e.BOLUS.to_numpy(dtype=np.float64)
        mask = e.ISMEASUREMENT.to_numpy(dtype=np.int32)
        concentration = o.CP.to_numpy(dtype=np.float64)
        if len(times) == 0 or np.any(np.diff(times) < 0):
            raise ValueError(f"Unsorted or missing event times for ID {patient_id}")
        if len(concentration) != int(mask.sum()):
            raise ValueError(f"Measurement count mismatch for ID {patient_id}: {len(concentration)} vs {mask.sum()}")
        if np.any(~np.isfinite(times)) or np.any(rate < 0) or np.any(bolus < 0):
            raise ValueError(f"Invalid dosing for ID {patient_id}")
        patient_cov = obs.loc[obs.ID == patient_id, covariates].iloc[0].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(patient_cov)):
            raise ValueError(f"Missing covariate for ID {patient_id}")
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
        if height <= 0 or weight <= 0: 
            raise ValueError(f"Invalid height/weight for ID {ids[i]}") 
        cov[i] = [age, weight, weight / (height / 100.) ** 2, sex - 1.5, av - 1.5]
    arrays["covariates"] = cov
    arrays["patient_ids"] = np.asarray(ids)
    return arrays


if __name__ == "__main__":
    data = load_data()
    print("Patients:", len(data["patient_ids"]))
    print("Event array shape:", data["time"].shape)
    print("Measured concentrations:", int(data["observed"].sum()))
