"""Inspect the published propofol dataset before fitting BET models.

Download the three CSV files from the upstream repository's symbreg/data/
directory into a local data/ directory next to this script. The original
files remain upstream and are not duplicated in BET.

Run:
    python Examples/PropofolCovariates/inspect_data.py

This script deliberately performs no fitting or preprocessing on the held-out
patients. It checks the patient universe and event/measurement consistency.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = {
    "observations": "dataframe.csv",
    "events": "eleveld_infusiondata.csv",
    "parameters": "eleveld_modelparams.csv",
}
EXCLUDED_IDS = {893, 897}  # No measurements, per upstream get_elevelddata.jl


def read_tables(directory=DATA_DIR):
    tables = {}
    for name, filename in FILES.items():
        path = Path(directory) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; obtain it from the upstream symbreg/data directory")
        table = pd.read_csv(path)
        table.columns = table.columns.str.strip().str.upper()
        if name == "observations":
            # The upstream README calls height HGT, while its Julia loader
            # indexes HT. Accept either spelling and normalize internally.
            if "HT" in table and "HGT" in table:
                try:
                    same_height = np.allclose(
                        table["HT"].to_numpy(dtype=float),
                        table["HGT"].to_numpy(dtype=float),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                except (TypeError, ValueError):
                    same_height = False
                if not same_height:
                    raise ValueError("dataframe.csv has conflicting HT and HGT height columns")
                table = table.drop(columns="HGT")
            elif "HGT" in table:
                table = table.rename(columns={"HGT": "HT"})
        if "ID" not in table:
            raise ValueError(f"{filename} is missing ID")
        tables[name] = table
    return tables


def patient_ids(tables):
    observations = tables["observations"]
    events = tables["events"]
    if "CP" not in observations:
        raise ValueError("dataframe.csv is missing CP concentration measurements")
    if "ISMEASUREMENT" not in events:
        raise ValueError("eleveld_infusiondata.csv is missing IsMeasurement")
    measured = set(observations.loc[observations["CP"] > 0, "ID"].astype(int))
    with_events = set(events.loc[events["ISMEASUREMENT"] > 0, "ID"].astype(int))
    with_parameters = set(tables["parameters"]["ID"].astype(int))
    ids = sorted((measured & with_events & with_parameters) - EXCLUDED_IDS)
    if len(ids) != 1031:
        raise ValueError(f"Expected 1031 usable patients; found {len(ids)}. Inspect exclusions and CSV schema.")
    return ids



def validate_measurement_counts(tables, ids):
    """Check every included patient's event count against positive CP rows."""
    observations = tables["observations"]
    events = tables["events"]
    # Match the upstream get_ymeas convention: only CP > 0 is observed.
    # Nonpositive or missing rows are not concentration measurements.
    selected_observations = observations.loc[observations["ID"].isin(ids)]
    measured_rows = selected_observations.loc[selected_observations["CP"] > 0]
    if not np.isfinite(measured_rows["CP"].to_numpy(dtype=float)).all():
        raise ValueError("Invalid positive observed concentration values")
    measured = measured_rows.groupby("ID").size()
    # Only validate the selected usable cohort; upstream excludes IDs 893/897.
    selected_events = events.loc[events["ID"].isin(ids)]
    raw_counts = selected_events["ISMEASUREMENT"].to_numpy(dtype=float)
    if np.any(~np.isfinite(raw_counts)) or np.any(raw_counts < 0) or np.any(raw_counts != np.floor(raw_counts)):
        raise ValueError("Invalid individual event measurement counts")
    counts = selected_events.groupby("ID")["ISMEASUREMENT"].sum()
    bad = []
    for patient_id in ids:
        expected = int(measured.get(patient_id, 0))
        actual = counts.get(patient_id, 0)
        if not np.isfinite(actual) or actual < 0 or actual != np.floor(actual) or actual != expected:
            bad.append((patient_id, expected, actual))
    if bad:
        raise ValueError(f"Measurement counts disagree for {len(bad)} patients; first five: {bad[:5]}")
    return int(measured.reindex(ids, fill_value=0).sum())


def validate_measurement_times(tables, ids, tolerance_seconds=30.5):
    """Ensure observation order matches infusion-event measurement order.

    Observation TIME is in minutes, rounded to the nearest minute; infusion-event Time is seconds.
    The default tolerance therefore allows half a minute plus a small floating-point margin.
    """
    observations = tables["observations"]
    events = tables["events"]
    for patient_id in ids:
        obs = observations.loc[(observations.ID == patient_id) & (observations.CP > 0), "TIME"].to_numpy(dtype=float) * 60.
        rows = events.loc[events.ID == patient_id]
        event_times = np.repeat(rows["TIME"].to_numpy(dtype=float), rows["ISMEASUREMENT"].to_numpy(dtype=int))
        if len(obs) != len(event_times) or not np.isfinite(obs).all() or not np.isfinite(event_times).all():
            raise ValueError(f"Invalid measurement times for ID {patient_id}")
        if np.any(np.abs(obs - event_times) > tolerance_seconds):
            raise ValueError(f"Observation/event measurement times disagree for ID {patient_id}")



def main():
    tables = read_tables()
    ids = patient_ids(tables)
    n_measurements = validate_measurement_counts(tables, ids)
    validate_measurement_times(tables, ids)
    print(f"Validated {n_measurements} measured concentrations against event counts and times")
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows, {table.ID.nunique()} patients")
        print("  columns:", ", ".join(table.columns))
    print(f"Patients with concentration, dosing events, and PK metadata: {len(ids)}")
    print("First and last patient IDs:", ids[0], ids[-1])
    print("Note: indices from split.py index this sorted patient-ID list, not raw ID values.")


if __name__ == "__main__":
    main()
