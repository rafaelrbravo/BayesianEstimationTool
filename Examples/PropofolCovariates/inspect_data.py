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
    measured = observations.loc[observations["CP"] > 0].groupby("ID").size()
    counts = events.groupby("ID")["ISMEASUREMENT"].sum()
    bad = []
    for patient_id in ids:
        expected = int(measured.get(patient_id, 0))
        actual = counts.get(patient_id, 0)
        if not pd.notna(actual) or actual < 0 or actual != int(actual) or int(actual) != expected:
            bad.append((patient_id, expected, actual))
    if bad:
        raise ValueError(f"Measurement counts disagree for {len(bad)} patients; first five: {bad[:5]}")
    return int(measured.reindex(ids, fill_value=0).sum())


def main():
    tables = read_tables()
    ids = patient_ids(tables)
    n_measurements = validate_measurement_counts(tables, ids)
    print(f"Validated {n_measurements} measured concentrations against event counts")
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows, {table.ID.nunique()} patients")
        print("  columns:", ", ".join(table.columns))
    print(f"Patients with concentration, dosing events, and PK metadata: {len(ids)}")
    print("First and last patient IDs:", ids[0], ids[-1])
    print("Note: indices from split.py index this sorted patient-ID list, not raw ID values.")


if __name__ == "__main__":
    main()
