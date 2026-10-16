"""One-file BET demonstration: do patient covariates improve held-out propofol PK predictions?

Run: python Examples/PropofolCovariates/example.py
Requires: pip install -e . pandas
Downloads upstream CSVs once into ./data, then fits two otherwise identical models.
The three-compartment dynamics are solved exactly by a matrix exponential.
This is an exploratory linear covariate model, not a reproduction of published symbolic equations.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import os
import tempfile

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import numpyro.distributions as dist
import BayesianEstimationTool as bet

DATA_DIR = Path(__file__).resolve().parent / "data"
UPSTREAM_COMMIT = "939d30699fa6157c14c601712882c51106b82a24"
BASE_URL = f"https://raw.githubusercontent.com/wahlquisty/learning-pharmacometric-covariate-structures/{UPSTREAM_COMMIT}/symbreg/data"
FILES = ("dataframe.csv", "eleveld_infusiondata.csv", "eleveld_modelparams.csv")
EXCLUDED_IDS = {893, 897}

def download_data(directory=DATA_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        if (directory / filename).is_file():
            continue
        destination = directory / filename
        temp_path = None
        try:
            # Fetch the immutable upstream revision even for cached files: this
            # detects stale or modified local CSVs before inference.
            remote_hash = hashlib.sha256()
            with urlopen(f"{BASE_URL}/{filename}", timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=filename + ".", suffix=".part", delete=False) as output:
                    temp_path = Path(output.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        remote_hash.update(chunk)
                        output.write(chunk)
            if temp_path.stat().st_size == 0:
                raise ValueError(f"Empty upstream file: {filename}")
            if destination.exists():
                local_hash = hashlib.sha256()
                with destination.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        local_hash.update(chunk)
                if local_hash.digest() != remote_hash.digest():
                    raise ValueError(f"Cached upstream file differs from pinned commit {UPSTREAM_COMMIT}: {destination}")
                print(f"Verified cached file: {destination} (SHA-256 {remote_hash.hexdigest()})")
            else:
                os.replace(temp_path, destination)
                print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes; SHA-256 {remote_hash.hexdigest()})")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

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

def load_data(directory=None, selected_ids=None, tables=None):
    if tables is None:
        download_data(DATA_DIR if directory is None else directory)
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
        if len(concentration) != n_measurements:
            raise ValueError(f"Measurement count mismatch for ID {patient_id}: {len(concentration)} vs {n_measurements}")
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

def concentrations(parameters, time, rate, bolus):
    """parameters = [V1, V2, V3, CL, Q2, Q3] in L and L/min."""
    v1, v2, v3, cl, q2, q3 = parameters
    # Time is in seconds. The upstream data README documents infusion rates
    # as ug/s; preserve them unchanged. The Julia reader has a conflicting ug/min
    # comment, but ug/s is consistent with the simulator's time and volume units.
    cl, q2, q3 = cl / 60, q2 / 60, q3 / 60
    a = jnp.array([
        [-(cl + q2 + q3) / v1, q2 / v2, q3 / v3],
        [q2 / v1, -q2 / v2, 0.0],
        [q3 / v1, 0.0, -q3 / v3],
    ])
    # Augmented linear system integrates constant infusion exactly, including
    # the singular zero eigenvalue of the infusion source.
    aug = jnp.zeros((4, 4)).at[:3, :3].set(a).at[0, 3].set(1.0)
    dt = jnp.diff(time, append=time[-1])

    def step(amount, event):
        duration, infusion, dose = event
        amount = amount.at[0].add(dose)
        concentration = amount[0] / (v1 * 1000.0)
        def propagate(_):
            transition = expm(aug * duration)
            return (transition @ jnp.concatenate((amount, jnp.array([infusion]))))[:3]

        # The data loader validates nondecreasing times before tracing. The
        # positive-duration branch skips padded and repeated-measurement rows;
        # this traced kernel does not itself raise on negative durations.
        next_amount = jax.lax.cond(duration > 0, propagate, lambda _: amount, operand=None)
        return next_amount, concentration

    _, result = jax.lax.scan(step, jnp.zeros(3), (dt, rate, bolus))
    return result

PARAMS = ("V1", "V2", "V3", "CL", "Q2", "Q3")
REFERENCE = jnp.array([10., 20., 100., 1.5, 2., 1.])


def model(with_covariates):
    def predict(g, z, d):
        log_pk = g["mean"] + g["sd"] * jnp.stack([z[name] for name in PARAMS])
        if with_covariates:
            log_pk = log_pk + g["effects"] @ d["covariates"]
        return {"cp": concentrations(jnp.exp(log_pk), d["time"], d["rate"], d["bolus"])}
    return predict


def priors(with_covariates):
    def make(_):
        result = {"mean": dist.Normal(jnp.log(REFERENCE), 0.8).to_event(1),
                  "sd": dist.HalfNormal(0.5).expand((6,)).to_event(1),
                  "noise": dist.HalfNormal(0.5)}
        if with_covariates:
            result["effects"] = dist.Normal(0, 0.2).expand((6, 5)).to_event(2)
        return result
    return make


def likelihood(g, z, d, out):
    mask = d["observed"].astype(bool)
    observed = jnp.where(mask, jnp.maximum(d["concentration"], 1e-6), 1.)
    return {"cp": (observed, dist.LogNormal(jnp.log(jnp.maximum(out["cp"], 1e-6)), g["noise"]).mask(mask))}


def error(out, d):
    # Score only future concentrations, after the first three measurements.
    mask = d["observed"].astype(bool) & (jnp.cumsum(d["observed"]) > 3)
    residual = jnp.abs(jnp.log(jnp.maximum(out["cp"], 1e-6)) -
                       jnp.log(jnp.maximum(d["concentration"], 1e-6)))
    return jnp.sum(jnp.where(mask, residual, 0.)) / jnp.maximum(mask.sum(), 1)


def main():
    raw = load_data()
    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(raw["patient_ids"]))
    train = permutation[:100]
    eligible = raw["observed"].sum(axis=1) > 3
    test = np.array([i for i in permutation[100:] if eligible[i]])[:25]
    if len(test) < 25:
        raise ValueError("Not enough held-out patients with four measurements")
    selected = np.r_[train, test]
    data = {k: jnp.asarray(v[selected]) for k, v in raw.items() if k != "patient_ids"}
    cov = np.asarray(data["covariates"])
    mean, sd = cov[:100].mean(axis=0), cov[:100].std(axis=0)
    data["covariates"] = jnp.asarray((cov - mean) / np.where(sd > 0, sd, 1))
    for with_covariates in (False, True):
        est = bet.Estimator(data=data, ModelFn=model(with_covariates),
                            localZNames=list(PARAMS), GlobalPriorFn=priors(with_covariates), rngSeed=0)
        est.Train(trainDataIdxs=jnp.arange(100), TrainLikelihoodFn=likelihood,
                  Method=bet.Svi(numSteps=1000, numSamples=50, progressBar=False))
        est.Test(testDataIdxs=jnp.arange(100, 125), nGlobalSamples=1, nLocalZSamples=50)
        score = jnp.mean(est.ScoreTestSamples(ScoreFn=error))
        print(f"{'With' if with_covariates else 'Without'} covariates: future log error = {float(score):.4f}")

if __name__ == '__main__':
    main()
