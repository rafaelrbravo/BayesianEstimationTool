"""One-file BET demonstration: do patient covariates improve held-out propofol PK predictions?

Run: python Examples/PropofolCovariates/example.py
Requires: pip install -e . pandas
Downloads upstream CSVs once into ./data, then fits two otherwise identical models.
The three-compartment dynamics are solved exactly by a matrix exponential.
This is an exploratory linear covariate model, not a reproduction of published symbolic equations.
"""
from pathlib import Path
from urllib.request import urlopen
import shutil

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
FILES = {"observations": "dataframe.csv", "events": "eleveld_infusiondata.csv", "parameters": "eleveld_modelparams.csv"}
EXCLUDED_IDS = {893, 897}

def load_data(directory=DATA_DIR):
    """Read the pinned source tables and align concentration observations to dosing events."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        path = directory / filename
        if not path.exists():
            with urlopen(f"{BASE_URL}/{filename}", timeout=60) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)

    obs = pd.read_csv(directory / FILES["observations"])
    events = pd.read_csv(directory / FILES["events"])
    params = pd.read_csv(directory / FILES["parameters"])
    for table in (obs, events, params):
        table.columns = table.columns.str.strip().str.upper()
    if "HT" not in obs:
        obs = obs.rename(columns={"HGT": "HT"})

    obs = obs[obs.CP > 0]
    ids = sorted(set(obs.ID) & set(events.ID) & set(params.ID) - {893, 897})
    assert len(ids) == 1031
    records = []
    for patient_id in ids:
        o = obs[obs.ID == patient_id]
        e = events[events.ID == patient_id]
        counts = e.ISMEASUREMENT.to_numpy(dtype=int)
        assert counts.sum() == len(o)
        last = np.flatnonzero(counts)[-1] + 1
        e, counts = e.iloc[:last], counts[:last]
        repeats = np.maximum(counts, 1)
        first = np.repeat(np.cumsum(repeats) - repeats, repeats)
        time = np.repeat(e.TIME.to_numpy(), repeats)
        rate = np.repeat(e.INFUSIONRATE.to_numpy(), repeats)
        bolus = np.repeat(e.BOLUS.to_numpy(), repeats)
        bolus[np.arange(len(bolus)) != first] = 0
        mask = np.repeat(counts > 0, repeats)
        assert mask.sum() == len(o) and np.all(np.diff(time) >= 0)
        assert np.all(np.abs(time[mask] - o.TIME.to_numpy() * 60) <= 30.5)
        y = np.zeros(len(time))
        y[mask] = o.CP.to_numpy()
        age, weight, height, sex, route = o.iloc[0][["AGE", "WGT", "HT", "M1F2", "A1V2"]]
        cov = [age, weight, weight / (height / 100) ** 2, sex - 1.5, route - 1.5]
        records.append((time, rate, bolus, mask, y, cov))

    width = max(len(record[0]) for record in records)
    data = {name: np.zeros((len(ids), width), dtype=np.float32)
            for name in ("time", "rate", "bolus", "observed", "concentration")}
    for i, record in enumerate(records):
        n = len(record[0])
        for name, values in zip(data, record[:5]):
            data[name][i, :n] = values
        data["time"][i, n:] = record[0][-1]
    data["covariates"] = np.asarray([r[5] for r in records], dtype=np.float32)
    data["patient_ids"] = np.asarray(ids)
    return data


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
