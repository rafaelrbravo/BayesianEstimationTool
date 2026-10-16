"""Minimal BET wiring for the propofol three-compartment model.

Run from repository root after installing numpyro, jax, scipy and pandas and
placing upstream CSV files in this example's data/ directory.

This is a development-size inference smoke test, NOT the final comparison.
"""
import jax.numpy as jnp
import numpyro.distributions as dist

import BayesianEstimationTool as bet
from load_data import load_data
from inspect_data import read_tables, patient_ids
from pk_model import concentrations
from split import make_split, eligible_test_indices

PARAMETERS = ("V1", "V2", "V3", "CL", "Q2", "Q3")
# Positive PK parameters, in liters and liters/minute respectively.
REFERENCE = jnp.array([10., 20., 100., 1.5, 2., 1.])
N_FIT = 3


def ModelFn(globalParams, localZs, data):
    typical = globalParams["logTypical"]
    spread = globalParams["spread"]
    values = jnp.exp(typical + spread * jnp.stack([localZs[name] for name in PARAMETERS]))
    prediction = concentrations(values, data["time"], data["rate"], data["bolus"])
    return {"concentration": prediction}


def GlobalPriorFn(dataShapes):
    return {
        "logTypical": dist.Normal(jnp.log(REFERENCE), 0.8).to_event(1),
        "spread": dist.HalfNormal(0.5).expand((6,)).to_event(1),
        "sigma": dist.HalfNormal(0.5),
    }


def TrainLikelihoodFn(globalParams, localZs, data, modelOut):
    mask = data["observed"].astype(bool)
    predicted = jnp.maximum(modelOut["concentration"], 1e-6)
    observed = jnp.where(mask, jnp.maximum(data["concentration"], 1e-6), 1.)
    return {"concentration": (observed, dist.LogNormal(jnp.log(predicted), globalParams["sigma"]).mask(mask))}


def TestLikelihoodFn(globalParams, localZs, data, modelOut):
    # Fit only the first N_FIT measured concentrations, regardless of
    # the number of infusion/bolus event rows preceding them.
    fit = jnp.cumsum(data["observed"]) <= N_FIT
    mask = data["observed"].astype(bool) & fit
    predicted = jnp.maximum(modelOut["concentration"], 1e-6)
    observed = jnp.where(mask, jnp.maximum(data["concentration"], 1e-6), 1.)
    return {"concentration": (observed, dist.LogNormal(jnp.log(predicted), globalParams["sigma"]).mask(mask))}


def ScoreFn(modelOut, data):
    # Evaluate only concentrations AFTER the first N_FIT measured values.
    # This ensures conditional inference is never scored on data it used.
    measured_index = jnp.cumsum(data["observed"])
    mask = data["observed"].astype(bool) & (measured_index > N_FIT)
    error = jnp.abs(jnp.log(jnp.maximum(modelOut["concentration"], 1e-6)) -
                    jnp.log(jnp.maximum(data["concentration"], 1e-6)))
    return jnp.sum(jnp.where(mask, error, 0.)) / jnp.maximum(mask.sum(), 1)


def main():
    tables = read_tables()
    ids = patient_ids(tables)
    train, test = make_split(len(ids))
    train = train[:100]
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    eligible_test = eligible_test_indices(test, ids, counts, n_fit=N_FIT, limit=25)
    print(f"Evaluable held-out patients selected: {len(eligible_test)}/{len(test)} (need >{N_FIT} concentrations)")
    if not eligible_test:
        raise ValueError("No held-out patients have enough measurements for future scoring")
    selected_ids = [ids[i] for i in list(train) + eligible_test]
    arrays = load_data(selected_ids=selected_ids, tables=tables)
    positions = {patient_id: i for i, patient_id in enumerate(arrays["patient_ids"])}
    selected = [positions[patient_id] for patient_id in selected_ids]
    # Keep only development patients in the JAX arrays; remap split indices.
    # This avoids allocating inference tensors for all 1,031 patients.
    n_train = 100
    train = jnp.arange(n_train)
    test = jnp.arange(n_train, len(selected))
    data = {key: jnp.asarray(value[selected]) for key, value in arrays.items() if key != "patient_ids"}
    # A future-only score is undefined for patients with <= N_FIT measurements.
    n_observed = jnp.sum(data["observed"][test], axis=1)
    if bool(jnp.any(n_observed <= N_FIT)):
        raise ValueError("Event counts and loaded concentrations disagree for held-out patients")
    est = bet.Estimator(data=data, ModelFn=ModelFn, localZNames=list(PARAMETERS),
                        GlobalPriorFn=GlobalPriorFn, rngSeed=0)
    est.Train(trainDataIdxs=train, TrainLikelihoodFn=TrainLikelihoodFn,
              Method=bet.Svi(numSteps=1000, numSamples=50, progressBar=False), printSummary=True)
    test_indices = test
    est.Test(testDataIdxs=test_indices, nLocalZSamples=50)
    print("Unconditional future log error:", jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn)))
    # Updating held-out patient effects from the first N_FIT observations
    # is deliberately a separate, more expensive inference run.
    est.Test(testDataIdxs=test_indices, TestLikelihoodFn=TestLikelihoodFn,
             Method=bet.Svi(numSteps=300, numSamples=50, progressBar=False))
    print("Conditional future log error:", jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn)))


if __name__ == "__main__":
    main()
