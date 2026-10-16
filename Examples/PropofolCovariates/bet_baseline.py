"""Minimal BET wiring for the propofol three-compartment model.

Run from repository root after installing numpyro, jax, scipy and pandas and
placing upstream CSV files in this example's data/ directory.

This is a development-size inference smoke test, NOT the final comparison.
"""
import jax.numpy as jnp
import numpyro.distributions as dist

import BayesianEstimationTool as bet
from load_data import load_data
from pk_model import concentrations
from split import make_split

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
    # Scores are averaged over actual measurements, not padded events.
    mask = data["observed"].astype(bool)
    error = jnp.abs(jnp.log(jnp.maximum(modelOut["concentration"], 1e-6)) -
                    jnp.log(jnp.maximum(data["concentration"], 1e-6)))
    return jnp.sum(jnp.where(mask, error, 0.)) / jnp.maximum(mask.sum(), 1)


def main():
    arrays = load_data()
    train, test = make_split(len(arrays["patient_ids"]))
    data = {key: jnp.asarray(value) for key, value in arrays.items() if key != "patient_ids"}
    est = bet.Estimator(data=data, ModelFn=ModelFn, localZNames=list(PARAMETERS),
                        GlobalPriorFn=GlobalPriorFn, rngSeed=0)
    est.Train(trainDataIdxs=jnp.asarray(train[:100]), TrainLikelihoodFn=TrainLikelihoodFn,
              Method=bet.Svi(numSteps=1000), printSummary=True)
    est.Test(testDataIdxs=jnp.asarray(test[:25]), nLocalZSamples=50)
    print("Mean held-out absolute log error:", jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn)))


if __name__ == "__main__":
    main()
