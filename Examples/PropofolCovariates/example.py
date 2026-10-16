"""Minimal BET example: identical propofol PK models, with and without covariates.

Run after downloading the upstream data (see README). Uses the existing
load_data.py and pk_model.py for dataset parsing and three-compartment physics.
Covariate effects here are simple linear log-parameter effects, NOT the
published symbolic-regression equations.
"""
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import BayesianEstimationTool as bet
from load_data import load_data
from pk_model import concentrations

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


if __name__ == "__main__":
    main()
