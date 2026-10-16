"""Exploratory BET covariate comparison (NOT yet published functional forms).

Runs the same held-out split and future-only scoring for both models.
Covariates are standardized using training patients only. This is an
initial linear log-parameter covariate baseline; the published functional
forms must be verified before calling it a reproduction of the paper.
"""
import argparse

import jax.numpy as jnp
import numpyro.distributions as dist

import BayesianEstimationTool as bet
from bet_baseline import PARAMETERS, REFERENCE, ScoreFn, TrainLikelihoodFn, TestLikelihoodFn
from load_data import load_data
from inspect_data import read_tables, patient_ids
from pk_model import concentrations
from paired_stats import paired_bootstrap_interval
from split import make_split

COVARIATE_NAMES = ("AGE", "WGT", "BMI", "GDR", "AV")


def make_model(use_covariates):
    def ModelFn(globalParams, localZs, data):
        log_params = globalParams["logTypical"] + globalParams["spread"] * jnp.stack([localZs[name] for name in PARAMETERS])
        if use_covariates:
            log_params = log_params + globalParams["covariateEffects"] @ data["covariates"]
        return {"concentration": concentrations(jnp.exp(log_params), data["time"], data["rate"], data["bolus"])}
    return ModelFn


def make_prior(use_covariates):
    def GlobalPriorFn(dataShapes):
        priors = {
            "logTypical": dist.Normal(jnp.log(REFERENCE), 0.8).to_event(1),
            "spread": dist.HalfNormal(0.5).expand((6,)).to_event(1),
            "sigma": dist.HalfNormal(0.5),
        }
        if use_covariates:
            priors["covariateEffects"] = dist.Normal(0., 0.2).expand((6, len(COVARIATE_NAMES))).to_event(2)
        return priors
    return GlobalPriorFn


def main():
    parser = argparse.ArgumentParser(description="Compare BET propofol PK models on held-out patients")
    parser.add_argument("--full", action="store_true", help="Use all 831 training and 200 held-out patients")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--test-steps", type=int, default=300)
    args = parser.parse_args()
    if args.train_steps < 1 or args.test_steps < 1:
        parser.error("SVI steps must be positive")
    tables = read_tables()
    ids = patient_ids(tables)
    train, test = make_split(len(ids))
    # Use the identical development patients and future-only evaluation.
    if not args.full:
        train, test = train[:100], test[:25]
    # Keep the prespecified test partition, excluding only patients without
    # a future concentration after the three conditioning measurements.
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    eligible_test = [i for i in test if counts.get(ids[i], 0) > 3]
    print(f"Evaluable held-out patients: {len(eligible_test)}/{len(test)} (need >=4 concentrations)")
    if not eligible_test:
        raise ValueError("No held-out patients have at least four measured concentrations")
    test = eligible_test
    selected_ids = [ids[i] for i in list(train) + list(test)]
    arrays = load_data(selected_ids=selected_ids, tables=tables)
    positions = {patient_id: i for i, patient_id in enumerate(arrays["patient_ids"])}
    selected = [positions[patient_id] for patient_id in selected_ids]
    n_observed = arrays["observed"][selected[len(train):]].sum(axis=1)
    if (n_observed <= 3).any():
        raise ValueError("Event counts and loaded concentrations disagree for held-out patients")
    # Fit centering/scaling on TRAINING patients, never held-out patients.
    cov = arrays["covariates"]
    train_positions = selected[:len(train)]
    mean = cov[train_positions].mean(axis=0)
    scale = cov[train_positions].std(axis=0)
    scale = jnp.where(scale > 0, scale, 1.)
    arrays["covariates"] = (cov - mean) / scale
    # Reorder selected patients as training followed by held-out testing.
    n_train = len(train)
    train = jnp.arange(n_train)
    test = jnp.arange(n_train, len(selected))
    data = {key: jnp.asarray(value[selected]) for key, value in arrays.items() if key != "patient_ids"}
    scores = {}
    for use_covariates in (False, True):
        label = "covariates" if use_covariates else "no covariates"
        est = bet.Estimator(data=data, ModelFn=make_model(use_covariates), localZNames=list(PARAMETERS),
                            GlobalPriorFn=make_prior(use_covariates), rngSeed=0)
        est.Train(trainDataIdxs=jnp.asarray(train), TrainLikelihoodFn=TrainLikelihoodFn,
                  Method=bet.Svi(numSteps=args.train_steps, numSamples=50, progressBar=False))
        est.Test(testDataIdxs=jnp.asarray(test), nLocalZSamples=50)
        unconditioned = jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn), axis=(0, 1))
        est.Test(testDataIdxs=jnp.asarray(test), TestLikelihoodFn=TestLikelihoodFn,
                 Method=bet.Svi(numSteps=args.test_steps, numSamples=50, progressBar=False))
        conditioned = jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn), axis=(0, 1))
        scores[label] = (unconditioned, conditioned)
        print(f"{label}: unconditioned future log error={float(jnp.mean(unconditioned)):.4f}; conditioned={float(jnp.mean(conditioned)):.4f}")
    for i, setting in enumerate(("unconditioned", "conditioned")):
        delta = scores["covariates"][i] - scores["no covariates"][i]
        lo, hi = paired_bootstrap_interval(delta)
        print(f"Paired {setting} covariate minus baseline log error: mean={float(jnp.mean(delta)):.4f}, median={float(jnp.median(delta)):.4f}, improved patients={int(jnp.sum(delta < 0))}/{len(test)}, patient-bootstrap 95% CI for mean=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
