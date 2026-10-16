"""Exploratory BET covariate comparison (NOT yet published functional forms).

Runs the same held-out split and future-only scoring for both models.
Covariates are standardized using training patients only. This is an
initial linear log-parameter covariate baseline; the published functional
forms must be verified before calling it a reproduction of the paper.
"""
import argparse
import csv

import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist

import BayesianEstimationTool as bet
from bet_baseline import PARAMETERS, REFERENCE, N_FIT, ScoreFn, TrainLikelihoodFn, TestLikelihoodFn
from load_data import load_data
from inspect_data import read_tables, patient_ids, validate_measurement_counts, validate_measurement_times
from pk_model import concentrations
from paired_stats import paired_bootstrap_interval
from split import make_split, eligible_test_indices

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



def standardize_covariates(covariates, train_positions):
    """Center and scale every patient using training patients only."""
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim != 2 or not np.isfinite(covariates).all():
        raise ValueError("Covariates must be a finite two-dimensional array")
    positions = np.asarray(train_positions)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("At least one training patient is required for covariate standardization")
    if not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0) or np.any(positions >= len(covariates)):
        raise ValueError("Training positions must be valid nonnegative integer indices")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Training positions must be unique")
    training = covariates[positions]
    if training.shape[0] == 0:
        raise ValueError("At least one training patient is required for covariate standardization")
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    return (covariates - mean) / scale


def validate_scores(patient_ids, scores):
    """Reject invalid IDs or prediction errors before reporting paired effects."""
    if not patient_ids or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Held-out patient IDs must be nonempty and unique")
    for model in ("no covariates", "covariates"):
        if model not in scores or len(scores[model]) != 2:
            raise ValueError(f"Missing unconditioned/conditioned scores for {model}")
        for setting, values in enumerate(scores[model]):
            values = np.asarray(values)
            if values.shape != (len(patient_ids),) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid patient scores for {model}, setting {setting}")


def write_patient_scores(path, patient_ids, scores):
    """Export the exact patient-aligned values used in paired comparisons."""
    validate_scores(patient_ids, scores)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["patient_id", "baseline_unconditioned", "covariates_unconditioned",
                         "baseline_conditioned", "covariates_conditioned"])
        for i, patient_id in enumerate(patient_ids):
            writer.writerow([int(patient_id), *[float(scores[model][setting][i])
                              for setting in (0, 1) for model in ("no covariates", "covariates")]])


def main():
    parser = argparse.ArgumentParser(description="Compare BET propofol PK models on held-out patients")
    parser.add_argument("--full", action="store_true", help="Use all 831 training patients and all eligible patients in the fixed 200-patient test partition")
    parser.add_argument("--train-patients", type=int, default=100, help="Development training cohort size")
    parser.add_argument("--test-patients", type=int, default=25, help="Development held-out cohort size")
    parser.add_argument("--global-samples", type=int, default=1, help="Population posterior draws for held-out scoring (conditional SVI repeats per draw)")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--test-steps", type=int, default=300)
    parser.add_argument("--output-csv", help="Optional per-patient held-out errors for auditing")
    args = parser.parse_args()
    if args.global_samples < 1:
        parser.error("--global-samples must be positive")
    if args.train_steps < 1 or args.test_steps < 1:
        parser.error("SVI steps must be positive")
    if args.train_patients < 1 or args.test_patients < 1:
        parser.error("Development cohort sizes must be positive")
    if args.full and (args.train_patients != 100 or args.test_patients != 25):
        print("Note: --full overrides --train-patients and --test-patients.", flush=True)
    tables = read_tables()
    ids = patient_ids(tables)
    validate_measurement_counts(tables, ids)
    validate_measurement_times(tables, ids)
    train, test = make_split(len(ids))
    # Use the identical development patients and future-only evaluation.
    if not args.full:
        if args.train_patients > len(train):
            parser.error(f"Requested {args.train_patients} training patients, but only {len(train)} are available")
        train = train[:args.train_patients]
    # Keep the prespecified test partition, excluding only patients without
    # a future concentration after the three conditioning measurements.
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    eligible_test = eligible_test_indices(test, ids, counts, n_fit=N_FIT, limit=None if args.full else args.test_patients)
    print(f"Evaluable held-out patients: {len(eligible_test)}/{len(test)} (need >{N_FIT} concentrations)")
    if not eligible_test:
        raise ValueError("No held-out patients have at least four measured concentrations")
    if not args.full and len(eligible_test) != args.test_patients:
        raise ValueError(f"Requested {args.test_patients} evaluable held-out patients, found {len(eligible_test)}")
    test = eligible_test
    selected_ids = [ids[i] for i in list(train) + list(test)]
    arrays = load_data(selected_ids=selected_ids, tables=tables)
    if len(set(arrays["patient_ids"])) != len(arrays["patient_ids"]):
        raise ValueError("Loaded patient IDs must be unique")
    if set(arrays["patient_ids"]) != set(selected_ids):
        raise ValueError("Loaded patient IDs do not match the requested training and test cohort")
    positions = {patient_id: i for i, patient_id in enumerate(arrays["patient_ids"])}
    selected = [positions[patient_id] for patient_id in selected_ids]
    if len(set(selected)) != len(selected):
        raise ValueError("Training and held-out patients must be disjoint and unique")
    n_observed = arrays["observed"][selected[len(train):]].sum(axis=1)
    if (n_observed <= N_FIT).any():
        raise ValueError("Event counts and loaded concentrations disagree for held-out patients")
    # Fit centering/scaling on TRAINING patients, never held-out patients.
    if arrays["covariates"].shape[1] != len(COVARIATE_NAMES):
        raise ValueError(f"Expected {len(COVARIATE_NAMES)} covariates ({COVARIATE_NAMES})")
    arrays["covariates"] = standardize_covariates(arrays["covariates"], selected[:len(train)])
    # Reorder selected patients as training followed by held-out testing.
    n_train = len(train)
    train = jnp.arange(n_train)
    test = jnp.arange(n_train, len(selected))
    data = {key: jnp.asarray(value[selected]) for key, value in arrays.items() if key != "patient_ids"}
    scores = {}
    for use_covariates in (False, True):
        label = "covariates" if use_covariates else "no covariates"
        print(f"Training {label} model on {n_train} patients", flush=True)
        est = bet.Estimator(data=data, ModelFn=make_model(use_covariates), localZNames=list(PARAMETERS),
                            GlobalPriorFn=make_prior(use_covariates), rngSeed=0)
        est.Train(trainDataIdxs=jnp.asarray(train), TrainLikelihoodFn=TrainLikelihoodFn,
                  Method=bet.Svi(numSteps=args.train_steps, numSamples=50, progressBar=False))
        print(f"Scoring {label} model without conditioning", flush=True)
        est.Test(testDataIdxs=jnp.asarray(test), nGlobalSamples=args.global_samples, nLocalZSamples=50)
        unconditioned = jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn), axis=(0, 1))
        print(f"Conditioning {label} model on first {N_FIT} concentrations", flush=True)
        est.Test(testDataIdxs=jnp.asarray(test), nGlobalSamples=args.global_samples, TestLikelihoodFn=TestLikelihoodFn,
                 Method=bet.Svi(numSteps=args.test_steps, numSamples=50, progressBar=False))
        conditioned = jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn), axis=(0, 1))
        scores[label] = (unconditioned, conditioned)
    validate_scores(selected_ids[n_train:], scores)
    for label in ("no covariates", "covariates"):
        unconditioned, conditioned = scores[label]
        print(f"{label}: unconditioned future log error={float(jnp.mean(unconditioned)):.4f}; conditioned={float(jnp.mean(conditioned)):.4f}")
    if args.output_csv:
        write_patient_scores(args.output_csv, selected_ids[n_train:], scores)
        print(f"Wrote held-out patient errors: {args.output_csv}")
    for i, setting in enumerate(("unconditioned", "conditioned")):
        delta = scores["covariates"][i] - scores["no covariates"][i]
        lo, hi = paired_bootstrap_interval(delta)
        print(f"Paired {setting} covariate minus baseline log error: mean={float(jnp.mean(delta)):.4f}, median={float(jnp.median(delta)):.4f}, improved patients={int(jnp.sum(delta < 0))}/{len(test)}, patient-bootstrap 95% CI for mean=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
