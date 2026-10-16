"""Compare published fixed PK equations to a train-fitted no-covariate BET model.

CAUTION: The published symbolic model was trained on overlapping Eleveld
patients. Its scores are descriptive, NOT independent held-out validation.
"""
import argparse
import jax.numpy as jnp

import BayesianEstimationTool as bet
from bet_baseline import PARAMETERS, N_FIT, ScoreFn, TrainLikelihoodFn, TestLikelihoodFn
from compare_covariates import COVARIATE_NAMES, make_model, make_prior, validate_scores, write_patient_scores
from published_bet_model import build_published_model, GlobalPriorFn as PublishedPrior
from load_data import load_data
from inspect_data import read_tables, patient_ids
from paired_stats import paired_bootstrap_interval
from split import make_split, eligible_test_indices

def main():
    parser = argparse.ArgumentParser(description="Compare fixed published propofol PK equations against a BET baseline (overlapping original training cohort)")
    parser.add_argument("--equations", help="Path to Julia-exported symbolic_equations.toml")
    parser.add_argument("--full", action="store_true", help="Use all 831 training patients and all eligible patients in the fixed 200-patient test partition")
    parser.add_argument("--train-patients", type=int, default=100, help="Development training cohort size")
    parser.add_argument("--test-patients", type=int, default=25, help="Development held-out cohort size")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--test-steps", type=int, default=300)
    parser.add_argument("--output-csv", help="Optional per-patient held-out errors for auditing")
    args = parser.parse_args()
    if args.train_steps < 1 or args.test_steps < 1:
        parser.error("SVI steps must be positive")
    if args.train_patients < 1 or args.test_patients < 1:
        parser.error("Development cohort sizes must be positive")
    print("WARNING: Published coefficients were trained on overlapping Eleveld patients; this is NOT an independent held-out generalization test.", flush=True)
    tables = read_tables()
    ids = patient_ids(tables)
    train, test = make_split(len(ids))
    # Use the identical development patients and future-only evaluation.
    if not args.full:
        train = train[:args.train_patients]
    # Keep the prespecified test partition, excluding only patients without
    # a future concentration after the three conditioning measurements.
    counts = tables["events"].groupby("ID")["ISMEASUREMENT"].sum()
    eligible_test = eligible_test_indices(test, ids, counts, n_fit=N_FIT, limit=None if args.full else args.test_patients)
    print(f"Evaluable held-out patients: {len(eligible_test)}/{len(test)} (need >{N_FIT} concentrations)")
    if not eligible_test:
        raise ValueError("No held-out patients have at least four measured concentrations")
    test = eligible_test
    selected_ids = [ids[i] for i in list(train) + list(test)]
    arrays = load_data(selected_ids=selected_ids, tables=tables)
    positions = {patient_id: i for i, patient_id in enumerate(arrays["patient_ids"])}
    selected = [positions[patient_id] for patient_id in selected_ids]
    n_observed = arrays["observed"][selected[len(train):]].sum(axis=1)
    if (n_observed <= N_FIT).any():
        raise ValueError("Event counts and loaded concentrations disagree for held-out patients")
    # Validate covariate layout while retaining original raw values.
    if arrays["covariates"].shape[1] != len(COVARIATE_NAMES):
        raise ValueError(f"Expected {len(COVARIATE_NAMES)} covariates ({COVARIATE_NAMES})")
    # Published equations require raw covariates; do not standardize.
    # Reorder selected patients as training followed by held-out testing.
    n_train = len(train)
    train = jnp.arange(n_train)
    test = jnp.arange(n_train, len(selected))
    data = {key: jnp.asarray(value[selected]) for key, value in arrays.items() if key != "patient_ids"}
    scores = {}
    for use_covariates in (False, True):
        label = "covariates" if use_covariates else "no covariates"
        print(f"Training {label} model on {n_train} patients", flush=True)
        model = build_published_model(args.equations) if use_covariates else make_model(False)
        prior = PublishedPrior if use_covariates else make_prior(False)
        est = bet.Estimator(data=data, ModelFn=model, localZNames=list(PARAMETERS),
                            GlobalPriorFn=prior, rngSeed=0)
        est.Train(trainDataIdxs=jnp.asarray(train), TrainLikelihoodFn=TrainLikelihoodFn,
                  Method=bet.Svi(numSteps=args.train_steps, numSamples=50, progressBar=False))
        print(f"Scoring {label} model without conditioning", flush=True)
        est.Test(testDataIdxs=jnp.asarray(test), nLocalZSamples=50)
        unconditioned = jnp.mean(est.ScoreTestSamples(ScoreFn=ScoreFn), axis=(0, 1))
        print(f"Conditioning {label} model on first {N_FIT} concentrations", flush=True)
        est.Test(testDataIdxs=jnp.asarray(test), TestLikelihoodFn=TestLikelihoodFn,
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
