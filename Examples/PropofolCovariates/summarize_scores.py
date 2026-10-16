"""Summarize paired held-out propofol errors from the inference workflow.

Run: python Examples/PropofolCovariates/summarize_scores.py propofol-dev-scores.csv
"""
import argparse
import csv
import numpy as np

from paired_stats import paired_bootstrap_interval

COLUMNS = ("baseline_unconditioned", "covariates_unconditioned",
           "baseline_conditioned", "covariates_conditioned")


def read_scores(path):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["patient_id", *COLUMNS]:
            raise ValueError("Unexpected paired score CSV columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty paired score CSV")
    ids = [int(row["patient_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate held-out patient IDs")
    scores = {name: np.asarray([float(row[name]) for row in rows]) for name in COLUMNS}
    if not all(np.isfinite(values).all() and np.all(values >= 0) for values in scores.values()):
        raise ValueError("Invalid held-out errors")
    return ids, scores


def summarize(ids, scores):
    for setting in ("unconditioned", "conditioned"):
        baseline = scores[f"baseline_{setting}"]
        covariates = scores[f"covariates_{setting}"]
        delta = covariates - baseline
        low, high = paired_bootstrap_interval(delta)
        print(f"{setting}: {len(ids)} patients")
        print(f"  baseline mean={baseline.mean():.4f}, median={np.median(baseline):.4f}")
        print(f"  covariate mean={covariates.mean():.4f}, median={np.median(covariates):.4f}")
        print(f"  covariate minus baseline mean={delta.mean():.4f}, median={np.median(delta):.4f}, 95% patient-bootstrap CI=[{low:.4f}, {high:.4f}]")
        print(f"  improved patients={np.sum(delta < 0)}/{len(ids)}")
        if low > 0:
            print("  Direction: covariates worse (bootstrap interval above zero)")
        elif high < 0:
            print("  Direction: covariates better (bootstrap interval below zero)")
        else:
            print("  Direction: inconclusive (bootstrap interval includes zero)")


def main():
    parser = argparse.ArgumentParser(description="Summarize held-out BET propofol comparison")
    parser.add_argument("scores_csv", help="Paired scores exported by compare_covariates.py")
    args = parser.parse_args()
    ids, scores = read_scores(args.scores_csv)
    summarize(ids, scores)


if __name__ == "__main__":
    main()
