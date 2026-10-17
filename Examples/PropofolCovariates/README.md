# Propofol covariate example

A self-contained demonstration of [BayesianEstimationTool](../../README.md) using a three-compartment propofol pharmacokinetic model and the dataset from [Learning pharmacometric covariate model structures with symbolic regression networks](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures).

From the BET repository root, install BET and pandas, then run:

```bash
python -m pip install -e . pandas
python Examples/PropofolCovariates/example.py
```

`example.py` downloads three source CSVs pinned to an upstream commit into a local `data/` directory on first use. It checks and aligns 1,031 patients' concentration and dosing records, solves the three-compartment dynamics analytically using a matrix exponential, and compares hierarchical BET models with and without covariates (age, weight, BMI, sex, and sampling route). Both models use the same 100 training patients and 25 eligible held-out patients. Covariates are standardized using training patients only. The printed score is mean absolute log error on held-out measurements after the first three observations.

This is an exploratory linear covariate model, **not** a reproduction of the original paper's learned symbolic equations. The standalone loader and upstream data checks passed in [GitHub Actions](https://github.com/rafaelrbravo/BayesianEstimationTool/actions/runs/38990330819). This validation **does not** establish that the full Bayesian inference comparison completes or that covariates improve prediction; those results require a separate inference run.
