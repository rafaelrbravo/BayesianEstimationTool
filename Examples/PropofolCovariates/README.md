# Propofol covariate example

This directory is reserved for a BET example using the three-compartment propofol pharmacokinetic model and published covariate relationships from [Learning pharmacometric covariate model structures with symbolic regression networks](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures).

## Planned comparison

- Fit the same hierarchical three-compartment model **without covariates** and **with the published covariate functional forms**.
- Estimate model coefficients using training patients only; do not reuse full-dataset fitted coefficients.
- Evaluate both models on identical held-out patients, first with no individual concentration measurements and then after conditioning on early measurements.
- Compare forward predictive error and uncertainty calibration.

Implementation, data provenance, and reproducible commands will be added as the example develops.


## Data preparation (implemented)

Install `numpy` and `pandas`. Download the three original CSVs from the upstream [symbreg/data directory](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures/tree/main/symbreg/data) into `Examples/PropofolCovariates/data/`:

- `dataframe.csv` (concentrations and covariates)
- `eleveld_infusiondata.csv` (infusions, boluses, measurement event times)
- `eleveld_modelparams.csv` (reference pharmacokinetic parameters; **not** training targets)

Run `python Examples/PropofolCovariates/inspect_data.py` to check the tables and the 1,031 usable patients. Upstream excludes IDs 893 and 897 for absent observations. Run `python Examples/PropofolCovariates/split.py` for the fixed patient-level split. Split indices refer to the sorted usable patient-ID list, **not** the original ID values. The development subset uses the first 100 training and first 25 test indices from the full randomized split.

**Status:** A padded loader, differentiable three-compartment solver, BET baseline, and exploratory covariate comparison are implemented, but full-data inference has not been run. The exploratory model uses AGE, WGT, BMI, centered sex, and centered arterial/venous status, as in the upstream data pipeline. It uses linear log-parameter effects rather than the published symbolic-regression equations. The original equations are extracted from a trained BSON network by upstream `symbreg/checkresults/get_results.jl`; do not describe the current comparison as a reproduction of the published model. Fit coefficients and scalers on training patients only.


## Local validation and inference

From the BET repository root, with `numpy`, `pandas`, `scipy`, `jax`, and `numpyro` installed:

```bash
python Examples/PropofolCovariates/test_load_data.py
python Examples/PropofolCovariates/test_pk_model.py
python Examples/PropofolCovariates/inspect_data.py
python Examples/PropofolCovariates/bet_baseline.py
python Examples/PropofolCovariates/compare_covariates.py
```

The synthetic checks include the loader, PK solver, and future-only scoring test (`python Examples/PropofolCovariates/test_scoring.py`). The inspection and two inference commands require the original CSVs. BET must be importable from the repository root (for example, install the package in editable mode). **Inference results are not yet available**; these commands are a validation plan, not a claim of completed runs. The baseline and exploratory comparison currently use the 100/25 development subset, not the planned 831/200 final evaluation. Both development fits now request only 50 SVI posterior draws; the default BET SVI draw count is 2,000 and would otherwise add substantial unnecessary sampling and prediction work.


**Validation note (2026-10-16):** The PK solver and its independent SciPy reference test were run successfully in a local reproduction of the repository code. A duplicate-event-time regression test was also executed successfully (2 tests passed). This is not a successful full-repository CI run or an inference result. NumPyro inference and source-data validation remain outstanding.
