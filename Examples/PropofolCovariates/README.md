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

**Status:** These are preparation and validation scripts, not a fitted model. Their full-data execution has not yet been verified. Next implement a padded per-patient event loader, a JAX three-compartment solver, the two hierarchical BET models, and paired held-out scoring. Do not derive covariate coefficients or preprocessing statistics from held-out patients.
