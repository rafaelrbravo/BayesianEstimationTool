# Propofol covariate example

This directory is reserved for a BET example using the three-compartment propofol pharmacokinetic model and published covariate relationships from [Learning pharmacometric covariate model structures with symbolic regression networks](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures).

## Planned comparison

- Fit the same hierarchical three-compartment model **without covariates** and **with the published covariate functional forms**.
- Estimate model coefficients using training patients only; do not reuse full-dataset fitted coefficients.
- Evaluate both models on identical held-out patients, first with no individual concentration measurements and then after conditioning on early measurements.
- Compare forward predictive error and uncertainty calibration.

Implementation, data provenance, and reproducible commands will be added as the example develops.


## Data preparation (implemented)

Install `numpy` and `pandas`. Run `python Examples/PropofolCovariates/download_data.py` to fetch the original CSVs (existing files are not overwritten), or download them manually from the upstream [symbreg/data directory](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures/tree/main/symbreg/data) into `Examples/PropofolCovariates/data/`:

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

Additional local validation reproduced the loader's NumPy/pandas event-expansion logic and passed three focused checks (multiple repeated-measurement events, a single-measurement event, and measurement-count mismatch). This was not a direct execution of the repository loader test module. The full loader test suite and original-data validation are still outstanding.

The downloader has three offline regression tests covering successful downloads and skipped existing files, empty responses, and interrupted transfers; run `python -m pytest -q Examples/PropofolCovariates/test_download_data.py`. These checks do not require the original CSVs or NumPyro.

Before fitting, `inspect_data.py` now checks each included patient's positive-concentration row count against the summed `IsMeasurement` event counts and prints the validated total. Offline tests: `python -m pytest -q Examples/PropofolCovariates/test_inspect_data.py`.

For the planned **831-training/200-test** experiment, run `python Examples/PropofolCovariates/compare_covariates.py --full`. The default remains the 100/25 development run. Optional `--train-steps` and `--test-steps` control SVI iteration counts (defaults 1000 and 300); the full run may be computationally expensive and has not yet been executed. The current covariate model remains an exploratory linear effect model, not the published symbolic-regression equations.

The patient split has standalone reproducibility/leakage checks: `python -m pytest -q Examples/PropofolCovariates/test_split.py` (six tests). They also run in the example's GitHub Actions workflow.

**Future-only evaluation eligibility:** The patient split is fixed before modeling. The first three measured concentrations are used for optional conditioning, and later measurements for scoring. Held-out patients with three or fewer measurements cannot contribute a future score, so both scripts exclude them from evaluation without replacement and report the evaluable fraction. The full split contains 200 held-out patients, but fewer may have future measurements. This eligibility filter does not use concentration values or model predictions.

**Paired uncertainty:** The comparison reports a patient-level bootstrap 95% interval for the mean covariate-minus-baseline future log-error difference (2,000 resamples, fixed seed), separately for unconditional and early-conditioned predictions. This is a descriptive percentile bootstrap interval, not a posterior credible interval; it does not account for refitting models across resamples. Check with `python -m pytest -q Examples/PropofolCovariates/test_bootstrap.py` (requires only NumPy and pytest, with no NumPyro/JAX or original data). The check is included in CI; CI success and real-data inference have not been verified.

The bootstrap helper is in `paired_stats.py`, independent of the inference code. A standalone reproduction of its core checks passed locally; the repository's pytest module and GitHub Actions execution have not been confirmed.

### Published-model compatibility checkpoint

The upstream `symbreg/checkresults/get_results.jl` identifies its six learned outputs as **`[k10, k12, k13, k21, k31, V1]`**, not the `[V1, V2, V3, CL, Q2, Q3]` used in this example's generic three-compartment parameterization. The published outputs are first learned in a scaled representation and converted to physical units by the upstream equation reader. Its final expression-printing code rescales age by 88, weight by 160, BMI by 52.84713965, and uses sex encoded as -0.5/0.5. **Do not insert published rate-constant equations directly into the current six-volume/clearance parameter slots.** Recover and validate the actual equations and map rates into the compartment matrix (or explicitly convert to equivalent volumes/clearances) before claiming a published-structure comparison. The current exploratory covariate model remains valid as its own baseline comparison but is not the paper's parameterization or equation family.
