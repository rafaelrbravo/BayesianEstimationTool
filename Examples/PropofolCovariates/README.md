# Propofol covariate example

This directory implements an exploratory BET example using a three-compartment propofol pharmacokinetic model, inspired by [Learning pharmacometric covariate model structures with symbolic regression networks](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures).

## GitHub Actions inference

Run **Propofol covariate inference** with **Run workflow**. The default settings (`smoke_only=true`, `run_published=false`, `full_cohort=false`) run the synthetic checks, data validation, and 10-train/5-test exploratory SVI smoke test without requiring a Julia export. Set `smoke_only=false` to run the 100-train/up-to-25-test development comparison and Eleveld/SciPy reference checks, or additionally set `full_cohort=true` for the resource-intensive full cohort. Set `run_published=true` to additionally run the fixed published-equation comparison; that option requires a successful **Propofol symbolic equation export** artifact for the exact BET commit. The published model is descriptive because its original coefficients were trained on overlapping patients. The inference workflow is manual-only; push-triggered synthetic checks run in the separate test workflow.

## Comparison design

- Compare the same hierarchical three-compartment model **without covariates** against (1) exploratory linear log-parameter covariate effects and (2) the fixed published symbolic covariate equations once the Julia export has been validated. The fixed published coefficients were trained on an overlapping cohort, so this second comparison is descriptive, not leakage-free generalization.
- Estimate model coefficients using training patients only; do not reuse full-dataset fitted coefficients.
- Evaluate both models on identical held-out patients, first with no individual concentration measurements and then after conditioning on early measurements.
- Compare future-only absolute log prediction error using paired patient-level differences and a bootstrap interval. Predictive uncertainty calibration remains future work.


## Data preparation (implemented)

Install `numpy` and `pandas`. Run `python Examples/PropofolCovariates/download_data.py` to fetch the original CSVs (existing files are verified against the pinned upstream commit and never silently overwritten), or download them manually from the upstream [symbreg/data directory](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures/tree/main/symbreg/data) into `Examples/PropofolCovariates/data/`:

- `dataframe.csv` (concentrations and covariates)
- `eleveld_infusiondata.csv` (infusions, boluses, measurement event times)
- `eleveld_modelparams.csv` (reference pharmacokinetic parameters; **not** training targets)

Run `python Examples/PropofolCovariates/inspect_data.py` to check the tables and the 1,031 usable patients.

**Infusion-rate units:** The upstream `symbreg/data/README.md` documents `eleveld_infusiondata.csv` `InfusionRate` as `ug/s` and `TIME` as seconds. The Julia reader contains a conflicting `ug/min` comment, but passes raw values into `FastPKSim.pk3sim!`, whose time steps are in seconds. We therefore preserve the raw rate values; `ug/s` is dimensionally consistent with bolus doses in micrograms, volumes converted to mL, and concentrations in `ug/mL`. Patient-level reference comparisons remain necessary to validate numerical implementation fidelity, but are not needed to resolve this source-unit label. `Bolus` is in micrograms and event `Time` is in seconds. The original observation `TIME` is in minutes, rounded to the nearest minute. `inspect_data.py` converts it to seconds and validates alignment with a 30.5-second tolerance (half a minute plus a small floating-point margin). The upstream dataset contains 1,031 usable patients after excluding IDs 893 and 897, which have no concentration measurements. Run `python Examples/PropofolCovariates/split.py` for the fixed patient-level split. Split indices refer to the sorted usable patient-ID list, **not** the original ID values. The development subset uses the first 100 training indices and the first 25 eligible held-out patients (at least four measured concentrations) from the fixed 200-patient test partition.

**Status:** A padded loader, differentiable three-compartment solver, BET baseline, and exploratory covariate comparison are implemented, but full-data inference has not been run. The exploratory model uses AGE, WGT, BMI, centered sex, and centered arterial/venous status, as in the upstream data pipeline. It uses linear log-parameter effects rather than the published symbolic-regression equations. The original equations are stored in the upstream trained BSON network. An exporter is now provided at `export_symbolic_equations.jl`; run it with Julia and a local checkout of the upstream repository to produce `symbolic_equations.toml` (instructions below). The exploratory BET comparison remains separate from the published-model comparison; `compare_published.py` implements the latter, but cannot run until the exported equations have been validated against the original Julia model.

To export the exact published expressions, install Julia, clone `https://github.com/wahlquisty/learning-pharmacometric-covariate-structures`, then run from the BET repository root:

```bash
julia Examples/PropofolCovariates/export_symbolic_equations.jl /path/to/learning-pharmacometric-covariate-structures
```

The script activates the upstream `symbreg/Project.toml`, loads the committed `bson/bson_ALE/nn_thread1loss_ALE.bson` model, and writes `Examples/PropofolCovariates/symbolic_equations.toml`. It does not train the network or require the patient CSVs. The exporter includes a built-in numerical check against the original Julia network functions for five representative covariate combinations, including male/female substitutions for `k13` and `k31`. The TOML artifact records the exact upstream commit, trained BSON model's Git blob SHA, Julia version, and numerical-validation tolerances alongside the equations. The exporter has not yet completed successfully in hosted CI, so the self-check remains unexecuted; integrate the output into BET only after the workflow passes. The upstream extraction code explicitly uses normalized covariates `AGE/88`, `WGT/160`, `BMI/52.84713965`, centered sex (`M1F2 - 1.5`), and centered arterial/venous status (`A1V2 - 1.5`), and produces rate constants `k10`, `k12`, `k13`, `k21`, `k31` in inverse seconds and central volume `V1` in milliliters. It prints separate male/female expressions for `k13` and `k31`. The exporter reconstructs these expressions from the trained BSON model. The published-model comparison runner is implemented but has not been executed successfully because the validated export is still unavailable. The exploratory log-linear model does not implement the published functional forms. Fit coefficients and scalers on training patients only.


## Published-equation JAX consumer (implemented; awaiting exported TOML)

`published_equations.py` now loads the Julia-exported `symbolic_equations.toml`, parses its six expressions using a restricted arithmetic syntax (without Python `eval`), evaluates them with differentiable JAX operations, selects the original male/female expressions for `k13` and `k31`, and converts `[k10,k12,k13,k21,k31,V1]` to the simulator's `[V1,V2,V3,CL,Q2,Q3]` parameter order and units. Pass **raw**, not training-standardized, `[AGE,WGT,BMI,GDR,AV]` inputs; gender is -0.5/+0.5. Synthetic conversion, differentiation, and parser-safety tests are in `test_published_equations.py`. The exporter has not yet generated a verified TOML artifact, so real published equations have **not** been exercised through this consumer.

**Interpretation caveat:** The published BSON model was fitted on the upstream Eleveld patient cohort, which overlaps the patient data used here. Predictions from those fixed published coefficients are therefore **not independent held-out generalization estimates** on this cohort, even if our BET split is disjoint. The exploratory linear BET comparison does fit its coefficients only on its training partition. Any eventual comparison involving the original published model must explicitly disclose this pretraining overlap; a leakage-free comparison would require retraining the published structure on training patients only or evaluating on an independent external cohort.

### Fixed published-model comparison runner

After producing and validating `symbolic_equations.toml`, run `python Examples/PropofolCovariates/compare_published.py` for the 100-training/25-evaluable-test development experiment, or append `--full` for the larger cohort. Use `--equations PATH` for a nondefault TOML path. This runs the same baseline, early-measurement conditioning, future-only score and paired bootstrap as the exploratory comparison, but uses **raw** covariates and fixed published equations. **Important:** The published coefficients were learned on overlapping patients; this is a descriptive compatibility comparison, not independent held-out generalization. This runner and its inference have not yet been executed successfully.

## Local validation and inference

From the BET repository root, with `numpy`, `pandas`, `scipy`, `jax`, and `numpyro` installed:

```bash
python Examples/PropofolCovariates/test_load_data.py
python Examples/PropofolCovariates/test_pk_model.py
python Examples/PropofolCovariates/inspect_data.py
python Examples/PropofolCovariates/bet_baseline.py
python Examples/PropofolCovariates/compare_covariates.py
```

The synthetic checks include the loader, PK solver, and future-only scoring test (`python Examples/PropofolCovariates/test_scoring.py`). The inspection and two inference commands require the original CSVs. BET must be importable from the repository root (for example, install the package in editable mode). **Inference results are not yet available**; these commands are a validation plan, not a claim of completed runs. The baseline and exploratory comparison default to the 100/25 development subset; `compare_covariates.py --full` selects all 831 training patients and eligible patients in the fixed 200-patient test partition. Both development fits now request only 50 SVI posterior draws; the default BET SVI draw count is 2,000 and would otherwise add substantial unnecessary sampling and prediction work.


**Validation note (2026-10-16):** The PK solver and its independent SciPy reference test were run successfully in a local reproduction of the repository code. A duplicate-event-time regression test was also executed successfully (2 tests passed). This is not a successful full-repository CI run or an inference result. NumPyro inference and source-data validation remain outstanding.

Additional local validation reproduced the loader's NumPy/pandas event-expansion logic and passed three focused checks (multiple repeated-measurement events, a single-measurement event, and measurement-count mismatch). This was not a direct execution of the repository loader test module. The full loader test suite and original-data validation are still outstanding.

The downloader has three offline regression tests covering successful downloads and skipped existing files, empty responses, and interrupted transfers; run `python -m pytest -q Examples/PropofolCovariates/test_download_data.py`. These checks do not require the original CSVs or NumPyro.

Before fitting, `inspect_data.py` now checks each included patient's positive-concentration row count against the summed `IsMeasurement` event counts and prints the validated total. Offline tests: `python -m pytest -q Examples/PropofolCovariates/test_inspect_data.py`.

For the planned **831-training/200-test** experiment, run `python Examples/PropofolCovariates/compare_covariates.py --full`. The default remains the 100/25 development run. Optional `--train-steps` and `--test-steps` control SVI iteration counts (defaults 1000 and 300); the full run may be computationally expensive and has not yet been executed. The current covariate model remains an exploratory linear effect model, not the published symbolic-regression equations.

The patient split has standalone reproducibility/leakage checks: `python -m pytest -q Examples/PropofolCovariates/test_split.py`. They also run in the example's GitHub Actions workflow.

**Future-only evaluation eligibility:** The patient split is fixed before modeling. The first three measured concentrations are used for optional conditioning, and later measurements for scoring. Held-out patients with three or fewer measurements cannot contribute a future score, so both scripts exclude them from evaluation without replacement and report the evaluable fraction. The full split contains 200 held-out patients, but fewer may have future measurements. This eligibility filter does not use concentration values or model predictions.

**Paired uncertainty:** The comparison reports a patient-level bootstrap 95% interval for the mean covariate-minus-baseline future log-error difference (2,000 resamples, fixed seed), separately for unconditional and early-conditioned predictions. This is a descriptive percentile bootstrap interval, not a posterior credible interval; it does not account for refitting models across resamples. Check with `python -m pytest -q Examples/PropofolCovariates/test_bootstrap.py` (requires only NumPy and pytest, with no NumPyro/JAX or original data). The check is included in CI; CI success and real-data inference have not been verified.

The bootstrap helper is in `paired_stats.py`, independent of the inference code. A standalone reproduction of its core checks passed locally; the repository's pytest module and GitHub Actions execution have not been confirmed.

### Published-model compatibility checkpoint

The upstream `symbreg/checkresults/get_results.jl` identifies its six learned outputs as **`[k10, k12, k13, k21, k31, V1]`**, not the `[V1, V2, V3, CL, Q2, Q3]` used in this example's generic three-compartment parameterization. The published outputs are first learned in a scaled representation and converted to physical units by the upstream equation reader. Its final expression-printing code rescales age by 88, weight by 160, BMI by 52.84713965, and uses sex encoded as -0.5/0.5. **Do not insert published rate-constant equations directly into the current six-volume/clearance parameter slots.** Recover and validate the actual equations and map rates into the compartment matrix (or explicitly convert to equivalent volumes/clearances) before claiming a published-structure comparison. The current exploratory covariate model remains valid as its own baseline comparison but is not the paper's parameterization or equation family.

**Infusion-rate provenance and remaining validation:** The upstream data README labels `InfusionRate` as µg/s, whereas a Julia reader comment says µg/min. The original seconds-based simulator consumes the raw rates without an evident conversion, supporting the README's µg/s interpretation. This example therefore treats raw infusion rates as µg/s, consistent with boluses in µg and event times in seconds. Numerical agreement with the original simulator on real patient histories is still unverified; do not conflate this pending numerical parity check with the documented unit interpretation. Resolve this by comparing the original and independent predictions on identical patient histories before interpreting dose-related errors. The published symbolic network returns rate constants in 1/s and `V1` in mL; `pk_model.published_rates_to_parameters` converts its volume to liters before translating rates into the solver's volume/clearance parameterization. The volume conversion is covered by synthetic regression assertions; raw-rate physical interpretation is not validated. Direct repository test execution remains pending.

Optional `--output-csv results.csv` saves patient IDs and paired held-out scores for independent analysis. No inference results have been generated yet.

The separate GitHub Actions workflow **Propofol upstream data validation** (`.github/workflows/propofol-data-validation.yml`) can be started manually via **Actions → Propofol upstream data validation → Run workflow**. It downloads the three original CSVs on a GitHub runner and validates all 1,031 patients and their event arrays without running Bayesian inference. It also runs automatically when the downloader, inspector, loader, or validation workflow changes (including matching pull requests); this check depends on upstream data availability. After validating the 1,031-patient dataset, it scores supplied Eleveld PK parameters on the development held-out patients and uploads `propofol-dev-eleveld.csv` as an artifact. This reference verifies the event loader, PK solver, and scoring path, but is **not** a BET-trained model or a covariate-comparison result. No successful workflow run has yet been verified.

The optional `eleveld_reference.py` evaluates supplied patient-specific Eleveld parameters on the same held-out patient partition using the future-only log-error metric. Run it after downloading data. These parameters come from the existing Eleveld model, not from training BET on the prescribed training split, so treat this as a solver/data sanity reference rather than a third controlled model arm.

To run inference on a GitHub runner, manually launch **Actions → Propofol covariate inference → Run workflow** (`.github/workflows/propofol-inference.yml`). Select **Smoke only** for the first attempt: it fits both BET models on 10 training patients and scores 5 eligible held-out patients, saving `propofol-smoke-scores.csv` and its log without starting the expensive development fit. The workflow also prints the paired score summary for the smoke cohort, including the direction of the bootstrap interval; with only five patients this is an execution check, **not** reliable evidence of predictive improvement. Leave **Smoke only** unchecked and **Full cohort** unchecked to run the smoke test followed by the development-size comparison. Select **Full cohort** to run the smoke test followed by all 831 training patients and eligible held-out patients from the fixed 200-patient test partition (2,000 training and 500 conditioning SVI steps); this is resource-intensive and remains unverified. The inference workflow is manual-only; no push automatically launches SVI. The separate synthetic regression workflow handles push checks. It installs BET, downloads and validates upstream data, fits both models with SVI on 100 training patients, evaluates up to 25 held-out patients, and uploads `propofol-dev-scores.csv` as a workflow artifact. The job has a six-hour timeout and may fail or require resource optimization; no inference run or numerical outcome has yet been verified. This is exploratory linear covariate modeling, not the paper's published symbolic equations.

The optional Eleveld reference runs on the same development held-out IDs and writes `propofol-dev-eleveld.csv`; it is not a third model trained on the BET split. The inference workflow also runs an independent SciPy Eleveld reference and checks patient-aligned agreement with the JAX reference before any BET fitting. Reference failures stop inference rather than silently allowing a potentially incorrect comparison. The workflow also preserves `propofol-dev-run.log` when BET inference fails.


The separate **Propofol synthetic regression tests** workflow (`.github/workflows/propofol-tests.yml`) runs the entire `test_*.py` suite on relevant code pushes and pull requests, and can also be started manually. It does not need the original dataset. Workflow success has not yet been verified.
