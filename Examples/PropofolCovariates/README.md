# Propofol covariate example

This directory is reserved for a BET example using the three-compartment propofol pharmacokinetic model and published covariate relationships from [Learning pharmacometric covariate model structures with symbolic regression networks](https://github.com/wahlquisty/learning-pharmacometric-covariate-structures).

## Planned comparison

- Fit the same hierarchical three-compartment model **without covariates** and **with the published covariate functional forms**.
- Estimate model coefficients using training patients only; do not reuse full-dataset fitted coefficients.
- Evaluate both models on identical held-out patients, first with no individual concentration measurements and then after conditioning on early measurements.
- Compare forward predictive error and uncertainty calibration.

Implementation, data provenance, and reproducible commands will be added as the example develops.
