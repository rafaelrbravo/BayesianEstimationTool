# Testing BET

Install the package and run the complete suite from the repository root:

```bash
python -m pip install -e . pytest
cd ..
JAX_PLATFORMS=cpu python -m pytest -q BayesianEstimationTool/tests
```

The GitHub Actions workflow at `.github/workflows/tests.yml` runs the same suite on Python 3.11. A successful commit or workflow configuration is **not** evidence that the tests passed; inspect the actual job conclusion and pytest output.

## Test coverage

| File | Scope |
| --- | --- |
| `test_method_contract.py` | Built-in Method signatures, saved constructor arguments, print hooks |
| `test_method_validation.py` | Structural validator acceptance and helpful rejection cases |
| `test_estimator_validation.py` | Argument/state guards and persistence before training |
| `test_estimator_integration.py` | Small SVI Train/Test/Score/Save/Load workflows, sample alignment, local-only models |
| `test_nuts_numerics.py` | Conjugate normal posterior and seeded NUTS reproducibility |
| `test_custom_method.py` | A user-defined Method through Train, Test, and scoring |

Tests with SVI or NUTS exercise JAX compilation and may take longer on their first run. The numerical NUTS test uses loose Monte Carlo tolerances and is not a general convergence guarantee. The custom Method uses prior-predictive sampling only to exercise the protocol; it does not infer a posterior.

## Investigating failures

Run one failing test with a traceback before changing the implementation:

```bash
JAX_PLATFORMS=cpu python -m pytest -x -vv BayesianEstimationTool/tests/test_estimator_integration.py
```

Distinguish test assumptions from library defects. Check posterior array axes (draw, individual; or global draw, local draw, individual), saved-data flags, and whether the chosen Method returned deterministic sample sites requested by BET. Verify fixes by rerunning both the failing test and the complete suite.

Structural `ValidateMethod` checks signatures and constructor configuration only. It does not execute `_Fit` or establish that returned samples are valid. Use `test_custom_method.py` as a starting point for an actual Method integration test.

## Status

The suite must be executed in an environment with NumPyro and JAX installed before any claim of passing tests or release readiness.

CI verification branch: this documentation-only change triggers pull-request checks before merging.
