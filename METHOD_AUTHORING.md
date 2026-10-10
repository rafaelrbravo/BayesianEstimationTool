# Writing a BET inference Method

BET Methods isolate inference algorithms from the estimator's population and individual workflows. Start with the smallest possible interface and add optional functionality only when useful.

## Minimum viable Method

A Method instance must implement two methods:

- `_Fit(ModelFn, rngKey, samplePrefixes)`: run inference and return a dictionary with `"output"` (algorithm-specific diagnostics/state) and `"samples"` (posterior-like draws).
- `_SaveArgs()`: return a dictionary of constructor keyword arguments so BET can reconstruct the Method after loading saved state.

`samples` maps NumPyro sample-site names to arrays whose first dimension indexes draws. Keep only sites beginning with one of `samplePrefixes`; BET uses these names to extract global parameters, local Z values, Cholesky factors, and optionally model outputs.

A skeleton for adapting an existing inference backend:

```python
class MyMethod:
    _printMode = None

    def __init__(self, nSamples=100):
        self.nSamples = nSamples

    def _Fit(self, ModelFn, rngKey, samplePrefixes):
        # Replace with a real inference backend:
        rawSamples, backendOutput = run_my_inference(ModelFn, rngKey, self.nSamples)
        samples = {name: value for name, value in rawSamples.items()
                   if any(name.startswith(prefix) for prefix in samplePrefixes)}
        return {"output": backendOutput, "samples": samples}

    def _SaveArgs(self):
        return {"nSamples": self.nSamples}
```

`run_my_inference` is a placeholder, not a BET function. The backend must generate meaningful posterior or explicitly identified approximate-posterior samples; repeating a MAP point does not quantify uncertainty.

Validate the structure before attempting a full fit:

```python
import BayesianEstimationTool as bet
method = MyMethod(nSamples=100)
bet.ValidateMethod(method)
```

The validator checks signatures, configuration keys and printing hooks. It deliberately does **not** execute `_Fit`, check returned sample shapes, or assess statistical correctness. Verify those properties separately with small synthetic models and posterior predictive tests.

## Optional capabilities

1. **Diagnostics and printing:** Set `_printMode` to `"before"`, `"after"`, or `"replace"` and implement `_Print(output, localNames, globalNames, title=None)`. With `None`, BET uses its generic sample summary.
2. **Configuration:** Add constructor options for optimization, initialization, sampling and progress reporting; return all reproducible options from `_SaveArgs()`.
3. **Persistence:** Keep inference configuration reconstructible through constructor keyword arguments. The `output` object is returned to callers but is not automatically serialized as an inference-engine session.
4. **Inference quality:** Compare posterior summaries, predictive distributions and computational costs against NUTS on a known model. This is behavioral testing, outside structural validation.

See `NutsMethod.py` and `SviMethod.py` for complete working Method implementations.
