"""Structural contract tests for BET inference Methods (no inference execution)."""
import inspect
import pytest
from BayesianEstimationTool.Method import Method
from BayesianEstimationTool.NutsMethod import Nuts
from BayesianEstimationTool.SviMethod import Svi


@pytest.mark.parametrize("method", [Nuts(numWarmup=0, numSamples=2, numChains=1), Svi(numSteps=1, numSamples=2, progressBar=False)])
def test_builtin_method_structure(method):
    assert isinstance(method, Method)
    assert callable(method._Fit)
    assert callable(method._SaveArgs)
    assert list(inspect.signature(method._Fit).parameters) == ["ModelFn", "rngKey", "samplePrefixes"]
    args = method._SaveArgs()
    assert isinstance(args, dict)
    assert type(method)(**args)._SaveArgs() == args
    assert method._printMode in (None, "before", "after", "replace")
    if method._printMode is not None:
        assert callable(method._Print)


def test_nuts_local_z_summary_prefix():
    import inspect
    assert '"__localZ__"' in inspect.getsource(Nuts._Print)
