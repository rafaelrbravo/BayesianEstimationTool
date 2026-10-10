"""Validation should reject malformed interfaces without executing inference."""
import pytest
from BayesianEstimationTool import ValidateMethod, Nuts, Svi


@pytest.mark.parametrize("method", [Nuts(numWarmup=0, numSamples=2, numChains=1), Svi(numSteps=1, numSamples=2, progressBar=False)])
def test_builtins_validate(method):
    assert ValidateMethod(method) is True


def test_class_not_instance():
    with pytest.raises(TypeError, match="instance"):
        ValidateMethod(Nuts)


def test_bad_fit_signature():
    class Bad:
        def _Fit(self, model):
            pass
        def _SaveArgs(self):
            return {}
    with pytest.raises(TypeError, match="_Fit"):
        ValidateMethod(Bad())


def test_bad_save_args():
    class Bad:
        def _Fit(self, model, key, prefixes):
            pass
        def _SaveArgs(self):
            return []
    with pytest.raises(TypeError, match="dictionary"):
        ValidateMethod(Bad())


def test_bad_constructor_configuration():
    class Bad:
        def __init__(self, steps=1):
            pass
        def _Fit(self, model, key, prefixes):
            pass
        def _SaveArgs(self):
            return {"unknown": 2}
    with pytest.raises(TypeError, match="constructor"):
        ValidateMethod(Bad())


def test_bad_print_hook():
    class Bad:
        _printMode = "after"
        def _Fit(self, model, key, prefixes):
            pass
        def _SaveArgs(self):
            return {}
    with pytest.raises(TypeError, match="_Print"):
        ValidateMethod(Bad())


def test_inherited_base_stub_rejected():
    from BayesianEstimationTool.Method import Method
    with pytest.raises(TypeError, match="override inherited _Fit"):
        ValidateMethod(Method())
