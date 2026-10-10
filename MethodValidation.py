"""Structural validation of BET inference Method instances.

This checks interfaces and saved configuration only; it does not run inference.
"""
import inspect


def ValidateMethod(method):
    """Validate a configured BET Method; raise a specific error or return True.

    _Fit must accept (ModelFn, rngKey, samplePrefixes) positionally and
    return {"output": ..., "samples": {site_name: sample_array, ...}}
    when executed. Runtime return values are deliberately not inspected here.
    """
    if method is None or inspect.isclass(method):
        raise TypeError("Method must be a configured instance, not None or a class.")
    for name, nargs in (("_Fit", 3), ("_SaveArgs", 0)):
        fn = getattr(method, name, None)
        if not callable(fn):
            raise TypeError(f"Method must implement callable {name}().")
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        try:
            signature.bind(*([None] * nargs))
        except TypeError as exc:
            raise TypeError(f"Method.{name} must accept {nargs} positional arguments; got {signature}.") from exc
    mode = getattr(method, "_printMode", None)
    if mode not in (None, "before", "after", "replace"):
        raise ValueError("Method._printMode must be None, 'before', 'after', or 'replace'.")
    if mode is not None:
        fn = getattr(method, "_Print", None)
        if not callable(fn):
            raise TypeError("Method._Print must be callable when _printMode is set.")
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            pass
        else:
            try:
                signature.bind(None, None, None, None)
            except TypeError as exc:
                raise TypeError(f"Method._Print must accept output, localNames, globalNames, title; got {signature}.") from exc
    try:
        args = method._SaveArgs()
    except Exception as exc:
        raise ValueError("Method._SaveArgs() raised an exception.") from exc
    if not isinstance(args, dict):
        raise TypeError("Method._SaveArgs() must return a dictionary.")
    if any(not isinstance(key, str) for key in args):
        raise TypeError("Method._SaveArgs() keys must be strings.")
    try:
        inspect.signature(type(method)).bind(**args)
    except (TypeError, ValueError) as exc:
        raise TypeError("Method._SaveArgs() keys must be accepted by the Method constructor.") from exc
    return True
