class Method:
    _printMode=None

    def _Fit(self,ModelFn,rngKey,samplePrefixes):
        raise NotImplementedError

    _Print=None

    def _SaveArgs(self):
        raise NotImplementedError
