from .Estimator import Estimator
from jax import random
from numpyro.infer import SVI as NumPyroSVI,Trace_ELBO,Predictive
from numpyro.infer.autoguide import AutoNormal,AutoMultivariateNormal
from numpyro.optim import Adam
import numpy as np
from numpyro.infer import init_to_median


class SviEstimator(Estimator):
    _printMode="after"
    def __init__(self,numSteps=10000,numSamples=2000,stepSize=0.01,numParticles=1,progressBar=True,medianSamples=50,multivariate=False):
        for value,name in ((numSteps,"numSteps"),(numSamples,"numSamples"),(numParticles,"numParticles"),(medianSamples,"medianSamples")):
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)): raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
            if value<=0: raise ValueError(f"{name} must be > 0.")
        if not isinstance(progressBar,(bool,np.bool_)): raise TypeError(f"progressBar must be bool; got {type(progressBar).__name__}.")
        if not isinstance(multivariate,(bool,np.bool_)): raise TypeError(f"multivariate must be bool; got {type(multivariate).__name__}.")
        if isinstance(stepSize,(bool,np.bool_)) or not isinstance(stepSize,(int,float,np.integer,np.floating)):
            raise TypeError(f"stepSize must be a real number; got {type(stepSize).__name__}.")
        if not np.isfinite(stepSize) or stepSize<=0: raise ValueError("stepSize must be finite and > 0.")
        self.numSteps=numSteps
        self.numSamples=numSamples
        self.stepSize=stepSize
        self.numParticles=numParticles
        self.progressBar=progressBar
        self.medianSamples=medianSamples
        self.multivariate=multivariate
        self._params={}

    def _Fit(self,ModelFn,rngKey,samplePrefixes):
        fitKey,sampleKey=random.split(rngKey)
        Guide=AutoMultivariateNormal if self.multivariate else AutoNormal
        guide=Guide(ModelFn,init_loc_fn=init_to_median(num_samples=self.medianSamples))
        svi=NumPyroSVI(ModelFn,guide,Adam(self.stepSize),Trace_ELBO(num_particles=self.numParticles))
        output=svi.run(fitKey,self.numSteps,progress_bar=self.progressBar)
        self._params=output.params
        predictive=Predictive(ModelFn,guide=guide,params=self._params,num_samples=self.numSamples)
        post=predictive(sampleKey)
        samples={name:value for name,value in post.items() if any(name.startswith(prefix) for prefix in samplePrefixes)}
        return {"output":output,"samples":samples}


    def _Print(self,output,localNames,globalNames,title=None):
        if title is not None: print(title)
        losses=np.asarray(output.losses)
        print(f"SVI steps: {len(losses)}")
        if len(losses)>0:
            print(f"Initial loss: {losses[0]:.6g}")
            print(f"Final loss:   {losses[-1]:.6g}")

    def _SaveArgs(self):
        return {"numSteps":self.numSteps,"numSamples":self.numSamples,"stepSize":self.stepSize,"numParticles":self.numParticles,"progressBar":self.progressBar,"medianSamples":self.medianSamples,"multivariate":self.multivariate}

