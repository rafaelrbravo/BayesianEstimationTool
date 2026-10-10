from .Method import Method
from numpyro.infer import init_to_median,NUTS,MCMC
from numpyro.diagnostics import summary
import numpy as np


class Nuts(Method):
    _printMode="replace"

    def __init__(self,numWarmup=2000,numSamples=2000,numChains=4,acceptProb=0.95,denseMass=False,medianSamples=50):
        if isinstance(numWarmup,(bool,np.bool_)) or not isinstance(numWarmup,(int,np.integer)): raise TypeError(f"numWarmup must be an integer; got {type(numWarmup).__name__}.")
        if numWarmup<0: raise ValueError("numWarmup must be >= 0.")
        for value,name in ((numSamples,"numSamples"),(numChains,"numChains"),(medianSamples,"medianSamples")):
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)): raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
            if value<=0: raise ValueError(f"{name} must be > 0.")
        if isinstance(acceptProb,(bool,np.bool_)) or not isinstance(acceptProb,(int,float,np.integer,np.floating)): raise TypeError(f"acceptProb must be a real number; got {type(acceptProb).__name__}.")
        if not np.isfinite(acceptProb): raise ValueError("acceptProb must be finite.")
        if not 0<acceptProb<1: raise ValueError("acceptProb must be between 0 and 1.")
        if not isinstance(denseMass,(bool,np.bool_)): raise TypeError(f"denseMass must be bool; got {type(denseMass).__name__}.")
        self.numWarmup=numWarmup
        self.numSamples=numSamples
        self.numChains=numChains
        self.acceptProb=acceptProb
        self.denseMass=denseMass
        self.medianSamples=medianSamples

    def _Fit(self,ModelFn,rngKey,samplePrefixes):
        kernel=NUTS(ModelFn,init_strategy=init_to_median(num_samples=self.medianSamples),target_accept_prob=self.acceptProb,dense_mass=self.denseMass)
        mcmc=MCMC(kernel,num_warmup=self.numWarmup,num_samples=self.numSamples,num_chains=self.numChains,chain_method="parallel")
        mcmc.run(rngKey,extra_fields=("num_steps","accept_prob"))
        post=mcmc.get_samples(group_by_chain=False)
        samples={name:value for name,value in post.items() if any(name.startswith(prefix) for prefix in samplePrefixes)}
        return {"output":mcmc,"samples":samples}

    def _Print(self,output,localNames,globalNames,title=None):
        stats=summary(output.get_samples(group_by_chain=True),prob=0.95)
        localNames=list(localNames)
        globalNames=[] if globalNames is None else list(globalNames)
        if title is not None: print(title)
        print(f"{'parameter':<20} {'mean':>10} {'std':>10} {'median':>10} {'2.5%':>10} {'97.5%':>10} {'n_eff':>10} {'r_hat':>8}")
        for prefix,names in (("__globalParam__",globalNames),("__localZ__",localNames)):
            for name in names:
                statName=f"{prefix}{name}"
                if statName not in stats: continue
                shape=np.shape(stats[statName]["mean"])
                if not shape: self._PrintRow(name,stats[statName])
                else:
                    for inds in np.ndindex(shape): self._PrintRow(f"{name}[{','.join(map(str,inds))}]",stats[statName],inds)

    def _SaveArgs(self):
        return {"numWarmup":self.numWarmup,"numSamples":self.numSamples,"numChains":self.numChains,"acceptProb":self.acceptProb,"denseMass":self.denseMass,"medianSamples":self.medianSamples}

    def _PrintRow(self,label,stats,inds=()):
        def get(key):
            x=np.asarray(stats[key])
            return x[inds] if inds else x.item()
        print(f"{label:<20} {get('mean'):>10.3f} {get('std'):>10.3f} {get('median'):>10.3f} {get('2.5%'):>10.3f} {get('97.5%'):>10.3f} {get('n_eff'):>10.1f} {get('r_hat'):>8.2f}")
