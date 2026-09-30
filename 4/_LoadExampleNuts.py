import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.experimental.ode import odeint

# Workaround to run this example directly from the repository without installing BayesianEstimationTool.
# For normal use, install from the repository folder with `python -m pip install -e .` and import with `from BayesianEstimationTool import BayesianEstimationTool`.
# If VS Code does not recognize the import for autocomplete, add `"python.analysis.extraPaths": ["/path/to/BayesianEstimationTool"]` to settings.json.

import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BayesianEstimationTool import BayesianEstimationTool, NutsEstimator


# ---------------------------------------------------------------------
# Recreate model data
# ---------------------------------------------------------------------

keyG,keyD,keySize,keyNoise=random.split(random.PRNGKey(2),4)

nPatients=10
nTimes=11
times=jnp.linspace(0,10,nTimes)
drugStart=4.0

rho=0.8
zG=random.normal(keyG,(nPatients,))
zDIndependent=random.normal(keyD,(nPatients,))
zD=rho*zG+jnp.sqrt(1.0-rho**2)*zDIndependent

trueG=0.15+0.03*zG
trueD=0.25+0.05*zD

v0=random.uniform(keySize,(nPatients,),minval=0.8,maxval=1.2)

timeOnDrug=jnp.maximum(times-drugStart,0.0)
trueTumorSize=v0[:,None]*jnp.exp(trueG[:,None]*times-trueD[:,None]*timeOnDrug)
tumorSize=trueTumorSize+0.05*random.normal(keyNoise,trueTumorSize.shape)

data={
    "times":jnp.tile(times,(nPatients,1)),
    "drugStart":jnp.full(nPatients,drugStart),
    "v0":v0,
    "tumorSize":tumorSize,
}


# ---------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------

def ODE(V,t,g,d,drugStart):
    drugEffect=jnp.where(t>=drugStart,d,0.0)
    return (g-drugEffect)*V


def ModelFn(globalParams,localParams,data):
    g=globalParams["gMean"]+globalParams["gStd"]*localParams["g"]
    d=globalParams["dMean"]+globalParams["dStd"]*localParams["d"]
    tumorSize=odeint(ODE,data["v0"],data["times"],g,d,data["drugStart"])
    return {"tumorSize":tumorSize}


def GlobalFn(dataShapes):
    return {
        "gMean":npo.sample("gMean",dist.Normal(0.15,0.1)),
        "gStd":npo.sample("gStd",dist.HalfNormal(0.1)),
        "dMean":npo.sample("dMean",dist.Normal(0.25,0.1)),
        "dStd":npo.sample("dStd",dist.HalfNormal(0.1)),
        "sigma":npo.sample("sigma",dist.HalfNormal(0.1)),
    }


# ---------------------------------------------------------------------
# Testing likelihoods
# ---------------------------------------------------------------------

def TestLikelihoodFewFn(globalParams,localParams,data,modelOut):
    nFit=3
    tumorSize=modelOut["tumorSize"]
    mask=jnp.arange(tumorSize.shape[-1])<nFit
    npo.sample("obsTest",dist.Normal(tumorSize,globalParams["sigma"]).mask(mask),obs=data["tumorSize"])


def TestLikelihoodMoreFn(globalParams,localParams,data,modelOut):
    nFit=7
    tumorSize=modelOut["tumorSize"]
    mask=jnp.arange(tumorSize.shape[-1])<nFit
    npo.sample("obsTest",dist.Normal(tumorSize,globalParams["sigma"]).mask(mask),obs=data["tumorSize"])


# ---------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------

def AbsoluteErrorFn(modelOut,data):
    return jnp.mean(jnp.abs(modelOut["tumorSize"]-data["tumorSize"]))


def SquaredErrorFn(modelOut,data):
    return jnp.mean((modelOut["tumorSize"]-data["tumorSize"])**2)


# ---------------------------------------------------------------------
# Load trained framework
#
# Functions are supplied explicitly rather than serialized.
# The original model data are also supplied because saveAllData=False
# was used when the framework was saved.
# ---------------------------------------------------------------------

bf=BayesianEstimationTool.Load("SimpleTumorGrowthNuts.npz",ModelFn=ModelFn,GlobalParamFn=GlobalFn,modelDataFull=data)


# ---------------------------------------------------------------------
# Score training
# ---------------------------------------------------------------------

trainAbsoluteError=bf.ScoreTrain(ScoreFn=AbsoluteErrorFn)
trainSquaredError=bf.ScoreTrain(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test without an estimator
#
# Use the median global parameter values from the training posterior.
# No test observations are used for inference.
#
# nLocalSamples controls the number of local samples generated for each
# test patient.
# ---------------------------------------------------------------------

bf.Test(testIndices=jnp.arange(5,10),TestLikelihoodFn=None,nGlobalSamples=None,nLocalSamples=500)

noMCMCAbsoluteError=bf.ScoreTest(ScoreFn=AbsoluteErrorFn)
noMCMCSquaredError=bf.ScoreTest(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test with 3 observations
#
# Use the median global parameter values from the training posterior.
# NUTS updates each test patient's local parameters using the first
# three observations.
# ---------------------------------------------------------------------

bf.Test(testIndices=jnp.arange(5,10),TestLikelihoodFn=TestLikelihoodFewFn,Estimator=NutsEstimator(numWarmup=500,numSamples=500,numChains=4),nGlobalSamples=None,printSummary=True)

fewAbsoluteError=bf.ScoreTest(ScoreFn=AbsoluteErrorFn)
fewSquaredError=bf.ScoreTest(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test with 7 observations
#
# Repeat the test inference using the first seven observations.
# ---------------------------------------------------------------------

bf.Test(testIndices=jnp.arange(5,10),TestLikelihoodFn=TestLikelihoodMoreFn,Estimator=NutsEstimator(numWarmup=500,numSamples=500,numChains=4),nGlobalSamples=None,printSummary=True)

moreAbsoluteError=bf.ScoreTest(ScoreFn=AbsoluteErrorFn)
moreSquaredError=bf.ScoreTest(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print()
print("Mean absolute error")
print("-------------------")
print("Training:             ",jnp.mean(trainAbsoluteError))
print("Test, no observations:",jnp.mean(noMCMCAbsoluteError))
print("Test, 3 observations: ",jnp.mean(fewAbsoluteError))
print("Test, 7 observations: ",jnp.mean(moreAbsoluteError))

print()
print("Mean squared error")
print("------------------")
print("Training:             ",jnp.mean(trainSquaredError))
print("Test, no observations:",jnp.mean(noMCMCSquaredError))
print("Test, 3 observations: ",jnp.mean(fewSquaredError))
print("Test, 7 observations: ",jnp.mean(moreSquaredError))