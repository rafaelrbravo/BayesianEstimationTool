import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.experimental.ode import odeint

import BayesianEstimationTool as bet


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


def ModelFn(globalParams,localZs,data):
    g=globalParams["gMean"]+globalParams["gStd"]*localZs["g"]
    d=globalParams["dMean"]+globalParams["dStd"]*localZs["d"]
    tumorSize=odeint(ODE,data["v0"],data["times"],g,d,data["drugStart"])
    return {"tumorSize":tumorSize}


def GlobalPriorFn(dataShapes):
    return {
        "gMean":dist.Normal(0.15,0.1),
        "gStd":dist.HalfNormal(0.1),
        "dMean":dist.Normal(0.25,0.1),
        "dStd":dist.HalfNormal(0.1),
        "sigma":dist.HalfNormal(0.1),
    }


# ---------------------------------------------------------------------
# Testing likelihoods
# ---------------------------------------------------------------------

def TestLikelihoodFewFn(globalParams,localZs,data,modelOut):
    nFit=3
    tumorSize=modelOut["tumorSize"]
    mask=jnp.arange(tumorSize.shape[-1])<nFit
    return {"tumorSize":(data["tumorSize"],dist.Normal(tumorSize,globalParams["sigma"]).mask(mask))}


def TestLikelihoodMoreFn(globalParams,localZs,data,modelOut):
    nFit=7
    tumorSize=modelOut["tumorSize"]
    mask=jnp.arange(tumorSize.shape[-1])<nFit
    return {"tumorSize":(data["tumorSize"],dist.Normal(tumorSize,globalParams["sigma"]).mask(mask))}


# ---------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------

def AbsoluteErrorFn(modelOut,data):
    return jnp.mean(jnp.abs(modelOut["tumorSize"]-data["tumorSize"]))


def SquaredErrorFn(modelOut,data):
    return jnp.mean((modelOut["tumorSize"]-data["tumorSize"])**2)


# ---------------------------------------------------------------------
# Load trained Estimator
#
# Functions are supplied explicitly rather than serialized.
# The original model data are also supplied because saveAllData=False
# was used when the framework was saved.
# ---------------------------------------------------------------------

est=bet.Estimator.Load("SimpleTumorGrowthSvi.npz",ModelFn=ModelFn,GlobalPriorFn=GlobalPriorFn,data=data)


# ---------------------------------------------------------------------
# Score training
# ---------------------------------------------------------------------

trainAbsoluteError=est.ScoreTrainSamples(ScoreFn=AbsoluteErrorFn)
trainSquaredError=est.ScoreTrainSamples(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test without a method
#
# Use the median global parameter values from the training posterior.
# No test observations are used for inference.
#
# nLocalZSamples controls the number of local Z samples generated for each
# test patient.
# ---------------------------------------------------------------------

est.Test(testDataIdxs=jnp.arange(5,10),TestLikelihoodFn=None,nGlobalSamples=None,nLocalZSamples=500)

noMCMCAbsoluteError=est.ScoreTestSamples(ScoreFn=AbsoluteErrorFn)
noMCMCSquaredError=est.ScoreTestSamples(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test with 3 observations
#
# Use the median global parameter values from the training posterior.
# SVI updates each test patient's local Zs using the first
# three observations.
# ---------------------------------------------------------------------

est.Test(testDataIdxs=jnp.arange(5,10),TestLikelihoodFn=TestLikelihoodFewFn,Method=bet.Svi,nGlobalSamples=None,printSummary=True)

fewAbsoluteError=est.ScoreTestSamples(ScoreFn=AbsoluteErrorFn)
fewSquaredError=est.ScoreTestSamples(ScoreFn=SquaredErrorFn)


# ---------------------------------------------------------------------
# Test with 7 observations
#
# Repeat the test inference using the first seven observations.
# ---------------------------------------------------------------------

est.Test(testDataIdxs=jnp.arange(5,10),TestLikelihoodFn=TestLikelihoodMoreFn,Method=bet.Svi,nGlobalSamples=None,printSummary=True)

moreAbsoluteError=est.ScoreTestSamples(ScoreFn=AbsoluteErrorFn)
moreSquaredError=est.ScoreTestSamples(ScoreFn=SquaredErrorFn)


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
