import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.experimental.ode import odeint

import BayesianEstimationTool as bet


# ---------------------------------------------------------------------
# Generate 10 synthetic patient trajectories
# ---------------------------------------------------------------------

keyG,keyNoise=random.split(random.PRNGKey(2),2)

nPatients=10
nTimes=8
times=jnp.linspace(0,10,nTimes)

# True population:
# mean growth rate = 0.15
# std growth rate  = 0.03

trueG=0.15+0.03*random.normal(keyG,(nPatients,))

# Give patients slightly different initial tumor sizes.

v0=jnp.linspace(0.8,1.2,nPatients)
trueTumorSize=v0[:,None]*jnp.exp(trueG[:,None]*times)
tumorSize=trueTumorSize+0.05*random.normal(keyNoise,trueTumorSize.shape)

data={
    "times":jnp.tile(times,(nPatients,1)),
    "v0":v0,
    "tumorSize":tumorSize,
}


# ---------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------

# Exponential tumor growth:
# dV/dt = gV

def ODE(V,t,g):
    return g*V


# localZs["g"] is the standardized patient-specific deviation.
# Convert it to the physical growth rate using the population mean/std.

def ModelFn(globalParams,localZs,data):
    g=globalParams["gMean"]+globalParams["gStd"]*localZs["g"]
    return {"tumorSize":odeint(ODE,data["v0"],data["times"],g)}


# Population parameters and observation noise.

def GlobalPriorFn(dataShapes):
    return {
        "gMean":dist.Normal(0.15,0.1),
        "gStd":dist.HalfNormal(0.1),
        "sigma":dist.HalfNormal(0.1),
    }


# Training likelihood for one patient; BET vectorizes it across patients.

def TrainLikelihoodFn(globalParams,localZs,data,modelOut):
    return {"tumorSize":(data["tumorSize"],dist.Normal(modelOut["tumorSize"],globalParams["sigma"]))}


# Test likelihood for one patient; only the first nFit observations are used.

def TestLikelihoodFn(globalParams,localZs,data,modelOut):
    nFit=3
    tumorSize=modelOut["tumorSize"]
    mask=jnp.arange(tumorSize.shape[-1])<nFit
    return {"tumorSize":(data["tumorSize"],dist.Normal(tumorSize,globalParams["sigma"]).mask(mask))}


# Score one trajectory against all available observations.

def ScoreFn(modelOut,data):
    return jnp.mean(jnp.abs(modelOut["tumorSize"]-data["tumorSize"]))


# ---------------------------------------------------------------------
# Construct Estimator
# ---------------------------------------------------------------------

est=bet.Estimator(data=data,ModelFn=ModelFn,localZNames=["g"],GlobalPriorFn=GlobalPriorFn,rngSeed=0)

# Patients 0-4 train the population model.
# Patients 5-9 are completely held out.


# ---------------------------------------------------------------------
# 1. TRAINING
#
# Infer:
# - population mean of g
# - population std of g
# - g for each training patient
# - observation noise sigma
#
# using all observations from the five training patients.
# ---------------------------------------------------------------------

est.Train( trainDataIdxs=jnp.arange(5), TrainLikelihoodFn=TrainLikelihoodFn, Method=bet.Nuts(numWarmup=500,numSamples=500,numChains=4), printSummary=True)
trainScores=est.ScoreTrainSamples(ScoreFn=ScoreFn)


# ---------------------------------------------------------------------
# 2. TESTING WITHOUT INFERENCE
#
# Do not use any test-patient observations for inference.
#
# Use the median global parameter values from the training posterior.
# For each test patient, sample local Zs from the learned
# population distribution and generate trajectories.
#
# numSamples controls the number of local Z samples generated.
# ---------------------------------------------------------------------

est.Test( testDataIdxs=jnp.arange(5,10), TestLikelihoodFn=None, nGlobalSamples=None, nLocalZSamples=2000)
noInferenceScores=est.ScoreTestSamples(ScoreFn=ScoreFn)


# ---------------------------------------------------------------------
# 3. TESTING WITH NUTS
#
# Use the median global parameter values from the training posterior
# and infer each test patient's local Z using early observations.
#
# TestLikelihoodFn exposes only the first nFit=3 tumor measurements,
# so each test patient's g is updated using its early trajectory.
#
# ScoreFn then compares posterior trajectories against all available
# measurements.
# ---------------------------------------------------------------------

est.Test( testDataIdxs=jnp.arange(5,10), TestLikelihoodFn=TestLikelihoodFn, Method=bet.Nuts, nGlobalSamples=None, printSummary=True)
testNutsScores=est.ScoreTestSamples(ScoreFn=ScoreFn)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print("Training score:      ",jnp.mean(trainScores))
print("Test score, no inference: ",jnp.mean(noInferenceScores))
print("Test score, NUTS:    ",jnp.mean(testNutsScores))
