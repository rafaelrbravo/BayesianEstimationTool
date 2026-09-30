import numpyro as npo
npo.set_host_device_count(4)

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.experimental.ode import odeint

# Workaround to run this example directly from the repository without installing BayesianFramework.
# For normal use, install from the repository folder with `python -m pip install -e .` and import with `from BayesianFramework import BayesianFramework`.
# If VS Code does not recognize the import for autocomplete, add `"python.analysis.extraPaths": ["/path/to/BayesianFramework"]` to settings.json.

import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BayesianEstimationTool import BayesianEstimationTool, NutsEstimator


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


# localParams["g"] is the standardized patient-specific deviation.
# Convert it to the physical growth rate using the population mean/std.

def ModelFn(globalParams,localParams,data):
    g=globalParams["gMean"]+globalParams["gStd"]*localParams["g"]
    return {"tumorSize":odeint(ODE,data["v0"],data["times"],g)}


# Population parameters and observation noise.

def GlobalFn(dataShapes):
    return {
        "gMean":npo.sample("gMean",dist.Normal(0.15,0.1)),
        "gStd":npo.sample("gStd",dist.HalfNormal(0.1)),
        "sigma":npo.sample("sigma",dist.HalfNormal(0.1)),
    }


# Training uses every tumor-size measurement.

def TrainLikelihoodFn(globalParams,localParams,data,modelOut):
    npo.sample("obs",dist.Normal(modelOut["tumorSize"],globalParams["sigma"]),obs=data["tumorSize"])


# Test MCMC only uses the first nFit observations.

def TestLikelihoodFn(globalParams,localParams,data,modelOut):
    nFit=3
    tumorSize=modelOut["tumorSize"]
    mask=jnp.arange(tumorSize.shape[-1])<nFit
    npo.sample("obsTest",dist.Normal(tumorSize,globalParams["sigma"]).mask(mask),obs=data["tumorSize"])


# Score one trajectory against all available observations.

def ScoreFn(modelOut,data):
    return jnp.mean(jnp.abs(modelOut["tumorSize"]-data["tumorSize"]))


# ---------------------------------------------------------------------
# Construct framework
# ---------------------------------------------------------------------

bf=BayesianEstimationTool(modelDataFull=data,ModelFn=ModelFn,localParamNames=["g"],GlobalParamFn=GlobalFn,rngSeed=0)

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

bf.Train( trainIndices=jnp.arange(5), TrainLikelihoodFn=TrainLikelihoodFn, Estimator=NutsEstimator(numWarmup=500,numSamples=500,numChains=4), printSummary=True)
trainScores=bf.ScoreTrain(ScoreFn=ScoreFn)


# ---------------------------------------------------------------------
# 2. TESTING WITHOUT MCMC
#
# Do not use any test-patient observations for inference.
#
# Use the median global parameter values from the training posterior.
# For each test patient, sample local parameters from the learned
# population distribution and generate trajectories.
#
# numSamples controls the number of local samples generated.
# ---------------------------------------------------------------------

bf.Test( testIndices=jnp.arange(5,10), TestLikelihoodFn=None, nGlobalSamples=None, nLocalSamples=2000)
noMCMCScores=bf.ScoreTest(ScoreFn=ScoreFn)


# ---------------------------------------------------------------------
# 3. TESTING WITH MCMC
#
# Use the median global parameter values from the training posterior
# and infer each test patient's local parameter using early observations.
#
# TestLikelihoodFn exposes only the first nFit=3 tumor measurements,
# so each test patient's g is updated using its early trajectory.
#
# ScoreFn then compares posterior trajectories against all available
# measurements.
# ---------------------------------------------------------------------

bf.Test( testIndices=jnp.arange(5,10), TestLikelihoodFn=TestLikelihoodFn, Estimator=NutsEstimator(numWarmup=500,numSamples=500,numChains=4), nGlobalSamples=None, printSummary=True)
testMCMCScores=bf.ScoreTest(ScoreFn=ScoreFn)


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

print("Training score:      ",jnp.mean(trainScores))
print("Test score, no MCMC: ",jnp.mean(noMCMCScores))
print("Test score, MCMC:    ",jnp.mean(testMCMCScores))