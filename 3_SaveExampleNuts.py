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
# Generate 10 synthetic patient trajectories
# ---------------------------------------------------------------------

keyG,keyD,keySize,keyNoise=random.split(random.PRNGKey(2),4)

nPatients=10
nTimes=11
times=jnp.linspace(0,10,nTimes)
drugStart=4.0

# Generate correlated patient-specific growth and drug-response effects.

rho=0.8
zG=random.normal(keyG,(nPatients,))
zDIndependent=random.normal(keyD,(nPatients,))
zD=rho*zG+jnp.sqrt(1.0-rho**2)*zDIndependent

# True population:
# mean growth rate       = 0.15
# std growth rate        = 0.03
# mean drug effect       = 0.25
# std drug effect        = 0.05

trueG=0.15+0.03*zG
trueD=0.25+0.05*zD

# Give patients different initial tumor sizes.

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

# Tumor growth model:
#
# before treatment:
# dV/dt = gV
#
# during treatment:
# dV/dt = (g-d)V

def ODE(V,t,g,d,drugStart):
    drugEffect=jnp.where(t>=drugStart,d,0.0)
    return (g-drugEffect)*V


# localParams["g"] and localParams["d"] are standardized
# patient-specific deviations.
#
# Convert them to physical parameter values using the inferred
# population means and standard deviations.

def ModelFn(globalParams,localParams,data):
    g=globalParams["gMean"]+globalParams["gStd"]*localParams["g"]
    d=globalParams["dMean"]+globalParams["dStd"]*localParams["d"]
    tumorSize=odeint(ODE,data["v0"],data["times"],g,d,data["drugStart"])
    return {"tumorSize":tumorSize}


# Population parameters and observation noise.

def GlobalFn(dataShapes):
    return {
        "gMean":npo.sample("gMean",dist.Normal(0.15,0.1)),
        "gStd":npo.sample("gStd",dist.HalfNormal(0.1)),
        "dMean":npo.sample("dMean",dist.Normal(0.25,0.1)),
        "dStd":npo.sample("dStd",dist.HalfNormal(0.1)),
        "sigma":npo.sample("sigma",dist.HalfNormal(0.1)),
    }


# Training uses every tumor-size measurement.

def TrainLikelihoodFn(globalParams,localParams,data,modelOut):
    npo.sample("obs",dist.Normal(modelOut["tumorSize"],globalParams["sigma"]),obs=data["tumorSize"])


# Score one trajectory against all available observations.

def ScoreFn(modelOut,data):
    return jnp.mean(jnp.abs(modelOut["tumorSize"]-data["tumorSize"]))


# ---------------------------------------------------------------------
# Construct framework
# ---------------------------------------------------------------------

bf=BayesianEstimationTool(modelDataFull=data,ModelFn=ModelFn,localParamNames=["g","d"],GlobalParamFn=GlobalFn,rngSeed=0)


# ---------------------------------------------------------------------
# 1. TRAINING
#
# Patients 0-4 train the population model.
#
# Infer:
# - population mean growth rate
# - population std growth rate
# - population mean drug effect
# - population std drug effect
# - correlation between patient-specific g and d
# - patient-specific g and d values
# - observation noise sigma
#
# using all observations from the five training patients.
# ---------------------------------------------------------------------

bf.Train(trainIndices=jnp.arange(5),TrainLikelihoodFn=TrainLikelihoodFn,Estimator=NutsEstimator(numWarmup=500,numSamples=500,numChains=4),printSummary=True)


# ---------------------------------------------------------------------
# 2. SCORE TRAINING POSTERIOR
# ---------------------------------------------------------------------

trainScores=bf.ScoreTrain(ScoreFn=ScoreFn)

print("Training score before save:",jnp.mean(trainScores))


# ---------------------------------------------------------------------
# 3. SAVE
#
# The framework is saved as a NumPy .npz archive.
#
# By default, Save stores:
# - framework metadata
# - training/test run settings, including estimator name and constructor arguments
# - training/test indices
# - RNG state
# - training global posterior samples
# - training Cholesky samples
# - training local posterior samples
# - test locals/globals/cholesky, if testing has been run
#
# Model outputs and the full input dataset are not saved by default.
# ---------------------------------------------------------------------

bf.Save("SimpleTumorGrowthNuts.npz")