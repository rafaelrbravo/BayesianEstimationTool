# BayesianEstimationTool Manual

BayesianEstimationTool (BET) helps you learn from a group of related individuals and make predictions for someone new. An *individual* (also called an *entry* in some API descriptions) might be a patient, an experiment, or a simulated system. This manual uses *individual* for all three; the examples happen to use people. You provide a model of how the system works and the measurements you have collected; BET estimates its unknown parameters and helps you test its predictions.

Imagine measuring how quickly tumors grow in several patients. The same growth equation might describe every patient, but each patient's growth rate can differ. If you estimate each rate separately, a patient with only one or two measurements gives you little information. BET instead learns from the whole group: what growth rates are typical, how much they vary, and what the measurements tell us about each patient.

You can then predict a new patient's trajectory in two ways. **Before seeing any measurements**, use what the group taught you. **After seeing a few measurements**, update the prediction to reflect what you have learned about that patient. BET also helps you measure whether those predictions were accurate and how uncertain they are.

The statistical name for learning about individuals and their shared population in this way is *hierarchical Bayesian inference*. You do not need to master that terminology before using the basic workflow. Throughout the manual, *population* means the group whose shared patterns BET learns; it does not require a medical or demographic population.

## The workflow at a glance

1. **Organize observations.** Put each individual's measurements and inputs into arrays.
2. **Describe the system.** Write a function that predicts what happens when its parameters are known.
3. **Describe what is unknown.** Specify which parameters are shared, which vary by individual, and how measurements differ from ideal predictions.
4. **Learn from the group.** Use `Train` to estimate the population and individual parameters.
5. **Predict for new individuals.** Use `Test` with or without their available measurements.
6. **Check the predictions.** Use scoring functions to compare predictions with observations held back from fitting.

## Core components

| Component | What it does |
| --- | --- |
| `data` | Holds measurements and inputs for each individual |
| `ModelFn` | Predicts what happens for one individual |
| `GlobalPriorFn` | Describes plausible values for unknown shared parameters before fitting |
| `localZNames` | Names the ways individuals can differ from one another |
| `TrainLikelihoodFn` | Describes how observed measurements relate to model predictions |
| `Estimator` | Connects your model, data, and inference results |
| `Train` / `Test` | Learns from the training group / predicts for new individuals |
| `ScoreFn` | Measures how well a prediction matches observations |

## Part 1: Core concepts

### What BET learns

**A model is a rule for making predictions.** For example, a growth equation uses a growth rate and a time to predict a size. BET does not choose that equation for you; you write it in `ModelFn`. BET helps determine which parameter values are consistent with your measurements.

**Some parameters describe the group; others describe individuals.** A population might have a typical growth rate, while each patient grows somewhat faster or slower. BET learns about both the shared pattern and the individual differences. In the API, shared unknown parameters are called *global parameters*. Individual differences are represented by *local effects*.

**Measurements are imperfect.** Even a good model will not match every observation exactly. The *likelihood* describes how observations can differ from model predictions, including measurement noise. A *prior* describes plausible parameter values before those observations are used. BET combines the prior information with the measurements to estimate what values are plausible afterward; these updated estimates form the *posterior*.

**Learning and predicting are different steps.** During `Train`, BET uses the selected individuals to learn the population and their individual effects. During `Test`, it predicts for individuals not included in training. Without their measurements, BET draws possible individual effects from the learned population. With some measurements and a test likelihood, BET updates those effects for each new individual. To assess prediction quality fairly, keep the observations you want to predict out of the fitting step.

**BET tracks uncertainty rather than returning only one answer.** Different plausible parameter values can produce different predictions. Samples from the fitted distribution let you examine that variation. Predictions of the underlying model are not automatically noisy future measurements; the observation model describes that additional variation.

### Designing a BET model

The following steps translate those ideas into the API:

1. Prepare `data` as a dictionary of arrays. Every array's first axis identifies the individual; later axes can represent times, features, or positions.
2. Write `ModelFn(globalParams, localZs, data)` to predict outputs for **one individual**. Return a dictionary of named outputs. Keep this function deterministic: do not sample random variables inside it.
3. If shared parameters are unknown, define `GlobalPriorFn(dataShapes)` to give their prior distributions. Otherwise, omit it.
4. Use `localZNames` for the individual differences you want BET to estimate. Inside `ModelFn`, transform these standardized effects into meaningful parameters such as growth rates.
5. Define `TrainLikelihoodFn` to connect observations to predictions using distributions. If measurements are missing, mask them explicitly.
6. Choose training individuals and an inference method, `Nuts(...)` or `Svi(...)`. Call `Train` and check whether the fitted model describes the data reasonably.
7. Choose held-out individuals. Call `Test` for predictions from the population alone, or provide a test likelihood and inference method to update individual effects using measurements available at prediction time.
8. Define a scoring function and call `ScoreTrainSamples` or `ScoreTestSamples` to evaluate predictions. Keep track of which dimensions represent samples and which represent individuals.

The next part puts these pieces together in a complete growth example. Later sections explain the technical details, including how BET represents correlated individual effects and how to configure inference methods.

## Part 2: Worked example

### Predicting growth for someone new

Suppose we measure a growing quantity eight times for each of ten people. We want to learn how growth rates differ across people, then predict the last five measurements for four people the model has never seen. We will first predict using only what we learned from the other six people, then see whether the first three measurements from each new person improve the predictions.

We generate example data ourselves, so you can follow the workflow without downloading a dataset. The growth equation is deliberately simple; the same approach works with more detailed scientific models.

#### Create example measurements and a growth model

```python
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from BayesianEstimationTool import Estimator, Svi

n_people = 10
times = jnp.linspace(0.0, 7.0, 8)
key_g, key_noise = random.split(random.PRNGKey(42))
true_g = 0.15 + 0.025 * random.normal(key_g, (n_people,))
true_y = jnp.exp(true_g[:, None] * times[None, :])
y = true_y + 0.05 * random.normal(key_noise, true_y.shape)
data = {"time": jnp.tile(times, (n_people, 1)), "y": y}

def global_prior(shapes):
    return {
        "g_mean": dist.Normal(0.15, 0.10),
        "g_sd": dist.HalfNormal(0.10),
        "noise": dist.HalfNormal(0.10),
    }

def model_fn(global_params, local_zs, data):
    g = global_params["g_mean"] + global_params["g_sd"] * local_zs["g"]
    return {"prediction": jnp.exp(g * data["time"])}

def train_likelihood(global_params, local_zs, data, model_out):
    return {"y": (data["y"], dist.Normal(model_out["prediction"], global_params["noise"]))}
```

The model has three unknown population-level quantities. `g_mean` describes a typical growth rate; `g_sd` describes how much rates differ between people; and `noise` describes random variation in measurements. The individual value `local_zs["g"]` says where one person falls relative to the typical rate. The model converts it to that person's growth rate using `g_mean + g_sd * z`.

`global_prior` describes plausible starting ranges for the three unknown quantities. `train_likelihood` says that each measurement is expected to be near the model's prediction, with variation controlled by `noise`. The data generator uses known values only to create this synthetic example; BET does not see those values when fitting.

You can run the Python blocks in this worked example in order, in one Python session. Part 5 reuses the resulting `est`, `before`, and `after` variables rather than repeating the fits. The other code blocks in the manual illustrate separate features and are not all intended to run as one script.

#### Train on six individuals

```python
est = Estimator(data, model_fn, localZNames=["g"],
                GlobalPriorFn=global_prior, rngSeed=123)
train_ids = list(range(6))
test_ids = list(range(6, 10))

est.Train(train_ids, train_likelihood,
          Svi(numSteps=1500, numSamples=300, progressBar=False))
```

The first six people form the **training group** (the individuals used to fit the model). BET uses their measurements to estimate typical growth, variation between people, and measurement noise. The other four form the **test group** (new individuals): their measurements are not used during training.

Here `Svi` is the inference method that estimates the unknown quantities. These settings are illustrative, not a guarantee of accurate inference. For research, check the fitted predictions and sensitivity to modeling choices. The later personalized-prediction step repeats inference for 20 sampled population states, so it may take time; use fewer states and steps for an initial run.

#### Predict using the group alone

To see how well our predictions work, we calculate the average absolute difference between predictions and measurements for the **last five time points** (indices 3 through 7). These measurements are available for evaluation, but we must not use them to personalize the predictions.

```python
def future_mae(model_out, data):
    return jnp.mean(jnp.abs(model_out["prediction"][3:] - data["y"][3:]))

est.Test(test_ids, nGlobalSamples=20, nLocalZSamples=100)
before = est.ScoreTestSamples(future_mae)
print("Population-only mean future MAE:", float(jnp.mean(before)))
assert before.shape == (20, 100, 4)
```

BET considers 20 plausible versions of the learned population and 100 possible growth-rate differences for each one. Because it has not used any test person's measurements to estimate their growth rate, these are **population-only predictions**. The score array has shape `[20, 100, 4]`: population draw, individual-effect draw, and test person. Each number measures a possible prediction's error.

#### Personalize predictions using three measurements

Now suppose we have received the first three measurements for each test person. We can use those observations to estimate their growth rate more specifically. A **mask** marks which measurements BET is allowed to use for that update. The last five remain hidden from inference, even though we retain them to check the predictions afterward.

```python
def test_likelihood(global_params, local_zs, data, model_out):
    observed = jnp.arange(data["y"].shape[0]) < 3
    prediction = model_out["prediction"]
    safe_y = jnp.where(observed, data["y"], prediction)
    return {
        "y": (safe_y, dist.Normal(prediction, global_params["noise"]).mask(observed))
    }

est.Test(test_ids, TestLikelihoodFn=test_likelihood,
         Method=Svi(numSteps=800, numSamples=100, progressBar=False),
         nGlobalSamples=20)
after = est.ScoreTestSamples(future_mae)
print("Conditioned mean future MAE:", float(jnp.mean(after)))
assert after.shape == (20, 100, 4)
```

BET updates each test person's growth-rate estimate using their first three measurements and the population information learned earlier. We still score only the last five measurements, which BET did not use for the update. Calling `Test` again replaces the previous test state inside `est`, but the earlier `before` scores remain in their own variable.

#### What did the extra measurements buy us?

Both `before` and `after` have shape `[20, 100, 4]`. Their means summarize errors across plausible population parameters, individual effects, and test people. If the `after` mean is lower, those three early measurements helped on average. With only ten synthetic people, an improvement is not guaranteed.

To examine uncertainty in trajectories rather than errors, use a score function returning `model_out["prediction"]` and summarize its posterior draw axes. The resulting intervals describe latent mean trajectories; they do not automatically include new measurement noise. For stronger evaluation, repeat across random seeds and vary the number of early observations.

The same **Train → Test → Score** workflow applies to more complicated models, including models defined by differential equations.

**Checking the example:** The two `assert` statements verify the expected score dimensions; if either fails, the script stops rather than silently treating the axes incorrectly. The printed MAE values will depend on the inferred parameters and need not match a fixed number. For a quick first run, reduce `nGlobalSamples` in both `Test` calls to the same smaller value, and adjust the assertions accordingly. That changes the number of population draws, not the underlying example. Part 5 revisits scoring in more detail and adds a check on the training group.

## Part 3: Building blocks and individual differences

The worked example used one growth rate per individual. This part explains how its pieces fit together, and how BET can learn relationships between several individual characteristics.

### What each function receives

`Estimator(data, ModelFn, localZNames=None, GlobalPriorFn=None, lkjConcentration=2.0, rngSeed=None)` connects the model and its data. The `data` argument is a dictionary of arrays. The first axis of each data array indexes individuals; all arrays have the same number of individuals. Remaining axes can represent times, features, or spatial coordinates.

`ModelFn(globalParams, localZs, data)` receives global parameters, one individual's local latent values, and that individual's data as dictionaries. It returns a dictionary of model outputs. BET vectorizes the function across individuals and posterior draws, so avoid Python side effects and converting traced JAX values to NumPy.

`GlobalPriorFn(dataShapes)` describes plausible values of shared unknown parameters before seeing the measurements. It returns a dictionary mapping parameter names to NumPyro distributions. The `dataShapes` argument describes each data array **for one individual**, without the leading individual axis.

For example, if `data["y"]` contains measurements from 10 people at 8 times, its full shape is `(10, 8)`, but `dataShapes["y"]` is `(8,)`. The first dimension selects the person; the remaining dimensions describe that person's data. BET removes only the first dimension when constructing `dataShapes`. The global parameters are shared by the training population.

`TrainLikelihoodFn(globalParams, localZs, data, modelOut)` and `TestLikelihoodFn` return dictionaries mapping site names to `(observed_value, distribution)` tuples. A site may contain a scalar or a vector of observations. Its observation, distribution, and mask shapes must agree.

`ScoreFn(modelOut, data)` returns a scalar or array for one individual; BET vectorizes it across posterior samples and individuals.

### Representing individual differences

The example so far has one individual-specific parameter: growth rate. BET can also represent several differences at once, and can learn whether they tend to occur together.

BET needs a consistent way to describe how each individual differs from the population. It uses **local z-scores**: numbers that describe an individual's position relative to a typical value. Local z-scores represent individual-to-individual differences on a standardized scale. For example, `z=0` gives the model's central growth rate; positive or negative values change it according to the transformation you define. With `rate = exp(log_rate_mean + log_rate_sd * z)`, `log_rate_mean` sets the center on the log scale and `log_rate_sd` sets its spread. Before conditioning on individual observations, each z-score has a standard-normal marginal distribution.

Extend the tumor-growth example by adding treatment. Each patient now has two parameters: a **growth rate**, describing how quickly the tumor grows without treatment, and a **treatment-response parameter**, describing how strongly treatment slows that growth. Both may vary between patients. For instance, patients whose tumors grow faster might also tend to respond more strongly to treatment, or the relationship might be reversed. Assuming the parameters vary independently would miss that population-level relationship.

Set `localZNames=["growth", "response"]` to represent both sources of patient-to-patient variation as standardized local effects. BET can learn their **correlation** from the training patients. A positive correlation means patients with above-average growth effects tend also to have above-average response effects; a negative correlation means higher growth effects tend to accompany lower response effects. These are correlations between the standardized effects, not necessarily between the parameters after arbitrary transformations.

BET represents this relationship using a **correlation matrix**. Internally, it uses a **Cholesky factor** of that matrix to generate related individual effects. A correlation matrix specifies how pairs of effects vary together; its Cholesky factor is a triangular matrix that can transform independent standard-normal draws into correlated ones. BET places an LKJ prior on the correlation structure, with `lkjConcentration` controlling how strongly the prior favors correlations near zero. The inferred relationship can improve predictions for a new patient: measurements that inform their growth rate can also inform their likely treatment response, even if response measurements are limited.

The model below illustrates the parameter transformation. It is a schematic treatment model, not a validated description of tumor dynamics:

```python
def model_fn(global_params, local_zs, data):
    growth = jnp.exp(global_params["log_growth_mean"] + global_params["log_growth_sd"] * local_zs["growth"])
    response = jnp.exp(global_params["log_response_mean"] + global_params["log_response_sd"] * local_zs["response"])
    net_rate = growth - response * data["treatment"]
    return {"prediction": data["baseline"] * jnp.exp(net_rate * data["time"])}
```

Define population means, scales, and transformations explicitly; BET does not create them automatically for each local variable.

## Part 4: How BET learns from data

Once you have defined a model and its observations, BET must find plausible parameter values. It offers two built-in ways to do this. Both produce samples of possible parameter values, but they differ in how they obtain those samples and how much computation they require.

### Choosing between NUTS and SVI

**NUTS** explores plausible parameter values by drawing samples through a Markov chain. It is often useful when the shape and spread of uncertainty matter. `Nuts(numWarmup=2000, numSamples=2000, numChains=4, acceptProb=0.95, denseMass=False, medianSamples=50)` uses NumPyro's NUTS MCMC kernel with median-based initialization. `numWarmup` is the adaptation period; `numSamples` is the retained number of draws **per chain**; `numChains` controls chain count; `acceptProb` is the target acceptance probability; `denseMass` enables dense mass-matrix adaptation; and `medianSamples` controls the initialization strategy. BET combines the chains into one leading draw axis for downstream scoring, so a run with four chains and 2,000 retained draws per chain provides up to 8,000 training posterior draws before any subsampling. Its reported summary includes effective sample size and R-hat. Inspect divergences, mixing, and posterior predictive adequacy rather than treating a successful run as validation.

**SVI** takes a different approach: it fits a simpler probability distribution to approximate the range of plausible parameter values. It can be faster, although the approximation may miss some uncertainty. `Svi(numSteps=10000, numSamples=2000, stepSize=0.01, numParticles=1, progressBar=True, medianSamples=50, multivariate=False)` optimizes a variational approximation with Adam and an ELBO objective. `numSteps` is the number of optimization updates; `numSamples` controls approximate posterior draws generated after optimization; `stepSize` is Adam's learning rate; `numParticles` controls Monte Carlo particles for the ELBO; and `medianSamples` controls initialization. `multivariate=False` uses `AutoNormal`; `True` uses `AutoMultivariateNormal`, which can represent posterior covariance but may be expensive in large hierarchical models.

**Extending inference methods.** NUTS and SVI are the built-in choices, not a closed list. Additional inference methods can be integrated through BET's Method interface; see **Adding a custom inference method** in Part 7 and `ValidateMethod`.

**Method selection.** NUTS is often preferable when reliable posterior uncertainty is central and runtime is manageable. SVI can be faster for large problems, but an apparently stable loss curve does not establish calibrated uncertainty. Compare posterior predictive behavior and uncertainty against NUTS or simulated ground truth when feasible. BET supports configured Method instances; using Method classes can reconstruct previously saved configuration when available.

## Part 5: Checking predictions

A model can fit the people it learned from and still predict poorly for someone new. **Scoring** means comparing predictions with observed outcomes. Part 2 compared population-only and personalized predictions for new individuals; here we add a training-set check and show how to interpret all three scores.

This section continues with the same `est`, `test_ids`, and `future_mae` from Part 2. The first three test measurements may be used for personalization; the last five are reserved for evaluation.

### What the score means

We use **mean absolute error (MAE)**: the average size of the differences between predicted and observed measurements. Zero means a perfect match; smaller is better. The `future_mae` function defined in Part 2 computes this for the last five time points. BET evaluates it separately for each individual and each sampled set of parameters.

### 1. Check fit on the training individuals

After calling `Train`, score the training individuals:

```python
train_scores = est.ScoreTrainSamples(future_mae)
print("Training MAE:", float(jnp.mean(train_scores)))
```

This measures how closely the fitted model describes the training trajectories. It is a useful fit check, but it is not an unbiased estimate of performance on new individuals: these observations contributed to fitting the model.

### Compare predictions for new individuals

Part 2 already computed two sets of test scores. The `before` scores use only the learned population: no measurements from the test individuals were used to estimate their growth rates. The `after` scores come from predictions updated using each test individual's first three measurements. In both cases, the score evaluates the final five measurements, which were held out from individual calibration.

```python
print("Test MAE without calibration:", float(jnp.mean(before)))
print("Test MAE with calibration:", float(jnp.mean(after)))
```

Comparing these values shows whether early measurements improved predictions on average. Improvement is not guaranteed, particularly with a small test group.

### Compare the results and their uncertainty

The **test** scores answer the main prediction question. The training score is useful for diagnosing fit, but should not be used as a substitute for either test score.

- **Training MAE** describes fit to individuals used to learn the model. Do not treat it as directly comparable to held-out test MAE.
- **Test MAE without calibration** measures predictions based on the population alone.
- **Test MAE with calibration** measures predictions after incorporating each individual's available measurements.

A score is computed once for each sampled prediction, not just once for the entire test group. This is why scoring produces an array rather than a single number: you can average scores for a quick comparison or inspect how they vary across individuals and plausible parameter values.

The score arrays retain variation across posterior draws and individuals. For these calls, `ScoreTrainSamples` returns an array shaped like `[posterior_draw, training_individual]`; `ScoreTestSamples` returns `[global_draw, local_draw, test_individual]` when the score is scalar per individual. Take means or quantiles over the intended axes rather than flattening them without thought. The example prints a grand mean for a simple summary; uncertainty can be examined by retaining and summarizing the draw-level scores.

BET's predictive uncertainty has multiple sources. Sampling global parameter and Cholesky states with positive `nGlobalSamples` propagates uncertainty in the learned population; sampling or inferring local effects represents differences between new individuals. The observation likelihood describes measurement noise, which is not automatically included in deterministic model outputs. Model-structure uncertainty is not automatically represented.

For reliable evaluation, keep test individuals out of training and ensure that only measurements available at the prediction time enter the test likelihood. Score later, held-out outcomes separately. The current `Test` implementation requires at least one named local Z, even though `Estimator` and `Train` can be used without local Zs.

### Return values and estimator state

The estimator retains the most recent training and test results, so later calls can reuse their parameter samples. A new call to `Train` clears the previous test state; a new call to `Test` replaces it. The score arrays already assigned to variables remain available.

- `Train(...)` returns the inference method's output object: a NumPyro MCMC object for NUTS or an SVI result for SVI. BET also stores the posterior samples needed for later testing and scoring.
- `Test(...)` without a likelihood generates population predictions and returns `None`. With a test likelihood, it returns a list containing one inference output per selected global posterior draw. With `nGlobalSamples=None`, BET uses one state assembled from the componentwise medians of the saved global posterior draws (and, when present, the Cholesky draws). This is a convenient point estimate, not a randomly sampled joint posterior state.
- `ScoreTrainSamples(ScoreFn, nSamples=None, rngSeed=None)` returns `[posterior_draw, training_individual, ...score_shape]`. For NUTS, this first axis combines draws across chains; for SVI, it contains draws from the fitted approximation.
- `ScoreTestSamples(ScoreFn, nSamples=None, rngSeed=None)` returns `[global_draw, local_draw, test_individual, ...score_shape]`. Here `nSamples`, when provided, selects the number of **local** draws scored for each global state; it does not change the number of global states selected by `Test`.

## Part 6: Working with incomplete data and numerical models

Real datasets are rarely perfectly rectangular. One individual may have ten measurements while another has only three, and some measurements may be missing. BET works with arrays, so these differences need to be represented explicitly without pretending missing values were observed.

### Different numbers of measurements

Use masks when entries have different observation counts, measurements are missing, or testing should reveal only part of a trajectory. A mask distinguishes **positions stored in an array** from **observations contributing to inference**. Held-out measurements can remain in the data for scoring while being excluded from the test likelihood.

Pad each time series to a common length `T`. Store `time`, `y`, and a boolean `valid` mask as `[N, T]` arrays; padded positions are placeholders, not observations. Repeating the last valid time and measurement can be safer than inserting zeros when a solver or transformation cannot accept zero. Choose padding to satisfy the model's and solver's time-ordering requirements, and exclude padded positions from the likelihood.

```python
data = {
    "time": jnp.array([[0., 1., 2.], [0., 3., 3.]]),
    "y": jnp.array([[1.0, 1.2, 1.4], [0.9, 1.1, 1.1]]),
    "valid": jnp.array([[True, True, True], [True, True, False]]),
}
```

Use a masked NumPyro distribution and replace invalid observed values with finite, safe values before log-probability evaluation:

```python
def likelihood(global_params, local_zs, data, model_out):
    valid = data["valid"]
    pred = model_out["prediction"]
    safe_y = jnp.where(valid, data["y"], pred)
    return {"y": (safe_y, dist.Normal(pred, 0.2).mask(valid))}
```

The model must produce finite predictions even at padded positions. Masking cannot make an intermediate `NaN` or infinity safe; multiplying a divergent log-probability by zero does not guarantee a finite gradient.

For partial-observation testing, use a separate mask such as `test_observed = valid & (time <= cutoff)`. Apply it in the test likelihood, but retain the original validity mask for scoring held-out observations. Ensure hidden measurements cannot enter the model through inputs or preprocessing.


#### Keeping calculations valid when data are missing

Scale observations and model outputs to reasonable ranges. Set likelihood scales according to measurement noise rather than choosing arbitrarily small values: large residuals or near-zero noise can destabilize log-probabilities and gradients. Replace missing observations with finite placeholders before masking, and check that model outputs remain finite at padded positions. Diagnose shape and gradient errors with small synthetic examples, `jax.debug.print`, and `jax.numpy.isfinite`.

Before population fitting, test the model and likelihood on one individual, then verify axes and broadcasting across individuals. Use posterior predictive checks: successful NUTS or SVI execution does not establish scientific adequacy.

### Models defined by differential equations

Some scientific models describe *how fast a quantity changes* rather than giving its value directly. A differential equation solver turns those rates of change into a predicted trajectory. `ModelFn` can call a JAX-compatible ODE solver such as `jax.experimental.ode.odeint`. Use JAX arrays and differentiable operations, and return outputs at observation times. In the workflows tested for this project, `odeint` has worked more reliably with BET's batching than Diffrax. Compatibility is model-dependent, so verify gradients and numerical behavior. Diffrax is an optional external dependency, not part of BET's installation requirements.

```python
from jax.experimental.ode import odeint

def rhs(state, time, rate):
    return rate * state

def model_fn(global_params, local_zs, data):
    rate = jnp.exp(global_params["log_rate"] + local_zs["rate"])
    times = data["time"]  # sorted, finite, compatible with odeint
    trajectory = odeint(rhs, data["initial"], times, rate)
    return {"prediction": trajectory}
```

Padded time grids must satisfy the solver's ordering requirements. One approach is to solve on a shared regular grid and interpolate to each individual's observation times. Check for duplicate or unsorted padded times.

## Part 7: Saving your work and extending BET

Once you have trained a model, you may want to return to it later without running inference again. BET can save the results of training and testing, along with enough configuration to reconstruct an estimator. You can also add an inference algorithm beyond the two built-in choices if your project needs one.

### Saving and loading

`Save` preserves the fitted population posterior, training/test configuration, selected entry indices, and RNG state in a compressed `.npz` file. You can also choose whether to retain individual local-effect samples, model outputs, and the original data.

```python
est.Save("fit.npz", saveAllData=True)
restored = Estimator.Load("fit.npz", ModelFn=model_fn, GlobalPriorFn=global_prior)
```

The full signature is:

```python
Save(fileName, saveTrainLocalZs=True, saveTestLocalZs=True,
     saveTrainModelOut=False, saveTestModelOut=False, saveAllData=False)
```

| Option | What it controls |
| --- | --- |
| `fileName` | Destination for the compressed `.npz` file. If `None`, return an in-memory nested record instead of writing a file. |
| `saveTrainLocalZs` | Include training individuals' posterior local-effect samples. Enabled by default. |
| `saveTestLocalZs` | Include the latest test individuals' local-effect samples and their associated sampled global/Cholesky states. Enabled by default. |
| `saveTrainModelOut` | Include previously retained training model outputs, if available. Disabled by default. |
| `saveTestModelOut` | Include previously retained test model outputs, if available. Disabled by default. |
| `saveAllData` | Include the original input data, allowing loading without supplying it separately. Disabled by default. |

Population posterior samples and saved Train/Test configuration are retained independently of these options. Omitting local samples or model outputs reduces file size but may limit subsequent scoring or require recomputation. In particular, to score a loaded training fit with local effects, retain training local draws or previously saved model outputs; to score loaded test predictions, retain test local draws or previously saved test model outputs. The latter outputs must have been requested during `Test` and included during `Save`. The model-output options only save outputs already retained by `Train(..., saveModelOut=True)` or `Test(..., saveModelOut=True)`; they do not generate missing outputs at save time.

To restore an estimator:

```python
restored = Estimator.Load("fit.npz", ModelFn=model_fn,
                          GlobalPriorFn=global_prior, data=data)
```

`Load(fileName, ModelFn, GlobalPriorFn=None, data=None)` reconstructs the saved estimator, including available posterior samples and configuration. Omit `data` if the file was saved with `saveAllData=True`; otherwise supply the original data. You must supply the model function and, when used, the global-prior function again because Python functions are not serialized. BET warns when their names differ from the saved names, but a matching name does not guarantee identical behavior.

For inspection without reconstructing an estimator, call `Estimator.Load("fit.npz", ModelFn=None)` to obtain the saved record as a dictionary. Saved Method names and constructor arguments support reconstruction of inference configuration; they do not serialize live NUTS or SVI engine objects.

### Adding a custom inference method

Most users can use NUTS or SVI without implementing anything new. This section is for developers who want BET to run a different inference algorithm while keeping its existing training, testing, and scoring workflows.

BET separates inference algorithms from its population and individual workflows. NUTS and SVI are built in; to integrate another backend, implement the minimum Method interface and validate it before fitting.

A configured Method instance must implement:

- `_Fit(ModelFn, rngKey, samplePrefixes)`: run inference and return a dictionary with `"output"` (backend-specific results or diagnostics) and `"samples"` (posterior or explicitly approximate-posterior draws).
- `_SaveArgs()`: return a dictionary of constructor keyword arguments so BET can reconstruct the Method from saved configuration.

The `samples` dictionary maps NumPyro sample-site names to arrays with **draws along the first axis**. Retain only sites whose names start with one of `samplePrefixes`; BET uses these names to identify global parameters, local effects, Cholesky factors, and optionally model outputs.

This minimal skeleton illustrates the interface:

```python
class MyMethod:
    _printMode = None

    def __init__(self, nSamples=100):
        self.nSamples = nSamples

    def _Fit(self, ModelFn, rngKey, samplePrefixes):
        # Replace with an actual inference backend.
        rawSamples, backendOutput = run_my_inference(ModelFn, rngKey, self.nSamples)
        samples = {name: value for name, value in rawSamples.items()
                   if any(name.startswith(prefix) for prefix in samplePrefixes)}
        return {"output": backendOutput, "samples": samples}

    def _SaveArgs(self):
        return {"nSamples": self.nSamples}
```

`run_my_inference` is a placeholder, **not a BET function**. The backend must produce meaningful posterior or clearly identified approximate-posterior draws. Repeating a maximum-a-posteriori (MAP) point does not measure posterior uncertainty.

Validate the Method's structure before using it:

```python
import BayesianEstimationTool as bet

method = MyMethod(nSamples=100)
bet.ValidateMethod(method)
```

`ValidateMethod` checks required callable signatures, saved constructor arguments, and printing hooks. It **does not execute `_Fit`**, inspect returned samples or shapes, or establish statistical correctness. Test those separately on small synthetic models and with posterior predictive checks.

Optional features include:

- **Diagnostics and printing:** Set `_printMode` to `"before"`, `"after"`, or `"replace"`, and implement `_Print(output, localNames, globalNames, title=None)`. With `_printMode=None`, BET uses its generic sample summary.
- **Configuration:** Accept options for initialization, optimization, sampling, and progress reporting, and return reproducible constructor options from `_SaveArgs()`.
- **Persistence:** Keep configuration reconstructible through constructor keyword arguments. The backend `output` is returned to the caller but is not automatically serialized as a live inference-engine session.
- **Inference quality:** Compare posterior summaries, predictive distributions, and computational costs against NUTS on a known model. This is behavioral testing, not structural validation.

See `NutsMethod.py` and `SviMethod.py` for complete Method implementations.

## Part 8: Troubleshooting and API reference

If a result looks wrong or a call fails, start with the practical checks below. The API reference that follows lists exact function signatures and options; you can consult it as needed rather than learning every argument before trying the worked example.

### Common mistakes and troubleshooting

**Shape mismatch.** Start with one entry and print array shapes before tracing. All top-level data arrays must have the same entry count, and per-entry model outputs must match likelihood observation shapes. Broadcasting can silently produce an incorrect likelihood, so verify dimensions explicitly.

**Invalid entry indices.** Training and test indices must be nonempty, one-dimensional collections of unique integers within the data's entry range. BET rejects duplicate, negative, and out-of-range indices; it does not silently remove duplicates or interpret negative indices as Python indexing from the end.

**Local z-scores mistaken for physical parameters.** A local z-score has a standard-normal marginal prior and is not measured in units of growth rate, drug sensitivity, or observation noise. Specify each transformation to physical units explicitly, and document the resulting units.

**Hidden test outcomes leaking into inference.** Keeping the full `y` array for scoring is safe only if the test likelihood masks unavailable outcomes and the mechanistic model does not read them. Check preprocessing, features, and masks to prevent future measurements from influencing predictions.

**Missing data and NaNs.** Masked observations still require finite values at points where JAX evaluates intermediate expressions. Use safe finite placeholders and keep ODE solver time grids valid even for padded observations.

**Unexpectedly narrow intervals.** With `nGlobalSamples=None`, BET fixes population parameters at posterior medians and does not propagate their uncertainty. Use a positive `nGlobalSamples` to include it. Approximate SVI posteriors may also underestimate uncertainty.

**Poor convergence.** Check observation scaling, priors, parameter identifiability, and solver gradients. For NUTS, inspect mixing and convergence diagnostics; for SVI, inspect the loss trajectory and sensitivity to initialization.

**Save/Load mismatch.** Restore compatible data shapes, Method configuration, model function, and global prior function. Python functions are not serialized; stored names are metadata, not executable code.

### Detailed public API reference

The preceding sections explain *why* you would use each operation. This reference records *how* to call it and what each option means.

The signatures below cover the public exports `Estimator`, `Nuts`, `Svi`, and `ValidateMethod`, together with public `Estimator` methods.

#### Estimator construction

`Estimator(data, ModelFn, localZNames=None, GlobalPriorFn=None, lkjConcentration=2.0, rngSeed=None)`

- `data`: dictionary of arrays sharing a leading entry axis.
- `ModelFn`: deterministic per-entry model function returning a dictionary.
- `localZNames`: optional list of named standardized local effects.
- `GlobalPriorFn`: optional function returning named NumPyro distributions for shared parameters.
- `lkjConcentration`: concentration of the LKJ correlation prior when multiple local effects are used.
- `rngSeed`: optional reproducibility seed.

#### Train

`Train(trainDataIdxs, TrainLikelihoodFn, Method, printSummary=False, rngSeed=None, saveLocalZs=True, saveModelOut=False)`

- `trainDataIdxs`: selected training entries.
- `TrainLikelihoodFn`: per-entry likelihood factory.
- `Method`: inference Method instance or supported Method class.
- `printSummary`: request inference summary output.
- `rngSeed`: optional call-specific seed.
- `saveLocalZs`: retain local posterior draws for later use.
- `saveModelOut`: retain generated model outputs rather than recomputing them.

#### Test

`Test(testDataIdxs, TestLikelihoodFn=None, Method=None, nGlobalSamples=None, nLocalZSamples=2000, printSummary=False, rngSeed=None, saveModelOut=False)`

- `testDataIdxs`: entries for population prediction or local updating.
- `TestLikelihoodFn`: omitted for unconditioned population prediction; provided for observation-conditioned inference.
- `Method`: inference Method for observation-conditioned testing; may reuse saved configuration.
- `nGlobalSamples`: `None` uses posterior median population state; a positive integer selects trained posterior states.
- `nLocalZSamples`: number of local draws in no-likelihood testing.
- `printSummary`, `rngSeed`, `saveModelOut`: reporting, reproducibility, and output retention.

#### Scoring

`ScoreTrainSamples(ScoreFn, nSamples=None, rngSeed=None)` scores training posterior predictions. Its output axes are posterior draw, training entry, then any score dimensions. If the model has local Zs, training must retain either local posterior draws (`saveLocalZs=True`, the default) or model outputs (`saveModelOut=True`); otherwise training scores cannot be reconstructed.

`ScoreTestSamples(ScoreFn, nSamples=None, rngSeed=None)` scores the most recent test state. Its output axes are global draw, local draw, test entry, then any score dimensions. Check what each axis represents before aggregating uncertainty.

**Models without local effects:** `Estimator` and `Train` support models without local Zs, but the current `Test` implementation rejects them (`Test() requires at least one local Z`). Setting `saveModelOut=True` does not bypass this restriction. Consequently, `ScoreTestSamples` cannot be used for such models through the `Test` workflow.

#### Persistence

`Save(fileName, saveTrainLocalZs=True, saveTestLocalZs=True, saveTrainModelOut=False, saveTestModelOut=False, saveAllData=False)` writes a compressed state file; with `fileName=None`, it returns a record. The flags control whether local draws, model outputs, and data are included. Saving data improves portability but increases file size and may create data-sensitivity obligations.

`Estimator.Load(fileName, ModelFn, GlobalPriorFn=None, data=None)` reconstructs an estimator using the supplied functions. Provide `data` if it was not saved. Passing `ModelFn=None` returns the saved record without reconstructing an executable estimator.

`ValidateMethod(method)` checks a custom Method's required interface; see **Adding a custom inference method** in Part 7 for implementation details. Public exports are defined in package `__init__.py`.

## Part 9: Checking that your model works

A program can run without errors and still give misleading scientific results. Software tests check whether BET behaves as implemented; model checks ask whether your assumptions and fitted predictions are appropriate for the problem you care about. Both matter.

### Verification and further reading

The repository's `TESTING.md` explains how to run automated tests, while Part 7 documents how to add an inference method. The included `Examples/` directory contains NUTS, SVI, and Save/Load scripts. The concise `docs/Cheatsheet.md` is a quick reference, not a replacement for this manual.

Smoke and integration tests show that supported workflows execute; they do not establish identifiability, credible-interval calibration, or scientific validity. For research claims, add application-specific simulation-based recovery and posterior predictive checks.

### Test strategy

The regression suite covers NUTS, SVI, scoring, validation, and Save/Load. Scientific validation also calls for synthetic hierarchical parameter recovery, transfer to new individuals with different observation counts, and numerical equivalence after persistence. See `TESTING.md` for test commands.

