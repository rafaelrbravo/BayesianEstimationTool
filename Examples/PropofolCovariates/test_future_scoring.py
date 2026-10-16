"""Guard against conditioning/scoring leakage in the propofol comparison."""
import numpy as np
import jax.numpy as jnp

from bet_baseline import N_FIT, ScoreFn, TestLikelihoodFn


def test_future_score_excludes_first_three_measurements():
    observed = jnp.array([0., 1., 0., 1., 1., 0., 1., 1.])
    concentration = jnp.array([0., 1., 0., 2., 4., 0., 8., 16.])
    prediction = concentration.at[1].set(1000.).at[3].set(1000.).at[4].set(1000.)
    data = {"observed": observed, "concentration": concentration}
    score = ScoreFn({"concentration": prediction}, data)
    np.testing.assert_allclose(score, 0., atol=1e-7)


def test_future_score_uses_only_measurements_after_conditioning():
    observed = jnp.array([1., 0., 1., 1., 0., 1., 1.])
    concentration = jnp.array([1., 0., 2., 4., 0., 8., 16.])
    prediction = concentration.at[5].set(16.).at[6].set(64.)
    data = {"observed": observed, "concentration": concentration}
    score = ScoreFn({"concentration": prediction}, data)
    np.testing.assert_allclose(score, (np.log(2.) + np.log(4.)) / 2., rtol=1e-6)


def test_conditioning_likelihood_masks_all_future_measurements():
    observed = jnp.array([0., 1., 1., 0., 1., 1., 1.])
    concentration = jnp.array([0., 1., 2., 0., 4., 8., 16.])
    data = {"observed": observed, "concentration": concentration}
    likelihood = TestLikelihoodFn({"sigma": jnp.array(0.2)}, {}, data,
                                  {"concentration": jnp.ones_like(concentration)})
    _, distribution = likelihood["concentration"]
    # A masked likelihood contributes zero log density at unobserved and future events.
    log_prob = np.asarray(distribution.log_prob(concentration))
    np.testing.assert_allclose(log_prob[[0, 3, 5, 6]], 0., atol=1e-7)
    assert np.all(np.abs(log_prob[[1, 2, 4]]) > 1e-6)
    assert N_FIT == 3


def test_future_score_uses_exactly_one_postconditioning_measurement():
    observed = jnp.array([1., 0., 1., 0., 1., 0., 1.])
    concentration = jnp.array([1., 0., 2., 0., 3., 0., 4.])
    prediction = jnp.array([100., 100., 200., 100., 300., 100., 8.])
    data = {"observed": observed, "concentration": concentration}
    np.testing.assert_allclose(ScoreFn({"concentration": prediction}, data), np.log(2.), rtol=1e-6)


def test_future_score_matches_numpy_reference_across_event_patterns():
    rng = np.random.default_rng(20261016)
    for _ in range(30):
        observed = rng.integers(0, 2, size=24)
        # Guarantee at least four measured concentrations in every trial.
        observed[:N_FIT + 1] = 1
        truth = np.exp(rng.normal(size=24))
        prediction = truth * np.exp(rng.normal(scale=0.5, size=24))
        measured_positions = np.flatnonzero(observed)
        future_positions = measured_positions[N_FIT:]
        expected = np.mean(np.abs(np.log(prediction[future_positions]) - np.log(truth[future_positions])))
        data = {"observed": jnp.asarray(observed), "concentration": jnp.asarray(truth)}
        actual = ScoreFn({"concentration": jnp.asarray(prediction)}, data)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
