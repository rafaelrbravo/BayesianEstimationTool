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
    mask = np.asarray(distribution._mask)
    np.testing.assert_array_equal(mask, [False, True, True, False, True, False, False])
    assert N_FIT == 3
