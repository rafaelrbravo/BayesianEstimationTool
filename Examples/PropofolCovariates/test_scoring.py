"""Check that only measurements after the first three contribute to scores.

Run: python Examples/PropofolCovariates/test_scoring.py
"""
import jax.numpy as jnp
import numpy as np

from bet_baseline import ScoreFn, TestLikelihoodFn


def test_future_only_score():
    data = {"observed": jnp.array([1, 0, 1, 1, 0, 1, 1]),
            "concentration": jnp.array([1., 0., 2., 3., 0., 4., 5.])}
    model = {"concentration": jnp.array([100., 1., 200., 300., 1., 4., 5.])}
    np.testing.assert_allclose(ScoreFn(model, data), 0., atol=1e-7)
    model["concentration"] = model["concentration"].at[5].set(8.)
    np.testing.assert_allclose(ScoreFn(model, data), np.log(2) / 2, rtol=1e-6)
    likelihood = TestLikelihoodFn({"sigma": jnp.array(0.2)}, {}, data, model)["concentration"][1]
    np.testing.assert_array_equal(np.asarray(likelihood._mask), [True, False, True, True, False, False, False])



def test_repeated_measurements_count_toward_conditioning():
    """Repeated same-time measurements each consume a conditioning slot."""
    data = {"observed": jnp.array([1, 1, 0, 1, 1, 0, 1]),
            "concentration": jnp.array([2., 2.1, 0., 3., 4., 0., 5.])}
    model = {"concentration": jnp.array([20., 21., 1., 30., 8., 1., 10.])}
    likelihood = TestLikelihoodFn({"sigma": jnp.array(0.2)}, {}, data, model)["concentration"][1]
    np.testing.assert_array_equal(np.asarray(likelihood._mask),
                                  [True, True, False, True, False, False, False])
    np.testing.assert_allclose(ScoreFn(model, data), np.log(2.), rtol=1e-6)


if __name__ == "__main__":
    test_future_only_score()
    test_repeated_measurements_count_toward_conditioning()
    print("Future-only scoring checks passed")
