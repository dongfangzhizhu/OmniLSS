"""BB sampling validity and independent distribution-level checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnilss.dpqr_functions import rBB
from tests.rbus.core import RTestBus


def test_zero_denominator_mixed_invalid_and_vector_parameters():
    key = jax.random.PRNGKey(41)
    sample = rBB(key, 6, [.2,.3,0.,.5,.6,.7], .2, [0,5,5,-1,2.5,10])
    assert sample[0] == 0
    assert 0 <= sample[1] <= 5
    assert np.all(np.isnan(sample[2:5]))
    assert 0 <= sample[5] <= 10
    for params in [dict(mu=np.nan),dict(sigma=0),dict(sigma=np.inf),dict(bd=np.inf)]:
        assert np.all(np.isnan(rBB(key, 4, **params)))
    np.testing.assert_array_equal(rBB(key, 50, .3, .2, 0), np.zeros(50))


def test_small_dispersion_exact_binomial_draw_jit_and_reproducibility():
    key = jax.random.PRNGKey(3)
    # Compare the actual limit branch with an independent JAX binomial draw,
    # including sigma whose reciprocal would produce enormous beta shapes.
    _, bin_key = jax.random.split(key)
    expected = jax.random.binomial(bin_key, n=10., p=.3, shape=(200,))
    actual = rBB(key, 200, .3, 1e-300, 10)
    np.testing.assert_array_equal(actual, expected)
    compiled = jax.jit(lambda k: rBB(k, 200, .3, 1e-300, 10))(key)
    np.testing.assert_array_equal(compiled, actual)
    np.testing.assert_array_equal(rBB(key, 200, .3, .2, 10), rBB(key, 200, .3, .2, 10))


def test_sample_size_contract():
    key = jax.random.PRNGKey(0)
    assert rBB(key, 0).shape == (0,)
    for n in [-1, True, 1.5]:
        with pytest.raises(ValueError, match="nonnegative integer"):
            rBB(key, n)
    with pytest.raises(ValueError):
        rBB(key, 3, bd=[1,2])


@pytest.mark.parametrize('sigma', [.2, 1e-8])
def test_empirical_mass_matches_live_r(sigma):
    expected = RTestBus().eval_family('BB', 'd', dict(x=list(range(11)), mu=.3, sigma=sigma, bd=10))
    draws = np.asarray(rBB(jax.random.PRNGKey(44), 20000, .3, sigma, 10), dtype=int)
    frequency = np.bincount(draws, minlength=11)/draws.size
    se = np.sqrt(expected*(1-expected)/draws.size)
    assert np.all(np.abs(frequency-expected) < 6*se + 2/draws.size)
