"""Independent PIG mixture references and regression coverage."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import invgauss, poisson

from omnilss.pig import dPIG, pPIG, qPIG, rPIG
from omnilss.distribution_registry import resolve
from omnilss.dpqr_functions import pPIG2, qPIG2


@pytest.mark.parametrize('mu,sigma', [(1., .2), (4., 2.), (800., .01)])
def test_independent_mixture(mu, sigma):
    x = int(mu)
    # Rate/mu ~ IG(mean=1, shape=1/sigma).
    reference = quad(lambda t: poisson.pmf(x, mu*t) *
                     invgauss.pdf(t, sigma, scale=1/sigma), 0, np.inf,
                     epsabs=1e-12)[0]
    np.testing.assert_allclose(dPIG(x, mu, sigma), reference, rtol=1e-8)


def test_broadcast_jit_and_quantile():
    mu = jnp.array([1., 20., 800.])
    sigma = jnp.array([.2, .5, .01])
    probs = jnp.array([.1, .5, .99])
    q = jax.jit(qPIG)(probs, mu, sigma)
    assert np.all(pPIG(q, mu, sigma) >= probs)
    assert np.all(pPIG(q-1, mu, sigma) < probs)
    np.testing.assert_allclose(jax.jit(dPIG)(q, mu, sigma),
                              [dPIG(q[i], mu[i], sigma[i]) for i in range(3)])
    assert dPIG(jnp.arange(3)[:, None], mu, sigma).shape == (3, 3)


@pytest.mark.parametrize('name', ['PIG', 'PIG2'])
def test_family_derivatives(name):
    fam = resolve(name)
    x = jnp.array([0., 2., 6.])
    np.testing.assert_allclose(fam.g_dev_inc(x, 3., .7), -2*fam.d(x, 3., .7, log=True))
    for par in ('mu', 'sigma'):
        assert np.all(np.isfinite(fam.score_functions[par](x, 3., .7)))
        assert np.all(np.isfinite(fam.hessian_functions[par](x, 3., .7)))


def test_boundaries_and_limits():
    np.testing.assert_array_equal(dPIG(jnp.array([-1., .5, jnp.inf])), 0.)
    np.testing.assert_array_equal(pPIG(jnp.array([-jnp.inf, jnp.inf])), [0., 1.])
    np.testing.assert_array_equal(qPIG(jnp.array([0., 1.])), [0., np.inf])
    assert np.isnan(dPIG(10001)) and np.isnan(pPIG(10001))
    assert np.isnan(qPIG(.999, max_value=0))
    assert np.isnan(dPIG(0, sigma=-1))
    assert np.isnan(qPIG(-.1))
    for pfun, qfun in [(pPIG, qPIG), (pPIG2, qPIG2)]:
        np.testing.assert_array_equal(qfun(jnp.array([0., 1.]), lower_tail=False), [np.inf, 0.])
        assert pfun(-1, lower_tail=False) == 1
        np.testing.assert_array_equal(qfun(jnp.log(jnp.array([.2, .8])), log_p=True,
                                         lower_tail=False), qfun(jnp.array([.8, .2])))


def test_sampling_moments_and_reproducibility():
    key = jax.random.PRNGKey(81)
    sample = rPIG(key, 60000, 5., .4)
    np.testing.assert_array_equal(sample, rPIG(key, 60000, 5., .4))
    assert np.all(sample >= 0) and np.all(sample == np.floor(sample))
    assert abs(np.mean(sample)-5) < .1
    assert abs(np.var(sample)-15) < .8
    assert np.max(rPIG(key, 100, 20000., .01)) > 10000
    assert rPIG(key, 0).shape == (0,)
    assert rPIG(key, 3, jnp.array([1., 5., 20.])).shape == (3,)