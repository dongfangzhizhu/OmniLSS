"""DEL regression checks against independent Poisson/NB convolution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from omnilss.distributions_b6 import DEL, _del_log_pdf
from omnilss.distribution_registry import resolve


@pytest.fixture(scope="module")
def family():
    return DEL()


@pytest.mark.parametrize("mu,sigma,nu", [(2., 1., .5), (150., .3, .2), (4., 2., .9)])
def test_convolution(family, mu, sigma, nu):
    x = np.arange(601)
    expected = np.convolve(
        poisson.pmf(x, mu * nu),
        nbinom.pmf(x, 1 / sigma, 1 / (1 + mu * sigma * (1 - nu))),
    )[:len(x)]
    # Sparse density queries keep AD test memory small while covering y > 100.
    points = np.array([0, 1, 2, 100, 101, 150, 300, 600])
    np.testing.assert_allclose(family.d(points, mu, sigma, nu), expected[points], rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(family.p(points, mu, sigma, nu), np.cumsum(expected)[points], atol=1e-11)
    p = np.array([.01, .5, .95])
    q = np.asarray(family.q(p, mu, sigma, nu))
    np.testing.assert_array_equal(q, np.searchsorted(np.cumsum(expected), p))
    assert np.all(np.asarray(family.p(q, mu, sigma, nu)) >= p)
    assert np.all(np.asarray(family.p(q - 1, mu, sigma, nu)) < p)


def test_boundaries(family):
    np.testing.assert_array_equal(family.d(np.array([-1., .5, np.inf, -np.inf]), 2., 1., .5), 0.)
    assert np.isneginf(family.d(-1., 2., 1., .5, log=True))
    np.testing.assert_array_equal(family.p(jnp.array([-np.inf, -1., np.inf])), [0., 0., 1.])
    np.testing.assert_array_equal(family.p(jnp.array([-np.inf, np.inf]), lower_tail=False), [1., 0.])
    np.testing.assert_array_equal(family.p(jnp.array([-np.inf, np.inf]), log_p=True), [-np.inf, 0.])
    np.testing.assert_array_equal(family.q(jnp.array([0., 1.])), [0., np.inf])
    assert np.all(np.isnan(family.q(jnp.array([-.1, 1.1, np.nan]))))
    assert np.isnan(family.p(np.nan))
    assert np.isnan(family.p(10001.))
    assert np.isnan(family.d(10001., 2., 1., .5))
    assert np.isnan(family.q(.99, 100., 1., .5, max_value=2))
    assert np.isnan(family.p(1., nu=2.))
    assert family.p(2.9) == family.p(2.)


def test_vectorization_and_transforms(family):
    mu = jnp.array([[1.], [3.]])
    p = jnp.array([.2, .5, .8])
    q = family.q(p, mu, .5, .3)
    assert q.shape == (2, 3)
    np.testing.assert_array_equal(q, family.q(jnp.log(p), mu, .5, .3, log_p=True))
    np.testing.assert_array_equal(q, family.q(1 - p, mu, .5, .3, lower_tail=False))
    cdf = family.p(q, mu, .5, .3)
    np.testing.assert_allclose(jnp.exp(family.p(q, mu, .5, .3, log_p=True)), cdf)
    np.testing.assert_allclose(family.p(q, mu, .5, .3, lower_tail=False), 1 - cdf)
    np.testing.assert_allclose(jax.jit(family.p)(q, mu, .5, .3), cdf)


def test_poisson_limit_and_derivatives(family):
    x = np.array([0., 2., 101.])
    np.testing.assert_allclose(family.d(x, 3., 1e-5, .5), poisson.pmf(x, 3.), atol=1e-13)
    np.testing.assert_allclose(family.p(x, 3., 1e-5, .5), poisson.cdf(x, 3.), atol=1e-13)
    fn = lambda mu: _del_log_pdf(150., mu, .3, .2)
    grad = jax.grad(fn)(150.)
    finite_diff = (fn(150.001) - fn(149.999)) / .002
    np.testing.assert_allclose(grad, finite_diff, atol=1e-7)
    assert np.isfinite(jax.grad(jax.grad(fn))(150.))


def test_sampling(family):
    key = jax.random.PRNGKey(42)
    sample = np.asarray(family.r(key, 10000, 4., .5, .3))
    np.testing.assert_array_equal(sample, family.r(key, 10000, 4., .5, .3))
    assert np.all(sample >= 0) and np.all(sample == np.floor(sample))
    assert abs(sample.mean() - 4.) < .15
    assert abs(sample.var() - (4 + 16 * .5 * .7**2)) < .6
    assert family.r(key, 0).shape == (0,)
    assert family.r(key, 3, jnp.array([1., 2., 3.])).shape == (3,)


def test_delaport_registry_alias():
    family = resolve("DELAPORT")
    from omnilss.distributions_b13 import DELAPORT
    direct = DELAPORT()
    assert family.name == direct.name == "DELAPORT"
    assert family.links["nu"] == direct.links["nu"] == "logit"
    np.testing.assert_allclose(
        family.d(jnp.arange(4.), 2., 1., .5),
        DEL().d(jnp.arange(4.), 2., 1., .5),
    )
    base = DEL()
    for entry in (family, direct):
        for method, values in (("p", jnp.arange(4.)), ("q", jnp.array([.1, .5, .9]))):
            np.testing.assert_allclose(getattr(entry, method)(values, 2., 1., .5), getattr(base, method)(values, 2., 1., .5))
        key = jax.random.PRNGKey(123)
        np.testing.assert_array_equal(entry.r(key, 12), base.r(key, 12))
        assert np.isnan(entry.d(2., 2., 1., 2.))


def test_r_del_dpq(family):
    from tests.rbus.core import RTestBus
    try:
        bus = RTestBus()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    args = {"mu": [2., 150., 8.], "sigma": [1., .3, 2.], "nu": [.5, .2, .9]}
    for kind, input_name, values in [("d", "x", [0., 150., 101.]), ("p", "q", [2., 150., 101.]), ("q", "p", [.1, .5, .95])]:
        expected = bus.eval_family("DEL", kind, {input_name: values, **args})
        actual = getattr(family, kind)(jnp.array(values), *[jnp.array(v) for v in args.values()])
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)