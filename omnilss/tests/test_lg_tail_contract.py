"""LG infinite-tail, broadcasting and live R regression checks."""
import numpy as np
import pytest
import jax.random as jr
from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


def test_floor_boundaries_and_invalid_inputs():
    f = resolve("LG")
    np.testing.assert_allclose(f.p([1., 1.9], .4), [.4/-np.log1p(-.4)]*2)
    np.testing.assert_array_equal(f.p([-np.inf, 0., .9, np.inf]), [0., 0., 0., 1.])
    np.testing.assert_array_equal(f.q([0., 1.]), [0., np.inf])
    assert f.q(-1000., log_p=True) == 1
    assert np.isnan(f.q(-1000., log_p=True, max_value=0))
    assert np.all(np.isnan(f.q([-1., 2., np.nan])))
    assert np.isnan(f.p(1., 1.))
    assert np.isnan(f.q(.5, 0.))


def test_broadcast_extreme_tail_and_search_limit():
    f = resolve("LG")
    q = float(f.q(.99, .99999))
    assert q > 10000
    assert f.p(q, .99999) >= .99
    assert f.p(q-1, .99999) < .99
    assert np.isnan(f.q(.99, .99999, max_value=10000))
    k = np.arange(1, 50001, dtype=float)
    expected = np.sum(.99999**k/k)/-np.log1p(-.99999)
    assert float(f.p(50000, .99999)) == pytest.approx(expected, abs=2e-12)
    assert f.p([[0.], [1.9]], [.2, .8]).shape == (2, 2)
    assert np.asarray(f.p(1.)).shape == ()
    x = float(f.q(-1000., .4, lower_tail=False, log_p=True))
    assert f.p(x, .4, lower_tail=False, log_p=True) <= -1000
    assert f.p(x-1, .4, lower_tail=False, log_p=True) > -1000


def test_live_r_interior_probabilities():
    bus, f = RTestBus(), resolve("LG")
    mu = [.2, .5, .9]
    for tail in [True, False]:
        args = {"mu": mu, "lower.tail": tail}
        np.testing.assert_allclose(f.p([1., 2., 5.], mu, lower_tail=tail),
            bus.eval_family("LG", "p", dict(q=[1., 2., 5.], **args)), atol=1e-12)
        np.testing.assert_array_equal(f.q([.1, .5, .9], mu, lower_tail=tail),
            bus.eval_family("LG", "q", dict(p=[.1, .5, .9], **args)))


def test_sampling_support_and_tail_frequency():
    f = resolve("LG")
    key = jr.PRNGKey(123)
    sample = np.asarray(f.r(key, 800, .99999))
    np.testing.assert_array_equal(sample, f.r(key, 800, .99999))
    assert np.all(np.isfinite(sample) & (sample >= 1))
    assert np.max(sample) > 10000
    assert abs(np.mean(sample > 10000) - float(f.p(10000, .99999, lower_tail=False))) < .05
