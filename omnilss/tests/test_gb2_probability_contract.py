"""GB2 signed shape and tail regressions, including live R comparisons."""
import numpy as np
import pytest
import jax.random as jr

from omnilss.dpqr_functions import pGB2, qGB2, rGB2
from tests.rbus.core import RTestBus


@pytest.mark.parametrize("sigma", [0.05, 2., -2.])
def test_live_r_probability(sigma):
    bus = RTestBus()
    params = dict(mu=1.2, sigma=sigma, nu=2.5, tau=3.)
    x = [0.1, 0.8, 1., 3., 10.]
    p = [0.01, 0.2, 0.5, 0.8, 0.99]
    np.testing.assert_allclose(pGB2(x, **params), bus.eval_family(
        "GB2", "p", dict(q=x, **params)), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(qGB2(p, **params), bus.eval_family(
        "GB2", "q", dict(p=p, **params)), rtol=1e-10)


@pytest.mark.parametrize("sigma", [1., -1.])
def test_lomax_extreme_tail_and_log_inverse(sigma):
    # nu=tau=1 gives F(x)=x/(1+x), for either sign of sigma.
    assert pGB2(1e100, sigma=sigma, lower_tail=False) == pytest.approx(1e-100, rel=1e-12, abs=0)
    assert pGB2(1e100, sigma=sigma, lower_tail=False, log_p=True) == pytest.approx(-100*np.log(10))
    assert qGB2(1e-100, sigma=sigma, lower_tail=False) == pytest.approx(1e100, rel=1e-12)
    assert qGB2(-1e-100, sigma=sigma, log_p=True) == pytest.approx(1e100, rel=1e-12)
    np.testing.assert_array_equal(qGB2([0., 1.], sigma=sigma), [0., np.inf])
    np.testing.assert_array_equal(pGB2([-np.inf, 0., np.inf], sigma=sigma), [0., 0., 1.])


def test_broadcast_invalid_and_sampling():
    sigma = np.array([[2.], [-2.]])
    p = np.array([.1, .5, .9])
    q = qGB2(p, sigma=sigma, tau=150.)
    assert q.shape == (2, 3)
    np.testing.assert_allclose(pGB2(q, sigma=sigma, tau=150.), np.broadcast_to(p, q.shape), rtol=1e-10)
    for params in [dict(mu=0), dict(sigma=0), dict(nu=-1), dict(tau=np.inf)]:
        assert np.isnan(pGB2(1., **params))
        assert np.isnan(qGB2(.5, **params))
        assert np.all(np.isnan(rGB2(jr.PRNGKey(1), 4, **params)))
    assert np.all(np.isnan(qGB2([-1., 2., np.nan])))
    samples = rGB2(jr.PRNGKey(2), 4000, sigma=-2., nu=2., tau=3.)
    uniform = np.asarray(pGB2(samples, sigma=-2., nu=2., tau=3.))
    assert abs(uniform.mean() - .5) < .02
    assert abs((uniform < .1).mean() - .1) < .02
