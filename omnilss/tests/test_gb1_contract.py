"""GB1 kernel validation against the installed R oracle and Beta special case."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta
from omnilss.distribution_registry import resolve
from omnilss.gb1 import dGB1, pGB1, qGB1, rGB1, log_density
from tests.rbus.core import RTestBus


@pytest.mark.parametrize("nu,tau", [(.3, .7), (1., 1.), (4., 2.)])
def test_live_r(nu, tau):
    bus = RTestBus()
    params = dict(mu=.35, sigma=.4, nu=nu, tau=tau)
    x, p = [.05,.2,.5,.8,.95], [.01,.2,.5,.8,.99]
    for name, function, arg, values in [('d',dGB1,'x',x), ('p',pGB1,'q',x), ('q',qGB1,'p',p)]:
        expected = bus.eval_family('GB1', name, dict(params, **{arg:values}))
        np.testing.assert_allclose(function(values, **params), expected, rtol=1e-9, atol=1e-11)


def test_beta_special_case_and_normalization():
    x = np.linspace(.01,.99,30)
    a,b = .35*(1/.4**2-1), .65*(1/.4**2-1)
    np.testing.assert_allclose(dGB1(x,.35,.4), beta.pdf(x,a,b), rtol=1e-12)
    integral, _ = quad(lambda x: float(dGB1(x,.35,.4,2.,.8)), 0, 1)
    assert integral == pytest.approx(1., abs=1e-7)


def test_family_ad_broadcast_and_boundaries():
    f = resolve('GB1')
    assert f.links['sigma'] == 'logit'
    x = jnp.array([[.2],[.7]])
    params = (.35,.4,jnp.array([.5,2.,4.]),.8)
    np.testing.assert_allclose(f.d(x,*params,log=True), jax.jit(log_density)(x,*params))
    assert f.p(x,*params).shape == (2,3)
    assert np.isfinite(jax.grad(lambda s: log_density(.5,.35,s,2.,.8))(.4))
    np.testing.assert_array_equal(f.p([-1,0,1,np.inf]), [0,0,1,1])
    np.testing.assert_array_equal(f.q([0,1]), [0,1])
    np.testing.assert_array_equal(f.q([0,1],lower_tail=False), [1,0])
    for params in [dict(mu=0),dict(sigma=1),dict(nu=-1),dict(tau=np.inf)]:
        for fun, v in [(dGB1,.5),(pGB1,.5),(qGB1,.5)]:
            assert np.isnan(fun(v, **params))
    assert np.all(np.isnan(qGB1([-1,2,np.nan])))


def test_tails_and_sample_distribution():
    p = np.array([1e-12,.1,.9])
    x = qGB1(np.log(p), .35,.4,2.,.8,lower_tail=False,log_p=True)
    np.testing.assert_allclose(pGB1(x,.35,.4,2.,.8,lower_tail=False),p,rtol=1e-4,atol=1e-15)
    sample = rGB1(jax.random.PRNGKey(17),3000,.35,.4,2.,.8)
    assert np.all((sample > 0) & (sample < 1))
    assert abs(np.mean(pGB1(sample,.35,.4,2.,.8))-.5) < .02
