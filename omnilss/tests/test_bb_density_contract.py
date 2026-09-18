"""BB fit/public density consistency and the binomial limit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binom
from omnilss.dpqr_functions import dBB, pBB
from omnilss.distribution_registry import resolve
from omnilss.distributions_b7 import _bb_log_pdf
from tests.rbus.core import RTestBus


@pytest.mark.parametrize('sigma', [1e-12, 9e-5, 1e-4, .2])
def test_live_r_density_and_cdf_increments(sigma):
    x = np.arange(11.)
    expected = RTestBus().eval_family('BB','d',dict(x=x.tolist(),mu=.3,sigma=sigma,bd=10))
    actual = dBB(x,.3,sigma,10)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(actual, pBB(x,.3,sigma,10)-pBB(x-1,.3,sigma,10), atol=1e-10)


def test_fit_kernel_jit_and_gradients():
    x = jnp.arange(11.)
    for sigma in [1e-30,.2]:
        np.testing.assert_allclose(resolve('BB').d(x,.3,sigma,10,log=True),
                                  jax.jit(_bb_log_pdf)(x,.3,sigma,10))
        assert np.isfinite(jax.grad(lambda m: jnp.sum(_bb_log_pdf(x,m,sigma,10)))(.3))
    np.testing.assert_allclose(dBB(x,.3,1e-30,10), binom.pmf(x,10,.3), atol=1e-14)
    assert jax.grad(lambda s: _bb_log_pdf(3.,.3,s,10))(1e-30) == 0


def test_support_invalid_parameters_and_broadcast():
    np.testing.assert_array_equal(dBB([-1.,.5,11.,np.inf],.3,.2,10), 0)
    assert np.isnan(dBB(np.nan))
    for params in [dict(mu=0),dict(sigma=0),dict(bd=-1),dict(bd=2.5)]:
        assert np.isnan(dBB(0,**params))
    assert dBB(0,.3,.2,0) == pytest.approx(1.)
    assert resolve('BB').d([[0.],[1.]], [.2,.8], .2, [2,5]).shape == (2,2)
