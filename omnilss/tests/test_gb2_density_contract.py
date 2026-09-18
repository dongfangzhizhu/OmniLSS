"""Regression tests for the shared, differentiable GB2 density kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.dpqr_functions import dGB2
from omnilss.distributions_b8 import _gb2_log_pdf
from tests.rbus.core import RTestBus


def test_density_extreme_inputs_are_finite_and_shared():
    x = jnp.array([1e-200, 1.0, 1e200])
    params = (2.0, 20.0, 3.0, 150.0)
    actual = dGB2(x, *params, log=True)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(_gb2_log_pdf(x, *params), actual)
    # sigma=20 previously overflowed exp(sigma*log(x/mu)) in the fit kernel.
    gradient = jax.grad(lambda sigma: jnp.sum(_gb2_log_pdf(x, 2., sigma, 3., 150.)))(20.)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(jax.jit(_gb2_log_pdf)(x, *params), actual)


def test_density_support_and_invalid_parameters():
    np.testing.assert_array_equal(dGB2([-1., 0., np.inf]), [0., 0., 0.])
    assert np.all(np.isneginf(dGB2([-1., 0., np.inf], log=True)))
    assert np.isnan(dGB2(np.nan))
    for params in [(0., 1., 1., 1.), (1., 0., 1., 1.),
                   (1., 1., -1., 1.), (1., 1., 1., np.inf)]:
        assert np.isnan(dGB2(1., *params))


@pytest.mark.parametrize("sigma,tau", [(0.05, 150.), (20., 2.), (-2., 3.)])
def test_density_against_live_r(sigma, tau):
    # No optional skip: this regression requires an available R environment.
    bus = RTestBus()
    x = np.array([0.3, 0.8, 1., 2., 5.])
    expected = bus.eval_family("GB2", "d", {
        "x": x.tolist(), "mu": 1.2, "sigma": sigma,
        "nu": 2.5, "tau": tau, "log": True,
    })
    np.testing.assert_allclose(dGB2(x, 1.2, sigma, 2.5, tau, log=True),
                               expected, rtol=1e-10, atol=1e-10)
