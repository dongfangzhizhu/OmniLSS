"""Shared LG/ZALG density validity, R parity and differentiation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.distribution_registry import resolve
from omnilss.distributions_b9 import _lg_log_pdf, _zalg_log_pdf
from tests.rbus.core import RTestBus


@pytest.mark.parametrize("name", ["LG", "ZALG"])
def test_invalid_mu_and_support(name):
    f = resolve(name)
    extra = (.2,) if name == "ZALG" else ()
    for mu in [-1., 0., 1., np.inf, np.nan]:
        assert np.all(np.isnan(f.d([-1., 0., 1., np.inf], mu, *extra)))
        assert np.isnan(f.p(1., mu, *extra))
    np.testing.assert_array_equal(f.d([-1., .5, np.inf], .4, *extra), [0., 0., 0.])
    assert np.all(np.isneginf(f.d([-1., np.inf], .4, *extra, log=True)))
    assert np.isnan(f.d(np.nan, .4, *extra))


def test_zalg_invalid_sigma_and_normalization():
    f = resolve("ZALG")
    for sigma in [-1., 0., 1., np.inf, np.nan]:
        assert np.all(np.isnan(f.d([0., 1.], .4, sigma)))
    assert np.sum(f.d(np.arange(200), .7, .2)) == pytest.approx(1., abs=1e-12)
    # Tiny valid parameters are not silently moved to machine epsilon.
    assert float(f.d(0., .4, 1e-30)) == pytest.approx(1e-30, rel=1e-12, abs=0)


def test_shared_kernel_jit_gradient_and_broadcast():
    x = jnp.array([0., 1., 2., 3.])
    f = resolve("ZALG")
    np.testing.assert_allclose(f.d(x, .4, .2, log=True),
                              jax.jit(_zalg_log_pdf)(x, .4, .2))
    grad = jax.grad(lambda s: jnp.sum(_zalg_log_pdf(x, .4, s)))(.2)
    assert float(grad) == pytest.approx(1/.2 - 3/.8)
    assert float(jax.grad(lambda m: _zalg_log_pdf(0., m, .2))(.4)) == 0.
    assert np.isfinite(jax.grad(lambda m: _lg_log_pdf(2., m))(.4))
    assert resolve("LG").d([[1.], [2.]], [.2, .8]).shape == (2, 2)


@pytest.mark.parametrize("name", ["LG", "ZALG"])
def test_live_r_density(name):
    bus, f = RTestBus(), resolve(name)
    x = [1., 2., 3., 5., 10.]
    args = {"x": x, "mu": .4}
    extra = ()
    if name == "ZALG":
        # Live installed R is the oracle; old vendored ZALG versions may
        # omit the (1-sigma) factor in their positive-density implementation.
        args["sigma"] = .2
        extra = (.2,)
    expected = bus.eval_family(name, "d", dict(args, log=True))
    np.testing.assert_allclose(f.d(x, .4, *extra, log=True), expected, atol=1e-10, rtol=1e-10)
