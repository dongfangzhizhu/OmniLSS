import jax
import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from omnilss.dpqr_functions import dSICHEL


def test_stable_density_support_and_likelihood():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        outside = jnp.array([-1., .5, jnp.inf, -jnp.inf])
        np.testing.assert_array_equal(fam.d(outside), 0.)
        np.testing.assert_array_equal(fam.d(outside, log=True), -np.inf)
        assert np.isnan(fam.d(jnp.array([jnp.nan, 10001.]))).all()
        for tau in (-.1, 1.1, np.nan):
            assert np.isnan(fam.d(0., tau=tau))
        assert np.isnan(fam.d(0., sigma=-1.))
        x = jnp.array([0., 1., 100.])
        for tau in (0., .3, 1.):
            actual = fam.d(x, 1., .5, .2, tau, log=True)
            np.testing.assert_allclose(fam.g_dev_inc(x, 1., .5, .2, tau), -2*actual)
        expected = np.log1p(-.3) + dSICHEL(100., 1., .5, .2, log=True)
        if name == "ZASICHEL":
            expected -= np.log(-np.expm1(dSICHEL(0., 1., .5, .2, log=True)))
        actual = fam.d(100., 1., .5, .2, .3, log=True)
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(fam.d([0., 1.], tau=1.), [1., 0.])


def test_zero_sichel_mass_and_cdf_consistency():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        tau = .3
        x = jnp.arange(0., 100.)
        mass = fam.d(x, 1., .5, .2, tau)
        cdf = fam.p(x, 1., .5, .2, tau)
        np.testing.assert_allclose(mass, cdf - fam.p(x - 1., 1., .5, .2, tau), atol=1e-7)
        np.testing.assert_allclose(jnp.sum(mass), 1., atol=1e-5)
        assert np.isfinite(mass).all()
        if name == "ZASICHEL":
            np.testing.assert_allclose(mass[0], tau)


def test_zero_sichel_broadcast_jit_and_random_generation():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        x = jnp.arange(3.)[:, None]
        values = jax.jit(fam.d)(x, jnp.array([1., 2.]), .5, .2, .3)
        assert values.shape == (3, 2)
        samples = jax.jit(fam.r, static_argnums=1)(jax.random.key(3), 8,
                                                    jnp.array([1., 2.]), .5, .2, .3)
        assert samples.shape == (8, 2)
        assert np.isfinite(samples).all()
        assert np.equal(samples, np.floor(samples)).all()


def test_zero_sichel_probability_boundaries():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        np.testing.assert_array_equal(fam.q(jnp.array([0., 1.]), 1., .5, .2, 1.), [0., 0.])
        assert np.isposinf(fam.q(1., 1., .5, .2, 0.))
        np.testing.assert_array_equal(fam.q(jnp.array([0., 1.]), 1., .5, .2, .3,
                                                       lower_tail=False), [np.inf, 0.])
        assert np.isnan(fam.q(-.1, 1., .5, .2, .3))
        assert np.isnan(fam.q(1.1, 1., .5, .2, .3))
        for lower_tail in (True, False):
            expected = [0., np.inf] if lower_tail else [np.inf, 0.]
            actual = jax.jit(lambda v: fam.q(
                v, 1., .5, .2, .3, lower_tail=lower_tail, log_p=True
            ))(jnp.array([-jnp.inf, 0.]))
            np.testing.assert_array_equal(actual, expected)