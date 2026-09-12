import jax
import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from omnilss.dpqr_functions import dSICHEL


def test_point_mass_beyond_numerical_support():
    # Boundary support is a project extension: R rejects tau=0 and tau=1.
    x = jnp.array([-np.inf, -1., 0., .5, 10001., 1e9, np.inf, np.nan])
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        for log in (False, True):
            density = jax.jit(lambda v: fam.d(v, tau=1., log=log))(x)
            expected = np.array([0., 0., 1., 0., 0., 0., 0., np.nan])
            if log:
                with np.errstate(divide="ignore"):
                    expected = np.log(expected)
            np.testing.assert_allclose(density, expected)
            for lower_tail in (False, True):
                expected = np.array([0., 0., 1., 1., 1., 1., 1., np.nan])
                if not lower_tail:
                    expected = 1. - expected
                if log:
                    with np.errstate(divide="ignore"):
                        expected = np.log(expected)
                actual = jax.jit(lambda v: fam.p(
                    v, tau=1., lower_tail=lower_tail, log_p=log))(x)
                np.testing.assert_allclose(actual, expected)
        assert np.isnan(fam.d(10001., tau=1., sigma=-1.))
        assert np.isnan(fam.p(10001., tau=1., sigma=-1.))
        assert np.isnan(fam.d(10001., tau=.5))
        assert np.isnan(fam.p(10001., tau=.5))


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
        assert np.isnan(fam.q(.9, max_value=0))
        with np.testing.assert_raises(ValueError):
            fam.q(.5, max_value=-1)
        for lower_tail in (True, False):
            expected = [0., np.inf] if lower_tail else [np.inf, 0.]
            actual = jax.jit(lambda v: fam.q(
                v, 1., .5, .2, .3, lower_tail=lower_tail, log_p=True
            ))(jnp.array([-jnp.inf, 0.]))
            np.testing.assert_array_equal(actual, expected)


def test_zero_sichel_log_upper_tail_preserves_small_probabilities():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        x = jnp.array([1., 2., 3.])
        upper = fam.p(x, 2., .7, .2, .35, lower_tail=False, log_p=True)
        actual = fam.q(upper, 2., .7, .2, .35,
                       lower_tail=False, log_p=True)
        np.testing.assert_array_equal(actual, x)


def test_zero_sichel_extreme_tails_and_roundtrips():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        x = jnp.array([0., 3., 100.])[:, None]
        tau = jnp.array([0., .35])
        for lower_tail in (True, False):
            log_prob = jax.jit(lambda v: fam.p(
                v, 2., .7, .2, tau, lower_tail=lower_tail, log_p=True))(x)
            assert np.isfinite(log_prob[1:]).all()
            if not lower_tail:
                assert (np.asarray(log_prob[-1]) < -40).all()
            # Lower-tail probabilities at x=100 round to one, so invert
            # central points there; the upper log tail retains all points.
            points = slice(None, 2) if lower_tail else slice(None)
            actual = jax.jit(lambda v: fam.q(
                v, 2., .7, .2, tau, lower_tail=lower_tail, log_p=True))(log_prob[points])
            np.testing.assert_array_equal(actual, np.broadcast_to(x[points], actual.shape))


def test_zero_sichel_cdf_support_and_invalid_parameters():
    for name in ("ZASICHEL", "ZISICHEL"):
        fam = resolve(name)
        for lower_tail in (True, False):
            expected = [0., 0., 1., 1.] if lower_tail else [1., 1., 0., 0.]
            np.testing.assert_array_equal(fam.p(
                [-np.inf, -1., 0., np.inf], tau=1., lower_tail=lower_tail), expected)
            np.testing.assert_allclose(fam.p([0., .5], tau=.3, lower_tail=lower_tail),
                                       fam.p(0., tau=.3, lower_tail=lower_tail))
        assert np.isnan(fam.p([np.nan, 10001.])).all()
        assert np.isnan(fam.p(np.inf, sigma=-1.))
        assert np.isnan(fam.q(0., sigma=-1., tau=1.))
        assert np.isnan(fam.q([.1, np.nan, np.inf], log_p=True)).all()