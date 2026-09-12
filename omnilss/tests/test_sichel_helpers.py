"""R-helpers for the SICHEL family: tofySICHEL and VSICHEL.

Golden constants were generated with R 4.6.0 / gamlss.dist 6.1.1
(``tofySICHEL`` drives the GAMLSS mu-score; ``VSICHEL`` is the variance
function) and are compared here at tight tolerance.
"""

import jax
import jax.numpy as jnp
import numpy as np

from omnilss.dpqr_functions import VSICHEL, tofySICHEL
from omnilss.distributions_b6 import SICHEL_DENSITY_MAX_COUNT


def test_tofy_sichel_matches_r():
    y = jnp.array([0., 1., 2., 5., 10., 100., 10000.])
    expected = np.array([
        1.1536147646703467, 1.5067296111896378, 1.9242441704113613,
        3.4248619270387208, 6.2344942898961646, 59.258908714461143,
        5906.8952274657795,
    ])
    np.testing.assert_allclose(
        tofySICHEL(y, 2., .5, .3), expected, rtol=1e-9, atol=1e-12)

    y2 = jnp.array([0., 1., 3., 50.])
    np.testing.assert_allclose(
        tofySICHEL(y2, 1., .5, .2),
        [0.71125198708488169, 0.96753516681773488,
         1.6247997369785334, 21.545814612583158],
        rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        tofySICHEL(y2, 10., .1, 1.),
        [5.7258999392003158, 6.0700915649196014,
         6.8113992192362778, 33.189949582465815],
        rtol=1e-9, atol=1e-12)


def test_tofy_sichel_broadcast_jit_and_gradient():
    y = jnp.array([1., 3.])[:, None]
    mu = jnp.array([1., 10.])
    nu = jnp.array([.2, 1.])
    actual = jax.jit(tofySICHEL)(y, mu, .5, nu)
    expected = [[tofySICHEL(float(v), float(m), .5, float(n))
                 for m, n in zip(mu, nu)] for v in (1., 3.)]
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (2, 2)
    gradient = jax.grad(lambda m: tofySICHEL(5., m, .5, .3))(2.)
    assert np.isfinite(gradient)


def test_tofy_sichel_boundaries():
    x = jnp.array([-1., .5, jnp.inf, -jnp.inf, jnp.nan, 10001., 0.])
    actual = tofySICHEL(x, 1., .5, .2)
    assert np.isnan(actual[jnp.array([0, 2, 3, 4])]).all()
    # Non-integer counts truncate like R (floor); interior and boundary
    # integer counts stay valid within the budget.
    np.testing.assert_allclose(actual[1], tofySICHEL(0., 1., .5, .2))
    assert np.isfinite(actual[5])
    assert np.isfinite(actual[6])
    assert np.isnan(tofySICHEL(0., sigma=-1.))
    assert np.isnan(tofySICHEL(0., nu=np.inf))
    assert np.isnan(tofySICHEL(float(SICHEL_DENSITY_MAX_COUNT) + 1.))


def test_vsichel_matches_pmf_moments():
    from omnilss.distributions_b6 import _sichel_log_pmf_sequence
    for mu, sigma, nu in ((1., .5, .2), (2., .5, .3), (10., .1, 1.)):
        support = jnp.arange(100001, dtype=jnp.float64)
        log_pmf = _sichel_log_pmf_sequence(mu, sigma, nu, 100000)
        weights = jnp.exp(log_pmf)
        mean = jnp.sum(support * weights)
        second = jnp.sum(jnp.square(support) * weights)
        variance = second - jnp.square(mean)
        np.testing.assert_allclose(
            VSICHEL(mu, sigma, nu), variance, rtol=1e-6, atol=1e-8)


def test_vsichel_invalid_parameters():
    assert np.isnan(VSICHEL(-1., .5, .2))
    assert np.isnan(VSICHEL(1., -.5, .2))
    assert np.isnan(VSICHEL(1., .5, np.inf))
    assert np.isnan(VSICHEL(np.nan, .5, .2))
    # Reduced form: nu = -0.5 gives c = 1, so varY = mu + mu^2 * sigma.
    np.testing.assert_allclose(VSICHEL(2., .5, -.5), 2. + 4. * .5)
    value = jax.jit(VSICHEL)(jnp.array([1., 2.]), .5, .2)
    assert value.shape == (2,)
    assert np.isfinite(value).all()
