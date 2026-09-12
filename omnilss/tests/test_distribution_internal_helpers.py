"""R gamlss.dist internal distribution helpers: tofyS, zetaP, GetBI_C, get_C.

Golden constants were generated with R 4.6.0 / gamlss.dist 6.1.1 at full
double precision.  get_C is validated elementwise because R's wrapper does
not recycle scalar arguments against vector x (documented below).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.dpqr_functions import (
    GetBI_C,
    SI_TOFY_MAX_COUNT,
    get_C,
    tofyS,
    tofySICHEL,
    zetaP,
)


def test_tofy_s_matches_r():
    y = jnp.array([0., 1., 3., 50.])
    np.testing.assert_allclose(
        tofyS(y, 1., .5, -.5),
        [0.70710678118654757, 0.95710678118654724,
         1.6429557958354901, 24.770600688382746],
        rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        tofyS(y, 5., 1., 1.),
        [2.2294321123667635, 2.8376016337125751,
         4.2805232440171377, 46.413580369427834],
        rtol=1e-9, atol=1e-12)


def test_tofy_s_truncates_and_bounds():
    # R indexes tofY[floor(y)]; non-integer counts truncate like C ints.
    np.testing.assert_allclose(tofyS(2.9, 1., .5, -.5), tofyS(2., 1., .5, -.5))
    np.testing.assert_allclose(
        tofySICHEL(2.9, 1., .5, .2), tofySICHEL(2., 1., .5, .2))
    assert np.isnan(tofyS(-1., 1., .5, -.5))
    assert np.isnan(tofyS(SI_TOFY_MAX_COUNT + 1., 1., .5, -.5))
    assert np.isnan(tofyS(1., sigma=-1.))
    with pytest.raises(ValueError):
        tofyS(1., max_count=-1)


def test_tofy_s_broadcast_jit_gradient():
    y = jnp.array([1., 3.])[:, None]
    mu = jnp.array([1., 5.])
    nu = jnp.array([-.5, 1.])
    actual = jax.jit(tofyS)(y, mu, .5, nu)
    expected = [[tofyS(float(v), float(m), .5, float(n))
                 for m, n in zip(mu, nu)] for v in (1., 3.)]
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (2, 2)
    assert np.isfinite(jax.grad(
        lambda m: tofyS(3., m, .5, .3))(1.))


def test_zeta_p_matches_r_and_scipy():
    x = jnp.array([12., 15., 20., 30.5])
    expected = np.array([1.000246086553308, 1.000030588236307,
                         1.0000009539620338, 1.0000000006585474])
    np.testing.assert_allclose(zetaP(x), expected, rtol=1e-12, atol=1e-15)
    from scipy.special import zeta as _zeta
    np.testing.assert_allclose(np.asarray(zetaP(x)), _zeta(x),
                               rtol=1e-10, atol=1e-15)
    # Broadcast and jit.
    grid = jnp.stack([x, x + 1.])
    np.testing.assert_allclose(jax.jit(zetaP)(grid), _zeta(grid),
                               rtol=1e-10, atol=1e-15)


def test_get_bi_c_matches_r():
    np.testing.assert_allclose(
        GetBI_C(jnp.array([.3, .5]), jnp.array([.5, 1.2]),
                jnp.array([10., 25.])),
        [0.36475594445521647, -0.093309953581539054],
        rtol=1e-12, atol=1e-15)
    # bd = 0 sums the single j = 0 term, so the constant is exactly 0.
    assert GetBI_C(.5, 1., 0.) == 0.
    np.testing.assert_allclose(
        jax.jit(GetBI_C)(jnp.array([.3]), jnp.array([.5]), jnp.array([10.])),
        [0.36475594445521647], rtol=1e-12)


def test_get_bi_c_invalid_parameters():
    assert np.isnan(GetBI_C(1., 1., 10.))   # mu must be < 1
    assert np.isnan(GetBI_C(0., 1., 10.))
    assert np.isnan(GetBI_C(.5, -1., 10.))
    assert np.isnan(GetBI_C(.5, 1., -1.))
    assert np.isnan(GetBI_C(.5, 1., 2.5))   # bd must be an integer
    assert np.isnan(GetBI_C(.5, 1., np.nan))


def test_get_c_matches_r_and_broadcasts():
    x = jnp.array([0., 5., 20.])
    expected = -0.021217228895772572
    actual = get_C(x, 3., 1.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-15)
    # R reads past the end of scalar mu/sigma here; the Python side
    # broadcasts and every element shares the same truncation.
    assert np.asarray(actual).shape == (3,)
    # Truncating the sum discards positive tail mass, enlarging the
    # normalizer constant.
    assert get_C(0., 3., 1.5, max_value=50) > expected
    assert np.isnan(get_C(0., -1., 1.5))
    assert np.isnan(get_C(0., 3., -1.5))
    with pytest.raises(ValueError):
        get_C(0., 3., 1.5, max_value=-1)
    with pytest.raises(ValueError):
        get_C(0., 3., 1.5, max_value=1.5)
