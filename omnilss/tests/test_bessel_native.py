"""The experimental native log K_nu kernel (D2.267, P6-1).

This kernel is verified but not enabled: the CPU fit cycle is 70x slower than
the scipy callback (371 s vs 5.31 s for one SI fit at max_iter=1), because the
fitting loop evaluates it per observation on scalars.  See
omnilss/bessel_native.py for the measurement and tools/bessel_native_scan.py
for the accuracy + timing artifact.

Three regions, each with its own contract:

* x >= 15  -- Debye asymptotic in log space (K underflows in double well before
  the asymptotic stops being valid, so the log form is not cosmetic);
* 3 <= x < 15 -- Taylor propagation of the ODE inward from x = 15;
* x < 3 -- the small-argument series.

The acceptance grid is nu in [-5, 20], x in [1e-8, 1e3]; the metric is the
*relative error of K* (|exp(native_logK - reference_logK) - 1|), because
comparing log K relatively blows up wherever log K -> 0.  tools/bessel_native_scan.py
produces the committed artifact with the mpmath reference; these tests keep a
fast scipy-backed version in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)  # parity prerequisite

import jax.numpy as jnp  # noqa: E402
from scipy import special  # noqa: E402

from omnilss.bessel import _log_bessel_kv_scipy  # noqa: E402
from omnilss.bessel_native import (  # noqa: E402
    _ASYMPTOTIC_START,
    _SERIES_LIMIT,
    _asymptotic_log_kv_slope,
    _ode_log_kv_slope,
    _series_log_kv_slope,
    log_bessel_kv_native,
    log_bessel_kv_ratio,
)


def _reference_log(nu, x):
    scaled = special.kve(np.asarray(nu, float), np.asarray(x, float))
    return np.log(np.maximum(scaled, np.finfo(np.float64).tiny)) - np.asarray(x, float)


def _kernel(values_nu, values_x):
    return np.asarray(jax.jit(log_bessel_kv_native)(jnp.asarray(values_nu), jnp.asarray(values_x)))


def _ratio_values(nu, x):
    value, ratio = jax.jit(log_bessel_kv_ratio)(jnp.asarray(nu), jnp.asarray(x))
    return np.asarray(value), np.asarray(ratio)


@pytest.mark.parametrize("nu", [-5.0, -2.5, -1.0, 0.0, 0.5, 1.0, 4.0, 9.5, 15.0, 20.0])
def test_matches_scipy_across_all_three_regions(nu):
    x = np.logspace(-8.0, 3.0, 23)
    native = _kernel(np.full(x.size, nu), x)
    expected = _reference_log(np.full(x.size, nu), x)
    np.testing.assert_allclose(
        np.expm1(native - expected), 0.0, atol=1e-12,
        err_msg=f"nu={nu}",
    )


def test_seams_are_continuous():
    # 3.0 and 15.0 are the branch boundaries: the two formulas must agree at the
    # same argument (comparing across x would just measure the slope).
    for nu in (0.0, 1.5, 7.0, 19.0):
        series = float(_series_log_kv_slope(jnp.float64(nu), jnp.float64(_SERIES_LIMIT))[0])
        ode = float(_ode_log_kv_slope(jnp.float64(nu), jnp.float64(_SERIES_LIMIT))[0])
        assert abs(np.expm1(series - ode)) < 1e-12, (nu, series, ode)

        asymptotic = float(
            _asymptotic_log_kv_slope(jnp.float64(nu), jnp.float64(_ASYMPTOTIC_START))[0]
        )
        ode_top = float(
            _ode_log_kv_slope(jnp.float64(nu), jnp.float64(_ASYMPTOTIC_START))[0]
        )
        assert abs(np.expm1(asymptotic - ode_top)) < 1e-12, (nu, asymptotic, ode_top)


@pytest.mark.parametrize("x", [0.01, 0.5, 2.0])
def test_half_integer_order_matches_the_closed_form(x):
    expected = 0.5 * np.log(np.pi / (2.0 * x)) - x
    for nu in (0.5, -0.5, 1.5, -1.5):
        # K_{-nu} = K_nu, so every half-integer reduces to the same closed form
        # only for |nu| = 1/2; the others are checked against scipy below.
        got = float(_kernel(np.float64(nu), np.float64(x)))
        if abs(abs(nu) - 0.5) < 1e-12:
            assert got == pytest.approx(expected, rel=1e-12)
        else:
            assert got == pytest.approx(
                float(_reference_log(np.float64(nu), np.float64(x))), rel=1e-11
            )


def test_large_argument_stays_finite_where_k_underflows():
    # K_20(1000) ~ 1e-436 in double: any implementation that forms K first
    # returns log(0) = -inf here (the D2.53 defect).
    native = float(_kernel(np.float64(20.0), np.float64(1000.0)))
    assert np.isfinite(native)
    assert native == pytest.approx(float(_reference_log(20.0, 1000.0)), rel=1e-10)


def test_negative_orders_equal_positive_orders():
    x = np.array([0.01, 0.4, 2.0, 6.0, 40.0])
    for nu in (0.5, 1.0, 3.5, 12.0):
        positive = _kernel(np.full(x.size, nu), x)
        negative = _kernel(np.full(x.size, -nu), x)
        np.testing.assert_allclose(negative, positive, rtol=0, atol=1e-14)


def test_ratio_matches_the_two_log_definition():
    # The ratio is defined as K_{nu+1}/K_nu; the point of the helper is that it
    # does not *compute* it as a difference of two logs, so the two agree to
    # ~1e-13 (the log-difference noise) but the helper is the accurate side.
    cases = [(0.0, 4.0), (1.0, 4.0), (2.5, 0.7), (5.0, 20.0), (12.0, 200.0),
             (-1.0, 4.0), (-1.5, 0.5), (-0.5, 2.0)]
    for nu, x in cases:
        _, ratio = _ratio_values(np.float64(nu), np.float64(x))
        expected = float(
            _reference_log(np.float64(nu + 1.0), np.float64(x))
            - _reference_log(np.float64(nu), np.float64(x))
        )
        assert abs(np.expm1(float(ratio) - expected)) < 1e-12, (nu, x)


def test_ratio_beats_the_log_difference_at_the_gig_tail():
    # dGIG(mu=2, sigma=0.5, nu=1) needs c at x = 1/sigma^2 = 4 and multiplies
    # log(c) by an exponent of ~300, so this is where the ratio matters.
    nu, x = 1.0, 4.0
    _, ratio = _ratio_values(np.float64(nu), np.float64(x))
    accurate = float(ratio)

    # The two-log route through the same kernel.
    two_logs = float(
        _kernel(np.float64(nu + 1.0), np.float64(x))
        - _kernel(np.float64(nu), np.float64(x))
    )
    expected = float(
        _reference_log(np.float64(nu + 1.0), np.float64(x))
        - _reference_log(np.float64(nu), np.float64(x))
    )
    assert abs(accurate - expected) < abs(two_logs - expected)
    assert abs(accurate - expected) < 1e-14


def test_ratio_gradient_matches_the_analytic_identity():
    nu, x = 2.0, 4.0

    def ratio_of_x(value):
        return log_bessel_kv_ratio(jnp.float64(nu), value)[1]

    def ratio_of_nu(value):
        return log_bessel_kv_ratio(value, jnp.float64(x))[1]

    gradient_x = float(jax.grad(ratio_of_x)(jnp.float64(x)))
    c = np.exp(float(ratio_of_x(jnp.float64(x))))
    c_next = np.exp(float(log_bessel_kv_ratio(jnp.float64(nu + 1.0), jnp.float64(x))[1]))
    assert gradient_x == pytest.approx(c - c_next + 1.0 / x, rel=1e-10)

    gradient_nu = float(jax.grad(ratio_of_nu)(jnp.float64(nu)))
    h = 1e-6
    finite_difference = float(
        (ratio_of_nu(jnp.float64(nu + h)) - ratio_of_nu(jnp.float64(nu - h))) / (2 * h)
    )
    assert gradient_nu == pytest.approx(finite_difference, rel=1e-6)


def test_vmap_and_elementwise_broadcast():
    x = np.linspace(0.5, 30.0, 9)
    batched = jax.vmap(lambda value: log_bessel_kv_native(jnp.float64(1.5), value))(jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(batched), _kernel(np.full(x.size, 1.5), x), atol=0)

    nu = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    paired = np.asarray(log_bessel_kv_native(jnp.asarray(nu), jnp.asarray(x[:5])))
    np.testing.assert_allclose(paired, _kernel(nu, x[:5]), atol=0)


def test_scalar_order_with_array_argument_matches_the_oracle():
    x = np.array([1e-6, 0.3, 2.5, 9.0, 60.0, 500.0])
    got = _kernel(np.float64(3.5), x)
    expected = [
        _log_bessel_kv_scipy(np.float64(3.5), np.float64(value)) for value in x
    ]
    np.testing.assert_allclose(np.expm1(got - np.asarray(expected)), 0.0, atol=1e-12)
