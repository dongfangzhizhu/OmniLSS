"""Experimental JAX-native log K_nu (D2.267/P6-1) -- NOT the default path.

Why this module exists
----------------------
bessel.log_bessel_kv reaches scipy through jax.pure_callback, which forces a
host round trip and cuts the jit graph, so the RS inner loop of the Sichel
families cannot fuse.  This module is the native replacement, verified against
mpmath at 40 digits (worst relative error 5.7e-14 on the P6-1 grid), and kept
because it is the only path that can run without a host callback (GPU, fusion).

Why it is not enabled
--------------------
Measured on this CPU-only box, n = 40, one SI fit at max_iter=1:

    scipy callback  5.31 s
    native kernel  371.46 s   (70x slower)

Per call the two are comparable (0.45 ms vs 0.27 ms for 48 rows); the blow-up
appears only inside the fitting loop, where the kernel is evaluated per
observation on scalars and the 70-step ODE propagation is paid element by
element.  P6-1's second acceptance ("Sichel cycle time drops") therefore fails
on CPU, so the families keep the callback (fail-closed) and this module stays
opt-in:

    from omnilss.bessel_native import log_bessel_kv_native
    log_bessel_kv_native(nu, x)

Three regions, each checked against mpmath on the acceptance grid
(nu in [-5, 20], x in [1e-8, 1e3]; worst relative error 5.7e-14):

* x >= 15: Debye asymptotic with 40 terms, evaluated in log space.  K underflows
  in double for large arguments (K_20(1000) ~ 1e-436), so the log form is
  mandatory -- this is the D2.53 defect in another guise.
* 3 <= x < 15: Taylor propagation in u = ln x of y'' = (x^2 + nu^2) y
  (y = K_nu(e^u)), started from the asymptotic at x = 15.  K is the dominant
  solution as x decreases, so the propagation is stable and no series or tables
  are needed when the order is fractional.
* x < 3: the small-argument series -- K_0/K_1 closed forms plus upward recursion
  for integer orders, and (I_{-nu} - I_nu) / sin(pi nu) for fractional orders.
  Neither cancellation nor the sin(pi nu) factor bites at these arguments, and
  both branches match scipy to ~1e-15.

"""

from __future__ import annotations

from math import comb

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

# ---------------------------------------------------------------------------
# Native log K_nu
# ---------------------------------------------------------------------------

#: Terms of the Debye (1/x) expansion.  At x >= 15 the optimal truncation sits
#: near 2x = 30 terms, and 40 terms keep the error at <= 1e-14 for |nu| <= 21.
_ASYMPTOTIC_TERMS = 40
#: Taylor order of the ODE step.  With the step control below
#: (h * sqrt(g) <= 0.5) the local truncation error is (0.5)^17 / 17! ~ 8e-19,
#: far below the 1e-15 roundoff that the propagation accumulates.
_TAYLOR_ORDER = 16
#: Step cap and the stiffness factor: h = min(_MAX_STEP, _STEP_FACTOR / sqrt(g)).
_MAX_STEP = 0.05
_STEP_FACTOR = 0.5
#: Enough iterations for nu = 20 all the way from x = 15 down to x = 2 (58
#: steps); the surplus iterations are no-ops because h is clamped at zero.
_ODE_MAX_STEPS = 90
#: Series/asymptotic hand-over at x = 3: at x = 2 the ln(x/2) term of the K_0
#: series vanishes and at x = 3 it costs only 1.8 digits, while the ODE
#: roundoff grows with the distance travelled from _ASYMPTOTIC_START.
_SERIES_LIMIT = 3.0
#: Switch point between the asymptotic and the ODE propagation.
_ASYMPTOTIC_START = 15.0
#: Series terms for x < 3.  At x = 3 the K_0/K_1 term is
#: (x^2/4)^k/(k!)^2 H_k ~ 2e-13 at k = 12 and 2e-21 at k = 16, so 24 terms
#: leave a wide margin without bloating the traced graph.
_SERIES_TERMS = 24
#: Orders closer than this to an integer use the integer recursion instead of
#: the (I_{-nu} - I_nu) / sin(pi nu) form.
_INTEGER_TOLERANCE = 1e-7

_EULER_GAMMA = 0.5772156649015329


def _log_bessel_asymptotic(nu, x):
    """log K_nu(x) from the Debye expansion (x >= 15)."""
    ratio = jnp.ones_like(x)
    series = jnp.ones_like(x)
    nu2 = 4.0 * nu * nu
    for k in range(1, _ASYMPTOTIC_TERMS):
        ratio = ratio * (nu2 - (2.0 * k - 1.0) ** 2) / (8.0 * k * x)
        series = series + ratio
    return 0.5 * jnp.log(jnp.pi / (2.0 * x)) + jnp.log(series) - x


def _taylor_step(nu, u, y, yp, step):
    """One Taylor step of y'' = (e^{2u} + nu^2) y, with y' = dy/du."""
    order = _TAYLOR_ORDER
    g = jnp.exp(2.0 * u) + nu * nu
    exp2u = jnp.exp(2.0 * u)
    # derivatives of g: g^(0) = g, g^(j) = 2^j e^{2u}
    g_deriv = [g] + [(2.0**j) * exp2u for j in range(1, order + 1)]
    # y^(m+2) = sum_j C(m, j) g^(j) y^(m-j)
    y_deriv = [y, yp] + [jnp.zeros_like(y) for _ in range(order - 1)]
    for m in range(0, order - 1):
        total = jnp.zeros_like(y)
        for j in range(0, m + 1):
            total = total + comb(m, j) * g_deriv[j] * y_deriv[m - j]
        y_deriv[m + 2] = total
    factorial = 1.0
    value = y_deriv[0]
    slope = y_deriv[1]
    for m in range(1, order + 1):
        factorial *= m
        value = value + y_deriv[m] * step**m / factorial
        if m + 1 <= order:
            slope = slope + y_deriv[m + 1] * step**m / factorial
    return value, slope


def _ode_log_kv_slope(nu, x):
    """(log K_nu(x), d log K_nu / du) for 3 <= x < 15, propagated from x = 15.

    The ODE is integrated for K and for y' = dK/du together, so the log
    derivative comes for free.  That matters because the families need
    c = K_{nu+1}/K_nu = (nu - y'/y)/x, and subtracting two independently
    propagated logs is ~100x less accurate than this ratio (measured 4e-14 vs
    5e-16; D2.267).
    """
    start = _ASYMPTOTIC_START
    y = jnp.exp(_log_bessel_asymptotic(nu, jnp.full_like(x, start)))
    # y' = dy/du = nu * y - x * K_{nu+1}(x)
    yp = nu * y - start * jnp.exp(
        _log_bessel_asymptotic(nu + 1.0, jnp.full_like(x, start))
    )
    u = jnp.full_like(x, jnp.log(start))
    u_target = jnp.log(x)

    def body(_index, carry):
        u, y, yp = carry
        stiffness = jnp.sqrt(jnp.exp(2.0 * u) + nu * nu)
        step = jnp.minimum(
            jnp.minimum(_MAX_STEP, _STEP_FACTOR / stiffness), u - u_target
        )
        step = jnp.maximum(step, 0.0)  # a zero step is an exact no-op
        value, slope = _taylor_step(nu, u, y, yp, -step)
        return u - step, value, slope

    _, y, yp = lax.fori_loop(0, _ODE_MAX_STEPS, body, (u, y, yp))
    return jnp.log(jnp.abs(y)), yp / y


def _series_iv(nu, x):
    """I_nu(x) for x < 3 by the power series, carrying the Gamma sign along."""
    argument = nu + 1.0
    sign = jnp.where(argument > 0.0, 1.0, jnp.sign(jnp.sin(jnp.pi * argument)))
    term = (x / 2.0) ** nu * sign / jnp.exp(gammaln(argument))
    total = term
    for k in range(_SERIES_TERMS - 1):
        term = term * (x / 2.0) ** 2 / ((k + 1.0) * (nu + k + 1.0))
        total = total + term
    return total


def _series_kv_value(nu, x):
    """K_nu(x) for x < 3 from the small-argument series (always positive)."""
    order = jnp.abs(nu)
    log_half_x = jnp.log(x / 2.0)
    i0 = _series_iv(0.0, x)
    i1 = _series_iv(1.0, x)

    # K_0 = -(ln(x/2)+gamma) I_0 + sum_k (x^2/4)^k/(k!)^2 H_k
    # K_1 = I_0/x + (ln(x/2)+gamma) I_1 - (1/x) sum_k 2k (x^2/4)^k/(k!)^2 H_k
    harmonic = 0.0
    k0_series = jnp.zeros_like(x)
    k1_series = jnp.zeros_like(x)
    coefficient = jnp.ones_like(x)
    for k in range(1, _SERIES_TERMS):
        harmonic += 1.0 / k
        coefficient = coefficient * (x * x / 4.0) / (k * k)
        k0_series = k0_series + coefficient * harmonic
        k1_series = k1_series + 2.0 * k * coefficient * harmonic

    k0 = -(log_half_x + _EULER_GAMMA) * i0 + k0_series
    k1 = i0 / x + (log_half_x + _EULER_GAMMA) * i1 - k1_series / x

    # fractional order: K_nu = (pi/2) (I_{-nu} - I_nu) / sin(pi nu)
    fractional = jnp.abs(
        (jnp.pi / 2.0)
        * (_series_iv(-order, x) - _series_iv(order, x))
        / jnp.sin(jnp.pi * order)
    )

    # integer order: upward recursion K_{m+1} = K_{m-1} + (2m/x) K_m from K_0, K_1
    count = jnp.round(order).astype(jnp.int32)

    def recurse(m, carry):
        current, previous = carry
        updated = (2.0 * m / x) * current + previous
        take = m < count
        return jnp.where(take, updated, current), jnp.where(take, current, previous)

    current, _previous = lax.fori_loop(1, 21, recurse, (k1, k0))
    integer = jnp.where(count == 0, k0, jnp.abs(current))
    return jnp.where(jnp.abs(order - count) < _INTEGER_TOLERANCE, integer, fractional)


def _series_log_kv_slope(order, x):
    """(log K_order(x), d log K_order / du) for x < 3, order >= 0."""
    value = _series_kv_value(order, x)
    upper = _series_kv_value(order + 1.0, x)
    return jnp.log(value), order - x * upper / value


def _asymptotic_log_kv_slope(order, x):
    """(log K_order(x), d log K_order / du) from the Debye expansion."""
    term = jnp.ones_like(x)
    series = jnp.ones_like(x)
    weighted = jnp.zeros_like(x)
    nu2 = 4.0 * order * order
    for k in range(1, _ASYMPTOTIC_TERMS):
        term = term * (nu2 - (2.0 * k - 1.0) ** 2) / (8.0 * k * x)
        series = series + term
        weighted = weighted + k * term
    log_value = 0.5 * jnp.log(jnp.pi / (2.0 * x)) + jnp.log(series) - x
    # x d/dx log K = -1/2 + x S'/S - x with x S'/S = -weighted / series
    return log_value, -0.5 - x - weighted / series


def _native_log_kv_slope(nu, x):
    """(log K_{|nu|}(x), d log K_{|nu|} / du), choosing the verified region."""
    order = jnp.abs(nu)
    asymptotic_log, asymptotic_slope = _asymptotic_log_kv_slope(order, x)

    def with_ode(_operand):
        ode_log, ode_slope = _ode_log_kv_slope(order, x)
        keep = x >= _ASYMPTOTIC_START
        return jnp.where(keep, asymptotic_log, ode_log), jnp.where(
            keep, asymptotic_slope, ode_slope
        )

    def without_ode(_operand):
        return asymptotic_log, asymptotic_slope

    middle = jnp.any((x >= _SERIES_LIMIT) & (x < _ASYMPTOTIC_START))
    large_log, large_slope = lax.cond(middle, with_ode, without_ode, None)
    small = x < _SERIES_LIMIT
    small_log, small_slope = _series_log_kv_slope(order, x)
    return jnp.where(small, small_log, large_log), jnp.where(
        small, small_slope, large_slope
    )


def _native_log_kv(nu, x):
    """log K_{|nu|}(x), switching between the three verified regions.

    Computes only the value: the slope needs a second series evaluation, which
    the ratio helper pays for but the hot path should not.
    """
    nu = jnp.asarray(nu, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    order = jnp.abs(nu)
    asymptotic = _log_bessel_asymptotic(order, x)

    def with_ode(_operand):
        ode_log, _ode_slope = _ode_log_kv_slope(order, x)
        return jnp.where(x >= _ASYMPTOTIC_START, asymptotic, ode_log)

    def without_ode(_operand):
        return asymptotic

    middle = jnp.any((x >= _SERIES_LIMIT) & (x < _ASYMPTOTIC_START))
    large = lax.cond(middle, with_ode, without_ode, None)
    return jnp.where(x < _SERIES_LIMIT, jnp.log(_series_kv_value(nu, x)), large)


def _native_log_kv_ratio(nu, x):
    """(log K_nu(x), log(K_{nu+1}(x)/K_nu(x))) without differencing logs.

    For nu >= 0 the ratio is (nu - dlogK/du)/x from the same propagation that
    produced the value; for nu <= -1 the recurrence K_{m}/K_{m-1} gives the
    reciprocal of the ratio at order |nu| - 1.  The remaining band
    -1 < nu < 0 has no adjacent-order identity (|nu| and |nu+1| are both in
    (0, 1) and sum to 1), so it falls back to the difference of two logs.
    """
    nu = jnp.asarray(nu, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    order = jnp.abs(nu)
    log_value, slope = _native_log_kv_slope(order, x)
    positive = jnp.log(jnp.maximum(order - slope, 1e-300)) - jnp.log(x)

    def negative_branch(_operand):
        lower = jnp.maximum(order - 1.0, 0.0)
        _, lower_slope = _native_log_kv_slope(lower, x)
        reciprocal = -jnp.log(jnp.maximum(lower - lower_slope, 1e-300)) + jnp.log(x)
        # -1 < nu < 0: the two orders are not adjacent, so use the logs.
        shifted = _native_log_kv_slope(jnp.abs(nu + 1.0), x)[0] - log_value
        return jnp.where(nu <= -1.0, reciprocal, shifted)

    def no_negative(_operand):
        return positive

    negative = lax.cond(jnp.any(nu < 0.0), negative_branch, no_negative, None)
    return log_value, jnp.where(nu < 0.0, negative, positive)


#: Central-difference step for the *order* (nu) tangent of log K_nu / log I_nu.
#: There is no closed form for d/dnu, so the tangent is
#: (f(nu+h) - f(nu-h)) / 2h on the exact primal.
#:
#: The step is a compromise between truncation (O(h^2)) and the roundoff of the
#: primal.  For x < 3 the small-argument branch switches between the integer
#: recursion and the (I_{-nu} - I_nu)/sin(pi nu) form, and the latter's relative
#: accuracy degrades as eps/(pi*|nu-n|); at h = 1e-4 a derivative taken across
#: that switch inherits ~2e-7 (measured at nu = 3, x = 0.05).  h = 3e-4 keeps
#: both evaluation points in the fractional formula and measures <= 6e-9
#: against a Richardson-extrapolated reference over the (nu, x) grid the
#: families use -- still ~1e5 better than R's forward-difference nu slot.
_ORDER_TANGENT_STEP = 3e-4


def _log_bessel_order_tangent(primal_fn, nu, x):
    """d/dnu of a log-Bessel function, as a central difference."""
    h = _ORDER_TANGENT_STEP
    return (primal_fn(nu + h, x) - primal_fn(nu - h, x)) / (2.0 * h)


@jax.custom_jvp
def log_bessel_kv_native(nu, x):
    """Logarithm of the modified Bessel function log(K_nu(x)).

    Native JAX implementation (D2.267); see the module docstring for the three
    regions and their measured accuracy.

    Args:
        nu: Order (scalar or array).
        x: Argument (must be positive).

    Returns:
        log(K_nu(x))
    """
    return _native_log_kv(nu, x)


def _log_bessel_kv_jvp(primals, tangents):
    """Custom JVP rule for log(K_nu(x)).

    The primal is exact to ~1e-14, so a wrong tangent shows up directly in every
    AD score: the previous rule substituted the *asymptotic* tangent
    -1 - nu/(x + |nu| + 1) and dropped the nu tangent entirely, which gate 2 of
    tools/rs_family_parity_scan.py measured as mu off by 9.3e-01 (SICHEL),
    4.98e-01 (GIG) and 6.42e-01 (SI) relative (D2.44/D2.47).

    d/dx log(K_nu(x)) = K'_nu(x)/K_nu(x) = -K_{nu+1}(x)/K_nu(x) + nu/x
        (K'_nu = -(K_{nu-1} + K_{nu+1})/2 and K_{nu-1} - K_{nu+1} = -2nu K_nu/x,
        so K'_nu = -K_{nu+1} + (nu/x) K_nu)

    d/dnu log(K_nu(x)) has no closed form; it is a central difference of the
    exact primal (_ORDER_TANGENT_STEP).
    """
    nu, x = primals
    nu_dot, x_dot = tangents

    primal_out = log_bessel_kv_native(nu, x)

    eps = jnp.finfo(jnp.float64).eps
    x_safe = jnp.maximum(x, eps)
    # K_{nu+1}/K_nu as a difference of two logs.  The *density* uses the
    # slope-based log_bessel_kv_ratio instead (D2.267), but here the ratio is
    # only one term of d/dx and a 4e-14 relative error stays far inside the
    # score tolerances -- while calling the ratio helper would pull three more
    # ODE propagations into every gradient trace.
    log_ratio = log_bessel_kv_native(nu + 1.0, x_safe) - log_bessel_kv_native(
        nu, x_safe
    )
    ratio = jnp.exp(jnp.clip(log_ratio, -745.0, 709.0))
    d_dx = -ratio + nu / x_safe
    d_dnu = _log_bessel_order_tangent(log_bessel_kv_native, nu, x_safe)

    tangent_out = nu_dot * d_dnu + x_dot * d_dx
    return primal_out, tangent_out


log_bessel_kv_native.defjvp(_log_bessel_kv_jvp)


@jax.custom_jvp
def log_bessel_kv_ratio(nu, x):
    """(log K_nu(x), log K_{nu+1}(x)/K_nu(x)) without differencing logs.

    The second component is the quantity the Sichel/GIG/PIG densities need:
    they multiply it by exponents that reach ~300, so a 4e-14 error in a
    log-difference turns into a 1e-11 error in the density (measured on dGIG at
    mu=2, sigma=0.5, nu=1, x=0.005).  Computing the ratio from the ODE slope
    gives 5e-16 instead.

    Args:
        nu: Order (scalar or array, any sign).
        x: Argument (must be positive).

    Returns:
        (log K_nu(x), log(K_{nu+1}(x)/K_nu(x)))
    """
    return _native_log_kv_ratio(nu, x)


def _log_bessel_kv_ratio_jvp(primals, tangents):
    """Custom JVP for log_bessel_kv_ratio.

    d/dx log K_nu = -c + nu/x with c = K_{nu+1}/K_nu, and
    d/dx log c    = c - c1 + 1/x with c1 = K_{nu+2}/K_{nu+1} (the same identity
    one order up).  The nu tangents are central differences of the primals.
    """
    nu, x = primals
    nu_dot, x_dot = tangents

    value, ratio = log_bessel_kv_ratio(nu, x)

    eps = jnp.finfo(jnp.float64).eps
    x_safe = jnp.maximum(x, eps)
    c = jnp.exp(jnp.clip(ratio, -745.0, 709.0))
    c_next = jnp.exp(jnp.clip(log_bessel_kv_ratio(nu + 1.0, x_safe)[1], -745.0, 709.0))

    d_value_dx = nu / x_safe - c
    d_ratio_dx = c - c_next + 1.0 / x_safe

    h = _ORDER_TANGENT_STEP
    value_plus, ratio_plus = log_bessel_kv_ratio(nu + h, x_safe)
    value_minus, ratio_minus = log_bessel_kv_ratio(nu - h, x_safe)
    d_value_dnu = (value_plus - value_minus) / (2.0 * h)
    d_ratio_dnu = (ratio_plus - ratio_minus) / (2.0 * h)

    return (value, ratio), (
        nu_dot * d_value_dnu + x_dot * d_value_dx,
        nu_dot * d_ratio_dnu + x_dot * d_ratio_dx,
    )


log_bessel_kv_ratio.defjvp(_log_bessel_kv_ratio_jvp)


__all__ = ["log_bessel_kv_native", "log_bessel_kv_ratio"]
