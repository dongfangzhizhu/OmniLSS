"""The log-Bessel JVP rules, checked against three independent references.

Until D2.47 both custom JVPs used asymptotic tangents -- ``-1 - nu/(x+|nu|+1)``
for K and ``1 - nu/(x+|nu|+1)`` for I -- and dropped the order (``nu``)
tangent entirely.  The primals come from scipy and were exact, so every
density was right while every AD score was wrong: gate 2 measured mu off by
9.3e-01 (SICHEL), 4.98e-01 (GIG) and 6.42e-01 (SI) relative, and D2.44 had to
hand SICHEL's families R's closed-form slots as a workaround.

References used here, deliberately none of them the rule being tested:

1. a Richardson-extrapolated central difference of the *scipy primal* (the
   JVP never enters);
2. closed forms at ``nu = ±1/2``, where
   ``K_{1/2}(x) = sqrt(pi/(2x)) e^{-x}`` and
   ``I_{1/2}(x) = sqrt(2/(pi x)) sinh(x)``;
3. live R's SICHEL derivative slots at digits=17 -- R's own closed forms for
   mu/sigma, R's forward difference for nu.

The third one is the delivery-level acceptance test: SICHEL's AD scores are
computed here *without* the D2.44 overrides, so they exercise the JVP
directly.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # parity prerequisite
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omnilss.bessel import (  # noqa: E402
    _log_bessel_kv_scipy,
    log_bessel_iv,
    log_bessel_kv,
)
from omnilss.distributions_b6 import _sichel_log_pdf  # noqa: E402

RSCRIPT = shutil.which("Rscript")

#: Per-(family, parameter) tolerance for the score-vs-central-difference
#: self-consistency check below, mirroring the parity scanner's
#: ``_DERIV_TOLERANCE`` discipline: where R's own slot is a coarse finite
#: difference, the true derivative of our density is a *different* number and
#: the tight default would fail on R's discretisation, not ours.
#: GIG nu (D2.48): R's dGIG nu slot is gamlss's ``numeric.deriv`` forward
#: difference with delta=0.01, transcribed verbatim (bit-level 9.7e-12 vs live
#: R at digits=17).  On this test's grid the central difference sees the true
#: derivative, measured gap 1.4687e-02 relative / 5.4568e-03 absolute.
_SELF_CONSISTENCY_TOLERANCE = {
    ("GIG", "nu"): (2e-2, 6e-3),
}

#: (nu, x) grid spanning the order and argument ranges the families use:
#: SICHEL/PIG evaluate log K_nu at 1/sigma and at alpha, GIG at
#: sqrt(chi*psi) with nu in (-2, 2).
NU_X_GRID = [
    (-0.5, 1.0),
    (0.0, 0.5),
    (0.3, 0.5),
    (1.0, 5.0),
    (2.0, 0.1),
    (3.0, 0.05),
    (-1.5, 2.0),
    (0.5, 1.0),
]

#: SICHEL profiles and their R slot dumps (D2.44 probe layout).
PROFILES = [
    dict(mu=1.0, sigma=0.5, nu=-0.5),
    dict(mu=2.0, sigma=1.2, nu=0.3),
    dict(mu=0.7, sigma=0.3, nu=-1.5),
    dict(mu=3.0, sigma=2.0, nu=2.0),
]
Y_GRID = np.arange(10, dtype=float)


def _run_r(lines, timeout=900):
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        proc = subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout


def _richardson(fn, value, h=1e-4):
    """Richardson-extrapolated central difference of a scalar function."""
    d1 = (fn(value + h) - fn(value - h)) / (2.0 * h)
    d2 = (fn(value + h / 2.0) - fn(value - h / 2.0)) / h
    return (4.0 * d2 - d1) / 3.0


@pytest.fixture(scope="module")
def r_sichel_slots():
    """SICHEL's dldm/dldd/dldv from live R, digits=17."""
    lines = ["suppressMessages(library(gamlss.dist))",
             "o <- SICHEL()", "out <- list()"]
    ys = ", ".join(str(int(v)) for v in Y_GRID)
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']!r}, sigma={p['sigma']!r}, nu={p['nu']!r}"
        for slot in ("dldm", "dldd", "dldv"):
            lines.append(
                f"out[['{pi}.{slot}']] <- vapply(c({ys}), function(y) "
                f"as.numeric(o[['{slot}']](y, {args})), numeric(1))"
            )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


# ---------------------------------------------------------------------------
# 1. Closed forms at order ±1/2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [-0.5, 0.5])
@pytest.mark.parametrize("x", [0.25, 1.0, 4.0])
def test_k_order_half_closed_form(nu, x):
    """K_{±1/2}(x) = sqrt(pi/(2x)) e^{-x} -- so d/dx log K = -1/(2x) - 1."""
    got = float(jax.grad(log_bessel_kv, argnums=1)(jnp.float64(nu),
                                                  jnp.float64(x)))
    expected = -1.0 / (2.0 * x) - 1.0
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_i_order_half_closed_form():
    """I_{1/2}(x) = sqrt(2/(pi x)) sinh(x) -- d/dx log I = coth(x) - 1/(2x)."""
    for x in (0.25, 1.0, 4.0):
        got = float(jax.grad(log_bessel_iv, argnums=1)(jnp.float64(0.5),
                                                       jnp.float64(x)))
        expected = 1.0 / math.tanh(x) - 1.0 / (2.0 * x)
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_second_derivative_matches_closed_form():
    """d²/dx² log K_{1/2} = 1/(2x²); the JVP must survive being differentiated."""
    x = 1.0
    hess = jax.hessian(log_bessel_kv, argnums=(0, 1))(
        jnp.float64(0.5), jnp.float64(x))
    assert float(hess[1][1]) == pytest.approx(0.5 / x ** 2, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Both partials against the scipy primal (no JVP involved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu,x", NU_X_GRID)
def test_dx_matches_richardson_reference(nu, x):
    """dx against a Richardson-extrapolated difference of the exact primal.

    Measured agreement <= 3.6e-11 relative across the grid; the tolerance
    leaves room for the reference's own extrapolation error rather than for
    ours.
    """
    got = float(jax.grad(log_bessel_kv, argnums=1)(jnp.float64(nu),
                                                   jnp.float64(x)))
    ref = _richardson(lambda v: float(_log_bessel_kv_scipy(nu, v)), x)
    assert got == pytest.approx(ref, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("nu,x", NU_X_GRID)
def test_dnu_matches_richardson_reference(nu, x):
    """The order tangent against the same flavour of reference.

    The rule itself is a central difference (``_ORDER_TANGENT_STEP`` = 1e-5);
    measured agreement is <= 3e-9 relative.  The 1e-7 tolerance is two orders
    above that, which is still three orders below R's forward-difference
    error in the SICHEL slot (5.5e-3 .. 1.0e-2, below).
    """
    got = float(jax.grad(log_bessel_kv, argnums=0)(jnp.float64(nu),
                                                   jnp.float64(x)))
    ref = _richardson(lambda v: float(_log_bessel_kv_scipy(v, x)), nu)
    assert got == pytest.approx(ref, rel=1e-7, abs=1e-10)


def test_asymptotic_tangent_is_not_reproduced():
    """The old rule is pinned out: at (nu=2, x=0.1) it is off by ~18.

    Old tangent: -1 - nu/(x + |nu| + 1) = -1.952.  True: -20.049.  A revert
    to the asymptotic form cannot pass this test even loosely.
    """
    nu, x = 2.0, 0.1
    got = float(jax.grad(log_bessel_kv, argnums=1)(jnp.float64(nu),
                                                   jnp.float64(x)))
    old = -1.0 - nu / (x + abs(nu) + 1.0)
    assert abs(got - old) > 10.0
    assert got == pytest.approx(-20.049391724, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Delivery-level acceptance: SICHEL AD scores (no overrides) vs live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_sichel_ad_scores_match_r_closed_forms(r_sichel_slots, pi):
    """AD of the density against R's own slots -- overrides deliberately off.

    R's ``dldm``/``dldd`` are closed forms, so this compares two independent
    evaluations of the same derivative: measured <= 1.3e-12 relative across
    the four profiles.  Before D2.47 the same comparison was off by 0.55-1.27.
    """
    p = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    for idx, slot, rtol in ((1, "dldm", 1e-10), (2, "dldd", 1e-10)):
        ad = jax.vmap(jax.grad(_sichel_log_pdf, argnums=idx),
                      in_axes=(0, None, None, None))(
            y, p["mu"], p["sigma"], p["nu"])
        r_vals = np.asarray(r_sichel_slots[f"{pi}.{slot}"], dtype=float)
        np.testing.assert_allclose(np.asarray(ad), r_vals, rtol=rtol,
                                   atol=1e-300)


@pytest.mark.parametrize("pi", [0, 3])
def test_sichel_ad_nu_score_beats_r_forward_difference(r_sichel_slots, pi):
    """The one slot where R is the less accurate of the two.

    R has no analytic nu derivative and uses a forward difference with
    ``delta = 1e-3``; our JVP uses a central difference with 1e-5.  Against a
    Richardson reference on the same density:

        AD vs true         <= 2.9e-6 relative
        R forward vs true  5.7e-3 .. 1.0e-2 relative

    so the ~1e-2 gap between our score and R's is R's truncation error, not
    ours.  Both halves are asserted.
    """
    p = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    ad = np.asarray(jax.vmap(jax.grad(_sichel_log_pdf, argnums=3),
                             in_axes=(0, None, None, None))(
        y, p["mu"], p["sigma"], p["nu"]))

    def log_density_at_nu(nu_value):
        return np.asarray(jax.vmap(
            lambda yy: _sichel_log_pdf(yy, p["mu"], p["sigma"], nu_value)
        )(y), dtype=float)

    h = 1e-4
    ref = ((4.0 * (log_density_at_nu(p["nu"] + h / 2)
                   - log_density_at_nu(p["nu"] - h / 2)) / h)
           - (log_density_at_nu(p["nu"] + h)
              - log_density_at_nu(p["nu"] - h)) / (2 * h)) / 3.0
    np.testing.assert_allclose(ad, ref, rtol=1e-4, atol=1e-8)

    r_vals = np.asarray(r_sichel_slots[f"{pi}.dldv"], dtype=float)
    gap = np.max(np.abs(r_vals - ref) / np.maximum(np.abs(ref), 1e-8))
    assert gap > 1e-3, gap


# ---------------------------------------------------------------------------
# 4. The two families gate 2 was failing on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["SI", "GIG", "PIG"])
def test_family_scores_match_their_own_density(family):
    """Self-consistency for the Bessel families whose gate 2 was red.

    This is what gate 2 measures: each score against a central difference of
    ``log(fam.d)``.  It says nothing about whether the density is right (that
    is gate 1's job), only that the scores differentiate it -- which is
    exactly what the broken JVP broke.
    """
    from omnilss.distribution_registry import resolve

    fam = resolve(family)
    # A parameter point the scanner's own calibration accepts.  SI carries a
    # third (identity-linked) shape parameter; GIG is continuous, so its grid
    # starts at 1 rather than 0.
    spec = {
        "SI": ((2.0, 1.0, -0.5), jnp.asarray(np.arange(6, dtype=float))),
        "GIG": ((1.0, 1.0, 0.5), jnp.asarray(np.arange(1, 7, dtype=float))),
        "PIG": ((2.0, 1.0), jnp.asarray(np.arange(6, dtype=float))),
    }
    base, y = spec[family]
    assert len(base) == len(fam.parameters), (family, fam.parameters)
    h = 1e-5
    for k, name in enumerate(fam.parameters):
        fn = fam.score_functions[name]

        def log_density_at(value):
            args = list(base)
            args[k] = value
            return jnp.log(fam.d(y, *args))

        plus = np.asarray(log_density_at(base[k] + h))
        minus = np.asarray(log_density_at(base[k] - h))
        fd = (plus - minus) / (2 * h)
        score = np.asarray(fn(y, *base))
        # GIG nu is R's own numeric.deriv forward difference (delta=0.01), so
        # its truncation error shows up here as a 1.47e-02 relative gap against
        # the true derivative of our density -- the same disposition the
        # parity scanner registers at _DERIV_TOLERANCE[("GIG","nu")].  mu and
        # sigma stay at the default tight tolerance, which is what pins the
        # D2.47 JVP fix.
        rtol, atol = _SELF_CONSISTENCY_TOLERANCE.get((family, name), (1e-4, 1e-6))
        np.testing.assert_allclose(score, fd, rtol=rtol, atol=atol,
                                   err_msg=f"{family}:{name}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
