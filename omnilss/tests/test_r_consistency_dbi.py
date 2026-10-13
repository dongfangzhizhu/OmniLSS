"""DBI (Double Binomial) dpqr parity against live R at digits=17.

What this replaces
------------------
The previous incarnation of this file compared ``DBI`` against R's **BI** --
the pre-D2.23 compatibility shell's forwarding target -- and called our
``fam.d(x, mu)`` with two arguments.  It could not have passed: the R side
raised ``x must be <= than the binomial denominator 1`` (BI's default
``bd`` is 1, the probe grid runs to 29), and DBI's ``d`` has required a
positional ``bd`` since D2.23.  A test that fails on both sides of the
comparison measures nothing, so it is rewritten here against the real
family, on the real numbers, at the precision the agreement actually has.

Methodology
-----------
d/p/q are compared element-wise against **live R** through ``digits=17``
subprocess probes (the ST1/ST2 house style), not ``RTestBus``: the bus
serialises with ``digits=10``, which alone puts a ~1e-10 relative floor
under every reference value -- coarser than the agreement being asserted.
Nothing is hard-coded; a mistyped 16-digit constant would make a parity
test pass, which is the one failure mode these tests must not have.

The three contract details a *fractional* ``x`` exposes
------------------------------------------------------
R's ``dDBI`` is a chimera at non-integer ``x`` and this family's previous
transcription rounded ``x`` up front, which hid all three:

1. ``lchoose(bd, x)`` rounds **both** arguments half-to-even (measured:
   ``lchoose(10, 1.5) == lchoose(10, 2)``), while every other term --
   ``x*logofx``, ``(bd-x)*logofbd_x``, ``(x/sigma)*log(mu)`` -- uses the
   raw ``x``.  Rounding made ``d(0.65) == d(1.0)`` exactly; live R does
   not (0.018833878244881946 vs the pmf at 1).
2. The ``|sigma - 1| < 0.001`` branch is plain ``dbinom``, which refuses a
   non-integer ``x`` (``-Inf`` on the log scale): the branch is *not* the
   general formula rounded.
3. ``pDBI`` sums over ``0:q``, and R's ``:`` operator is inclusive up to
   an absolute fuzz of ~1e-7 (measured: ``0:2.9999999`` is 0,1,2,3 while
   ``0:2.999999`` is 0,1,2), so the CDF staircase steps at
   ``floor(q + 1e-7)``, not at ``floor(q)``.

The boundary contract is the vendored 6.1-8 one where the installed 6.1.1
raises instead (the SHASHo2 convention): ``d`` returns 0 outside
``[0, bd]`` and ``p`` returns exactly 1 above ``bd``, while live 6.1.1
``stop()``s for ``x > bd`` and for ``q >= bd + 1``.  Both versions agree
everywhere inside the support, which is what the parity grid samples;
``test_boundary_contract_pins_both_r_versions`` pins the split so it
cannot silently drift.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # parity prerequisite

from omnilss.distributions import resolve_family  # noqa: E402
from omnilss.distributions_dbi import (  # noqa: E402
    _COLON_FUZZ,
    _dbi_cdf,
    _dbi_log_pdf,
    _dbi_quantile,
)

FAMILY = "DBI"
RSCRIPT = shutil.which("Rscript")

# (mu, sigma, bd) profiles: the sigma-one branch exactly (sigma = 1.0), a
# tight and a wide dispersion, an extreme mu, a small and a large bd.
# mu stays strictly inside (0, 1) -- outside it live R stops, so a parity
# grid that sampled there would measure the stop, not the density.
PROFILES = [
    dict(mu=0.50, sigma=1.2, bd=10),
    dict(mu=0.20, sigma=0.5, bd=10),
    dict(mu=0.80, sigma=2.0, bd=20),
    dict(mu=0.35, sigma=1.0, bd=10),  # exactly the dbinom branch
    dict(mu=0.95, sigma=0.7, bd=10),
    dict(mu=0.10, sigma=1.5, bd=10),
]

# Fractional responses: the only place the three contract details show.
# 9.9 sits just below bd = 10 so the largest profile still exercises it.
FRACTIONAL_X = (0.2, 0.65, 1.1, 1.55, 2.5, 3.7, 7.45, 9.9)

# R's `:` fuzz boundary points -- see the module docstring, detail 3.
FUZZ_Q = (2.9999999, 2.999999, 1.9999999, 1.99999999)

P_GRID = np.array(
    [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
)

# Measured agreement floors of the digits=17 probes, not round numbers.
RTOL = 1e-12
ATOL = 1e-300


def _x_grid(bd: int) -> np.ndarray:
    """Integers 0..bd plus the fractional points that fit below bd."""
    xs = {float(v) for v in range(0, int(bd) + 1)}
    xs.update(v for v in FRACTIONAL_X if v < bd)
    return np.array(sorted(xs))


def _run_r_raw(lines, timeout=900):
    """Run an R script, return the CompletedProcess; skip when R is absent."""
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        return subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)


def _run_r(lines, timeout=900):
    proc = _run_r_raw(lines, timeout=timeout)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout


def _rvec(values) -> str:
    return ", ".join(f"{float(v):.17g}" for v in values)


@pytest.fixture(scope="module")
def fam():
    return resolve_family(FAMILY)


@pytest.fixture(scope="module")
def r_dpqr_probe():
    """Element-wise live-R dDBI/pDBI/qDBI at digits=17 (house style)."""
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']}, sigma={p['sigma']}, bd={p['bd']}"
        xs = _x_grid(p["bd"])
        lines.append(
            f"out[['{pi}.d']] <- vapply(c({_rvec(xs)}), function(xi) "
            f"suppressWarnings(as.numeric(gamlss.dist::dDBI(xi, {args}))), "
            f"numeric(1))"
        )
        qs = sorted(set(xs.tolist()) | {v for v in FUZZ_Q if v < p["bd"]})
        lines.append(
            f"out[['{pi}.p']] <- vapply(c({_rvec(qs)}), function(qi) "
            f"suppressWarnings(as.numeric(gamlss.dist::pDBI(qi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.q']] <- vapply(c({_rvec(P_GRID)}), function(pp) "
            f"suppressWarnings(as.numeric(gamlss.dist::qDBI(pp, {args}))), "
            f"numeric(1))"
        )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


def _ours_d(fam, xs, profile):
    return np.asarray(
        fam.d(jnp.asarray(xs, dtype=jnp.float64),
              float(profile["mu"]), float(profile["sigma"]), float(profile["bd"]))
    )


# ---------------------------------------------------------------------------
# 1. Density against live R -- the fractional-x contract is the point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_d_matches_live_r_at_integer_and_fractional_x(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    xs = _x_grid(profile["bd"])
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    got = _ours_d(fam, xs, profile)
    np.testing.assert_allclose(got, r_d, rtol=RTOL, atol=ATOL)


def test_fractional_x_is_not_rounded(r_dpqr_probe, fam):
    """The defect this delivery's source fix closes.

    ``_dbi_log_pdf`` used to round ``x`` before evaluating anything, so
    ``d(0.65)`` and ``d(1.0)`` came out *identical* -- the pmf of a
    discrete distribution, evaluated at a point that is not in its
    support's integer set.  Live R keeps the raw ``x`` in every term
    except ``lchoose`` (docstring detail 1), so its two values differ by
    a factor of ~1.5.  Assert both: that they differ, and that each is
    R's.
    """
    profile = PROFILES[0]  # mu = 0.5, sigma = 1.2, bd = 10
    xs = _x_grid(profile["bd"])
    r_d = np.asarray(r_dpqr_probe["0.d"], dtype=float)
    got = _ours_d(fam, xs, profile)

    i_065 = int(np.where(xs == 0.65)[0][0])
    i_100 = int(np.where(xs == 1.0)[0][0])
    assert abs(got[i_065] - got[i_100]) > 1e-6, (
        "d(0.65) == d(1.0) again -- x is being rounded somewhere"
    )
    assert got[i_065] == pytest.approx(r_d[i_065], rel=RTOL)
    assert got[i_100] == pytest.approx(r_d[i_100], rel=RTOL)
    # And the pair really is lchoose-at-integer / rest-at-raw, i.e. the
    # chimera, not one or the other convention.
    assert 0.015 < got[i_065] < 0.025, got[i_065]


def test_dbinom_branch_refuses_fractional_x(r_dpqr_probe, fam):
    """Docstring detail 2: the sigma-one branch is dbinom, not a rounding.

    R's ``dbinom`` returns 0 for a non-integer ``x`` (``-Inf`` logged), so
    inside the ``|sigma - 1| < 0.001`` window the density is *identically
    zero* at 0.2 / 0.65 / 1.55 while remaining positive at the integers.
    An implementation that evaluates the general formula there (the
    obvious reading of "the branch is just a shortcut") produces the
    tilted pmf instead -- a family that is not R's at sigma = 1.
    """
    profile = PROFILES[3]  # sigma = 1.0 exactly
    xs = _x_grid(profile["bd"])
    r_d = np.asarray(r_dpqr_probe["3.d"], dtype=float)
    got = _ours_d(fam, xs, profile)

    for x in FRACTIONAL_X:
        if x >= profile["bd"]:
            continue
        i = int(np.where(xs == x)[0][0])
        assert got[i] == 0.0, f"d({x}) at sigma=1 must be 0, got {got[i]}"
        assert r_d[i] == 0.0, f"live R d({x}) at sigma=1 is {r_d[i]}"
    # The integers are still a genuine pmf in the same call.
    i_two = int(np.where(xs == 2.0)[0][0])
    assert got[i_two] > 0.0 and got[i_two] == pytest.approx(r_d[i_two], rel=RTOL)


def test_d_outside_the_support_is_zero(fam):
    """Docstring detail 3: ``fy[x < 0] <- 0; fy[x > bd] <- 0`` (vendored)."""
    profile = PROFILES[0]
    for x in (-0.5, -1.0, 10.5, 11.0, 25.0):
        got = float(fam.d(jnp.asarray(x), float(profile["mu"]),
                          float(profile["sigma"]), float(profile["bd"])))
        assert got == 0.0, f"d({x}) outside [0, bd] must be 0, got {got}"
    # ...and the endpoints are inside it.
    for x in (0.0, 10.0):
        got = float(fam.d(jnp.asarray(x), float(profile["mu"]),
                          float(profile["sigma"]), float(profile["bd"])))
        assert got > 0.0, f"d({x}) is inside the support and must be positive"


def test_live_r_stops_for_x_above_bd():
    """Pin the version split so it cannot silently drift.

    The installed 6.1.1 has ``if (any(bd < x)) stop(...)`` active; the
    vendored 6.1-8 comments it out and zeroes the density instead.  We
    follow vendored (see the module docstring), so the divergence is
    pinned here rather than hidden.
    """
    proc = _run_r_raw([
        "suppressMessages(library(gamlss.dist))",
        "gamlss.dist::dDBI(10.5, mu = 0.5, sigma = 1.2, bd = 10)",
        "cat('NO ERROR\\n')",
    ])
    assert proc.returncode != 0
    assert "must be <=  than the binomial denominator" in proc.stderr, (
        proc.stderr[-300:]
    )
    assert "NO ERROR" not in proc.stdout


# ---------------------------------------------------------------------------
# 2. CDF against live R -- the 0:q truncation and its fuzz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    xs = _x_grid(profile["bd"])
    qs = sorted(set(xs.tolist()) | {v for v in FUZZ_Q if v < profile["bd"]})
    qs = np.array(qs)
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    got = np.asarray(
        fam.p(jnp.asarray(qs, dtype=jnp.float64),
              float(profile["mu"]), float(profile["sigma"]), float(profile["bd"]))
    )
    np.testing.assert_allclose(got, r_p, rtol=RTOL, atol=ATOL)


def test_p_steps_at_the_colon_fuzz(r_dpqr_probe, fam):
    """Docstring detail 3: R's ``:`` is inclusive up to ~1e-7.

    ``pDBI`` sums over ``0:q``, and ``0:2.9999999`` is ``0,1,2,3`` while
    ``0:2.999999`` is ``0,1,2``.  The old implementation rounded ``q``
    (half-to-even), which put the staircase one step off in the other
    direction for *every* non-integer q: ``p(0.714)`` returned
    ``p(1.2286)``'s value.  Live R's grid shows the ×4/×3/×1 pattern of a
    floor; this pins the fuzz boundary on top of the floor.
    """
    profile = PROFILES[0]
    mu, sg, bd = (float(profile["mu"]), float(profile["sigma"]),
                  float(profile["bd"]))
    p_at = lambda q: float(fam.p(jnp.asarray(q), mu, sg, bd))

    assert _COLON_FUZZ == 1e-7
    assert p_at(2.9999999) == pytest.approx(p_at(3.0), rel=RTOL)
    assert p_at(2.999999) == pytest.approx(p_at(2.0), rel=RTOL)
    assert p_at(2.5) == pytest.approx(p_at(2.0), rel=RTOL)
    assert p_at(1.9999999) == pytest.approx(p_at(2.0), rel=RTOL)
    # The two sides of the fuzz must be observably different, or this
    # whole discussion is about an inert switch (D2.18 criterion 4).
    assert abs(p_at(2.9999999) - p_at(2.999999)) > 1e-3
    # And both must be live R's -- the probe grid contains these points.
    r_p = np.asarray(r_dpqr_probe["0.p"], dtype=float)
    qs = sorted(set(_x_grid(profile["bd"]).tolist())
                | {v for v in FUZZ_Q if v < profile["bd"]})
    for q in (2.9999999, 2.999999, 1.9999999):
        i = qs.index(q)
        assert p_at(q) == pytest.approx(float(r_p[i]), rel=RTOL)


def test_p_boundary_contract(fam):
    """``cdf[q < 0] <- 0; cdf[q > bd] <- 1`` (vendored 6.1-8)."""
    profile = PROFILES[0]
    mu, sg, bd = (float(profile["mu"]), float(profile["sigma"]),
                  float(profile["bd"]))
    assert float(fam.p(jnp.asarray(-0.5), mu, sg, bd)) == 0.0
    assert float(fam.p(jnp.asarray(-3.0), mu, sg, bd)) == 0.0
    assert float(fam.p(jnp.asarray(10.5), mu, sg, bd)) == 1.0
    assert float(fam.p(jnp.asarray(11.0), mu, sg, bd)) == 1.0
    # q == bd itself is *inside* the support: the raw sum, ~1 - 1 ULP.
    at_bd = float(fam.p(jnp.asarray(10.0), mu, sg, bd))
    assert at_bd == pytest.approx(1.0, abs=1e-14)
    assert at_bd < 1.0


def test_live_r_returns_the_raw_sum_just_above_bd():
    """Pin the other side of the p split.

    Live 6.1.1 has no ``cdf[q > bd] <- 1`` line: for ``q`` in
    ``(bd, bd + 1)`` it returns the accumulated sum -- measured
    0.99999999999999967 where we return 1.0 (3.3e-16 absolute) -- and for
    ``q >= bd + 1`` the ``0:q`` sequence finally steps past ``bd`` and
    ``dDBI`` stops.  Both sides pinned so neither can drift.
    """
    out = _run_r([
        "suppressMessages(library(gamlss.dist))",
        "options(digits = 17)",
        "cat(sprintf('%.17g\\n', gamlss.dist::pDBI(10.5, mu = 0.5, "
        "sigma = 1.2, bd = 10)))",
        "r <- tryCatch(gamlss.dist::pDBI(11, mu = 0.5, sigma = 1.2, "
        "bd = 10), error = function(e) 'STOP')",
        "cat(r, '\\n')",
    ]).strip().splitlines()
    assert out[0] == "0.99999999999999967", out
    assert out[1] == "STOP", out


def test_p_is_a_step_function_that_sums_the_pmf(fam):
    """Independent of R: the CDF is the running sum of the pmf."""
    profile = PROFILES[1]  # mu = 0.2, sigma = 0.5, bd = 10
    mu, sg, bd = (float(profile["mu"]), float(profile["sigma"]),
                  float(profile["bd"]))
    xs = jnp.arange(0, 11, dtype=jnp.float64)
    pmf = np.exp(np.asarray(_dbi_log_pdf(xs, mu, sg, bd), dtype=float))
    cdf = np.cumsum(pmf)
    for j in range(11):
        got = float(fam.p(jnp.asarray(float(j)), mu, sg, bd))
        assert got == pytest.approx(cdf[j], rel=1e-13), (
            f"p({j}) = {got} but cumsum(pmf)[{j}] = {cdf[j]}"
        )


# ---------------------------------------------------------------------------
# 3. Quantile against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_q_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_q = np.asarray(r_dpqr_probe[f"{pi}.q"], dtype=float)
    got = np.asarray(
        fam.q(jnp.asarray(P_GRID, dtype=jnp.float64),
              float(profile["mu"]), float(profile["sigma"]), float(profile["bd"]))
    )
    np.testing.assert_allclose(got, r_q, rtol=RTOL, atol=ATOL)


def test_q_inverts_p(fam):
    """Independent of R: ``q(p(q)) == q`` on the integers, both ways."""
    profile = PROFILES[2]  # mu = 0.8, sigma = 2.0, bd = 20
    mu, sg, bd = (float(profile["mu"]), float(profile["sigma"]),
                  float(profile["bd"]))
    for j in range(0, 21, 2):
        p_at_j = float(fam.p(jnp.asarray(float(j)), mu, sg, bd))
        back = float(fam.q(jnp.asarray(p_at_j), mu, sg, bd))
        assert back == float(j), f"q(p({j})) = {back}"
    # p(q(p)) == p for interior probabilities -- up to the step, which is
    # all a discrete distribution can give: q(p) is the *smallest* j with
    # CDF(j) >= p, so CDF(q(p)) >= p and CDF(q(p) - 1) < p.
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        q_at_p = float(fam.q(jnp.asarray(p), mu, sg, bd))
        back = float(fam.p(jnp.asarray(q_at_p), mu, sg, bd))
        assert back >= p, f"p(q({p})) = {back} < {p}"
        if q_at_p > 0:
            below = float(fam.p(jnp.asarray(q_at_p - 1.0), mu, sg, bd))
            assert below < p, f"p(q({p}) - 1) = {below} >= {p}"


def test_q_boundaries_are_bd_and_zero(fam):
    """``p = 1`` hits R's own ``p + 1e-09 >= 1`` branch; ``p = 0`` is the
    first ``j`` whose CDF is already reached."""
    profile = PROFILES[0]
    mu, sg, bd = (float(profile["mu"]), float(profile["sigma"]),
                  float(profile["bd"]))
    assert float(fam.q(jnp.asarray(1.0), mu, sg, bd)) == bd
    assert float(fam.q(jnp.asarray(0.0), mu, sg, bd)) == 0.0
    # The loop's own tolerance, not the boundary branch: a p just below 1
    # still has to land on the last integer whose CDF reaches it.
    last = float(fam.p(jnp.asarray(9.0), mu, sg, bd))
    assert float(fam.q(jnp.asarray(last), mu, sg, bd)) == 9.0


# ---------------------------------------------------------------------------
# 4. The deviance identity the fitter consumes
# ---------------------------------------------------------------------------


def test_g_dev_inc_is_minus_two_log_d(fam):
    """``DBI()$G.dev.incr = -2 * dDBI(y, mu, sigma, bd, log = TRUE)``.

    Asserted in log space: at the extreme profile the pmf underflows to
    exactly 0, where ``-2 * d`` is 0 and the identity is meaningless.
    """
    profile = PROFILES[4]  # mu = 0.95: the pmf is concentrated at 9, 10
    mu, sg, bd = (float(profile["mu"]), float(profile["sigma"]),
                  float(profile["bd"]))
    for y in (0.0, 3.0, 7.0, 9.0, 10.0):
        lp = float(_dbi_log_pdf(jnp.asarray(y), mu, sg, bd))
        inc = float(np.asarray(fam.g_dev_inc(y=y, mu=mu, sigma=sg, bd=bd)))
        assert inc == pytest.approx(-2.0 * lp, rel=1e-13, abs=1e-12), (
            f"y={y}: g_dev_inc {inc} != -2*log d {-2.0 * lp}"
        )


def test_mixed_bd_cdf_does_not_double_count(fam):
    """One call, two denominators: each observation sums only to its bd.

    The previous implementation capped the sum at the *global* maximum
    ``bd``, so an observation with ``bd = 2`` asked about ``q = 5`` summed
    ``d(0) + ... + d(5)`` -- and since ``_dbi_log_pdf`` clipped ``x`` into
    ``[0, bd]``, that double-counted ``d(2)``.  Per-observation results
    must equal the per-element calls.
    """
    mu = jnp.array([0.4, 0.6])
    sg = jnp.array([1.2, 0.8])
    bd = jnp.array([2.0, 10.0])
    q = jnp.array([5.0, 5.0])
    together = np.asarray(_dbi_cdf(q, mu, sg, bd), dtype=float)
    for i in (0, 1):
        alone = float(_dbi_cdf(q[i], mu[i], sg[i], bd[i]))
        assert together[i] == pytest.approx(alone, rel=1e-14), (
            f"obs {i}: together {together[i]} != alone {alone}"
        )
    # q = 5 > bd = 2 for the first observation: the support cap, not a sum.
    assert together[0] == 1.0
    assert together[1] < 1.0


def test_scan_calling_convention_is_what_r_sees():
    """The dpqr scan must pass fixed parameters -- the D2.41 gate fix.

    ``DBI``'s ``d``/``p``/``q`` take ``bd`` positionally (R's
    ``dDBI(x, mu, sigma, bd)`` has no default), so a scan that builds its
    argument list from ``fam.parameters`` alone gets a ``TypeError`` that
    reads as a family defect.  Pin the family's shape here so the two
    facts stay together.
    """
    fam = resolve_family(FAMILY)
    assert fam.parameters == ("mu", "sigma")
    assert fam.fixed_parameters == ("bd",)
    with pytest.raises(TypeError):
        fam.d(jnp.asarray(1.0), 0.5, 1.2)
    # ...and with bd it works, at the values the scan passes.
    xs = np.linspace(0.2, 2.0, 5)
    d = np.asarray(fam.d(jnp.asarray(xs), 0.5, 1.2, 10.0))
    assert np.all(np.isfinite(d))
