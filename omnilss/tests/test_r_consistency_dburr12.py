"""DBURR12 (discretised Burr XII) parity against live R at digits=17.

What this replaces
------------------
The previous incarnation of this file compared our DBURR12 against R's
*SICHEL* -- it asserted the compatibility alias, not the distribution.  That
alias was wrong: DBURR12 is the discretised Burr XII,

    Sur(y) = (1 + (y/mu)^sigma)^(-nu),      f(y) = Sur(y) - Sur(y+1)

with ``nu > 0`` and all three links ``log``, while SICHEL accepts any real
``nu`` and links ``nu`` with ``identity``.  Measured at ``(5, 2, 2)``:
``dDBURR12(0..3) = 0.07544, 0.18139, 0.20251, 0.16885`` versus
``dSICHEL(0..3) = 0.01323, 0.02949, 0.04538, 0.05833``.  The parity scanner
had been skipping the family because the shell inherited SICHEL's
``nu.link = identity``, so its probe grid contained negative ``nu`` and live
R refused it (D2.44 finding); declaring R's own links is what makes gate 1
probe ``nu > 0`` at all.

Install-versus-vendored splits, all measured
--------------------------------------------
* ``qDBURR12``: vendored 6.1-8 is ``trunc(w - 1)`` with ``0/Inf/NaN``
  boundary *assignments*; the installed 6.1.1 stops for any ``p`` outside
  ``(0, 1)`` and evaluates ``round(w - 1)`` with ``ifelse(q < 0, 0, q)``.
  We follow the installed body -- that is what runs inside a real
  ``gamlss`` fit -- and take the house boundary convention, pinning R's
  stop.  ``round`` is half-to-even, and it is *not* the CDF inverse:
  ``p = 0.5`` at the default profile gives ``w - 1 = 2.218`` -> 2, while
  ``F(2) = 0.4593 < 0.5``.
* ``dDBURR12`` with ``log = TRUE`` clamps ``x < 0`` to ``0`` *after* the log
  (R's logged density at a negative count is ``0``); ours is ``-inf``.
* ``pDBURR12`` clamps ``q < 0`` to ``0`` after the upper-tail flip, so R's
  survival at ``q < 0`` is ``0`` where it should be ``1``; ours is ``1``.
* Non-integer counts: R evaluates the continuous difference, we return 0
  (off a discrete support).

Methodology
-----------
d/p/q and all six diagonal slots are compared element-wise against **live R**
through ``digits=17`` subprocess probes.  Nothing is hard-coded; a mistyped
constant cannot make these tests pass.  The end-to-end fit uses one embedded
digits=17 sample so both sides provably see the same numbers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # parity prerequisite
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.distributions_discrete_compat import (  # noqa: E402
    _DBURR12_HESSIANS,
    _DBURR12_SCORES,
    _dburr12_log_pdf,
    dDBURR12,
    pDBURR12,
    qDBURR12,
)

FAMILY = "DBURR12"
RSCRIPT = shutil.which("Rscript")

#: (mu, sigma, nu) profiles.  R's own defaults first; the rest span small
#: mu / large nu (a very short tail) and large mu / small nu (a long one).
PROFILES = [
    dict(mu=5.0, sigma=2.0, nu=2.0),
    dict(mu=2.0, sigma=0.5, nu=1.5),
    dict(mu=10.0, sigma=3.0, nu=0.8),
    dict(mu=0.7, sigma=1.2, nu=4.0),
]

#: Counts 0..30 -- y = 0 (the support edge) and a point deep in the tail.
Y_GRID = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 9.0, 15.0, 30.0])

P_GRID = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

#: Measured agreement floors of the digits=17 probes: d <= 8.3e-14,
#: p <= 8.3e-14, slots <= 2.2e-13 (profile 3, the largest parameters).
RTOL = 1e-12
ATOL = 1e-300

#: One embedded digits=17 sample (live R: set.seed(20261026);
#: rDBURR12(30, mu = 5, sigma = 2, nu = 2)).  Embedded rather than
#: regenerated so both fitters provably see the same numbers.
FIT_Y = np.array([
    5.0, 2.0, 2.0, 6.0, 9.0, 2.0, 2.0, 1.0, 2.0, 2.0,
    1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0,
    1.0, 2.0, 0.0, 5.0, 2.0, 2.0, 1.0, 1.0, 9.0, 1.0,
])

# Same RS control on both sides (R's own default c.crit = 0.001).
R_FIT_CONTROL = "gamlss.control(n.cyc = 200, c.crit = 0.001, trace = FALSE)"


def _run_r(lines, timeout=900):
    """Run an R script, return stdout; skip the test when R is absent."""
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


def _rvec(values) -> str:
    return ", ".join(f"{float(v):.17g}" for v in values)


@pytest.fixture(scope="module")
def r_dpqr_probe():
    """d/p/q at every profile, digits=17."""
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']!r}, sigma={p['sigma']!r}, nu={p['nu']!r}"
        lines += [
            f"out[['{pi}.d']] <- vapply(c({_rvec(Y_GRID)}), function(x) "
            f"as.numeric(dDBURR12(x, {args})), numeric(1))",
            f"out[['{pi}.dlog']] <- vapply(c({_rvec(Y_GRID)}), function(x) "
            f"as.numeric(dDBURR12(x, {args}, log=TRUE)), numeric(1))",
            f"out[['{pi}.p']] <- vapply(c({_rvec(Y_GRID)}), function(q) "
            f"as.numeric(pDBURR12(q, {args})), numeric(1))",
            f"out[['{pi}.q']] <- vapply(c({_rvec(P_GRID)}), function(pp) "
            f"as.numeric(qDBURR12(pp, {args})), numeric(1))",
        ]
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_slot_probe():
    """R's own six diagonal slots at every profile, digits=17."""
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "o <- DBURR12()",
        "out <- list()",
    ]
    slots = ("dldm", "dldd", "dldv", "d2ldm2", "d2ldd2", "d2ldv2")
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']!r}, sigma={p['sigma']!r}, nu={p['nu']!r}"
        for slot in slots:
            lines.append(
                f"out[['{pi}.{slot}']] <- vapply(c({_rvec(Y_GRID)}), "
                f"function(y) as.numeric(o[['{slot}']](y, {args})), numeric(1))"
            )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_family_probe():
    """Links, type and R's initial-value expressions."""
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "o <- DBURR12()",
        "out <- list(",
        "  parameters = paste(names(o$parameters), o$parameters, collapse=','),",
        "  mu_link = o$mu.link, sigma_link = o$sigma.link, nu_link = o$nu.link,",
        "  type = o$type,",
        "  mu_initial = deparse(o$mu.initial),",
        "  sigma_initial = deparse(o$sigma.initial),",
        "  nu_initial = deparse(o$nu.initial)",
        ")",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_boundary_probe():
    """The boundaries R handles differently: stops, log-scale clamps."""
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    # q stops for any p outside (0, 1).
    for p in (0.0, 1.0, -0.5, 1.5):
        lines.append(
            f"out[['q_stop_{p}']] <- tryCatch("
            f"{{ qDBURR12({p!r}, 5, 2, 2); 'no stop' }}, "
            "error = function(e) conditionMessage(e))"
        )
    # d on the log scale below the support: clamped to 0 after the log.
    lines.append("out[['dlog_neg']] <- dDBURR12(-1, 5, 2, 2, log=TRUE)")
    lines.append("out[['d_neg']] <- dDBURR12(-1, 5, 2, 2)")
    # p below the support, with and without the tail / log transforms.
    lines.append("out[['p_neg']] <- pDBURR12(-1, 5, 2, 2)")
    lines.append("out[['p_neg_upper']] <- pDBURR12(-1, 5, 2, 2, lower.tail=FALSE)")
    lines.append("out[['p_neg_log']] <- pDBURR12(-1, 5, 2, 2, log.p=TRUE)")
    # Non-integer count: R evaluates the continuous difference.
    lines.append("out[['d_half']] <- dDBURR12(1.5, 5, 2, 2)")
    # Invalid nu: R stops.
    lines.append(
        "out[['d_nu0']] <- tryCatch("
        "{ dDBURR12(2, 5, 2, 0); 'no stop' }, "
        "error = function(e) conditionMessage(e))"
    )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_fit_probe():
    """Live-R fit on the embedded sample."""
    lines = [
        "suppressMessages({library(gamlss); library(gamlss.dist)})",
        "ctl <- " + R_FIT_CONTROL,
        f"y <- c({_rvec(FIT_Y)})",
        "m <- gamlss(y ~ 1, family = DBURR12, control = ctl)",
        "out <- list(",
        "  deviance = m$G.deviance, converged = isTRUE(m$converged),",
        "  mu = as.numeric(coef(m, what='mu')),",
        "  sigma = as.numeric(coef(m, what='sigma')),",
        "  nu = as.numeric(coef(m, what='nu'))",
        ")",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return json.loads(_run_r(lines).strip())


# ---------------------------------------------------------------------------
# 1. d/p/q against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_d_matches_live_r(r_dpqr_probe, pi):
    profile = PROFILES[pi]
    got = np.asarray(dDBURR12(jnp.asarray(Y_GRID), profile["mu"],
                              profile["sigma"], profile["nu"]))
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    np.testing.assert_allclose(got, r_d, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_log_density_matches_live_r(r_dpqr_probe, pi):
    """Our log-space form against R's ``log(Sur(y) - Sur(y+1))``.

    R computes the difference first and logs it; we keep the log-mass in log
    space, which is the more accurate of the two wherever the mass is small.
    The comparison is still tight because the grid stays above the underflow
    region -- the divergence R's form *would* show there is documented in
    ``_dburr12_log_pdf``.
    """
    profile = PROFILES[pi]
    got = np.asarray(dDBURR12(jnp.asarray(Y_GRID), profile["mu"],
                              profile["sigma"], profile["nu"], log=True))
    r_logd = np.asarray(r_dpqr_probe[f"{pi}.dlog"], dtype=float)
    np.testing.assert_allclose(got, r_logd, rtol=RTOL, atol=1e-300)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_matches_live_r(r_dpqr_probe, pi):
    profile = PROFILES[pi]
    got = np.asarray(pDBURR12(jnp.asarray(Y_GRID), profile["mu"],
                              profile["sigma"], profile["nu"]))
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    np.testing.assert_allclose(got, r_p, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_q_matches_live_r_bit_for_bit(r_dpqr_probe, pi):
    """``round`` makes q an exact integer map -- no tolerance needed.

    The installed R evaluates ``round(w - 1)``; our evaluation of the same
    expression in double precision must land on the same side of every
    rounding boundary, so the comparison is ``array_equal``.  It is: all
    seven probabilities agree exactly at all four profiles (measured).
    """
    profile = PROFILES[pi]
    got = np.asarray(qDBURR12(jnp.asarray(P_GRID), profile["mu"],
                              profile["sigma"], profile["nu"]))
    r_q = np.asarray(r_dpqr_probe[f"{pi}.q"], dtype=float)
    np.testing.assert_array_equal(got, r_q)


def test_q_reproduces_round_even_though_it_is_not_the_cdf_inverse(r_dpqr_probe):
    """The measured quirk: R's q is ``round(w-1)``, not the CDF inverse.

    At the default profile ``p = 0.5`` gives ``w - 1 = 2.218`` -> ``q = 2``
    while ``F(2) = 0.4593 < 0.5``; ``p = 0.25`` gives ``w - 1 = 0.967`` ->
    ``q = 1`` which *is* the CDF inverse.  Both halves are asserted from the
    probe: our q equals R's, and R's q is not ``min{q : F(q) >= p}`` for
    every p.  Pinning the inconsistency keeps a future R fix (or a
    deliberate CDF-consistent deviation) visible instead of silent.
    """
    mu, sigma, nu = 5.0, 2.0, 2.0
    r_q = np.asarray(r_dpqr_probe["0.q"], dtype=float)
    r_p = np.asarray(r_dpqr_probe["0.p"], dtype=float)
    p_of_y = dict(zip(Y_GRID.tolist(), r_p.tolist()))

    # p = 0.5 -> q = 2, and R's own CDF at 2 is below 0.5.
    i_half = int(np.flatnonzero(P_GRID == 0.5)[0])
    assert r_q[i_half] == 2.0
    assert p_of_y[2.0] < 0.5
    # The count that would be the CDF inverse is 3.
    assert r_p[Y_GRID == 3.0][0] >= 0.5

    # p = 0.25 -> q = 1, which *is* the CDF inverse: the quirk is decided
    # per point by where w - 1 sits relative to the nearest integer.
    i_quarter = int(np.flatnonzero(P_GRID == 0.25)[0])
    assert r_q[i_quarter] == 1.0
    assert r_p[Y_GRID == 1.0][0] >= 0.25 > r_p[Y_GRID == 0.0][0]


# ---------------------------------------------------------------------------
# 2. The boundaries where R and the house convention differ
# ---------------------------------------------------------------------------


def test_non_integer_counts_are_outside_our_support_but_inside_r(r_boundary_probe):
    """R evaluates the continuous difference at 1.5; we return a zero mass."""
    r_half = float(r_boundary_probe["d_half"])
    assert 0.0 < r_half < 1.0  # measured 0.20167999326655994
    assert float(dDBURR12(1.5, 5.0, 2.0, 2.0)) == 0.0
    assert float(dDBURR12(1.5, 5.0, 2.0, 2.0, log=True)) == -np.inf


def test_negative_counts_density_is_zero_on_both_scales_in_r(r_boundary_probe):
    """R clamps ``x < 0`` to 0 *after* logging; we keep the honest ``-inf``."""
    assert float(r_boundary_probe["d_neg"]) == 0.0
    assert float(r_boundary_probe["dlog_neg"]) == 0.0  # R's quirk
    assert float(dDBURR12(-1.0, 5.0, 2.0, 2.0)) == 0.0
    assert float(dDBURR12(-1.0, 5.0, 2.0, 2.0, log=True)) == -np.inf


def test_upper_tail_below_support_pins_r(r_boundary_probe):
    """R's survival at ``q < 0`` is 0; ours is 1 -- both asserted."""
    assert float(r_boundary_probe["p_neg"]) == 0.0
    assert float(r_boundary_probe["p_neg_upper"]) == 0.0  # R's clamp
    assert float(pDBURR12(-1.0, 5.0, 2.0, 2.0, lower_tail=False)) == 1.0
    # ...and R's logged CDF at q < 0 is 0, i.e. a probability of one.
    assert float(r_boundary_probe["p_neg_log"]) == 0.0
    assert float(pDBURR12(-1.0, 5.0, 2.0, 2.0, log_p=True)) == -np.inf


def test_quantile_boundaries_follow_the_house_convention(r_boundary_probe):
    """``p <= 0`` -> 0, ``p >= 1`` -> Inf, outside ``[0, 1]`` -> NaN; R stops."""
    for p in (0.0, 1.0, -0.5, 1.5):
        assert "p must be between 0 and 1" in r_boundary_probe[f"q_stop_{p}"], (
            p, r_boundary_probe[f"q_stop_{p}"])
    assert float(qDBURR12(0.0, 5.0, 2.0, 2.0)) == 0.0
    assert np.isnan(float(qDBURR12(-0.5, 5.0, 2.0, 2.0)))
    assert float(qDBURR12(1.0, 5.0, 2.0, 2.0)) == np.inf
    assert np.isnan(float(qDBURR12(1.5, 5.0, 2.0, 2.0)))
    assert np.isnan(float(qDBURR12(np.nan, 5.0, 2.0, 2.0)))
    # log_p / lower_tail feed the same transform.
    assert float(qDBURR12(-np.inf, 5.0, 2.0, 2.0, log_p=True)) == 0.0
    assert float(qDBURR12(0.0, 5.0, 2.0, 2.0, lower_tail=False)) == np.inf


def test_invalid_nu_is_nan_here_and_a_stop_in_r(r_boundary_probe):
    assert "nu must be greater than 0" in r_boundary_probe["d_nu0"]
    assert np.isnan(float(dDBURR12(2.0, 5.0, 2.0, 0.0)))
    assert np.isnan(float(pDBURR12(2.0, 5.0, 2.0, -1.0)))
    assert np.isnan(float(qDBURR12(0.5, 5.0, 2.0, 0.0)))


# ---------------------------------------------------------------------------
# 3. The six diagonal slots against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_score_slots_match_live_r(r_slot_probe, pi):
    profile = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    args = (y, profile["mu"], profile["sigma"], profile["nu"])
    for slot, param in (("dldm", "mu"), ("dldd", "sigma"), ("dldv", "nu")):
        got = np.asarray(_DBURR12_SCORES[param](*args))
        r_vals = np.asarray(r_slot_probe[f"{pi}.{slot}"], dtype=float)
        np.testing.assert_allclose(got, r_vals, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_hessian_slots_match_live_r(r_slot_probe, pi):
    profile = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    args = (y, profile["mu"], profile["sigma"], profile["nu"])
    for slot, param in (("d2ldm2", "mu"), ("d2ldd2", "sigma"),
                        ("d2ldv2", "nu")):
        got = np.asarray(_DBURR12_HESSIANS[param](*args))
        r_vals = np.asarray(r_slot_probe[f"{pi}.{slot}"], dtype=float)
        np.testing.assert_allclose(got, r_vals, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_hessian_is_minus_score_squared_without_a_floor(pi):
    """DBURR12's ``d2ldX2`` is a plain ``-score^2`` -- no ``-1e-15`` clamp.

    The Sichel cluster clamps at ``-1e-15``; this family does not (R's
    ``-(dldm^2)`` has no ``ifelse``).  The wrapper itself is pinned
    separately in ``test_negative_information_wrapper_does_not_floor``.
    """
    profile = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    args = (y, profile["mu"], profile["sigma"], profile["nu"])
    for param in ("mu", "sigma", "nu"):
        score = np.asarray(_DBURR12_SCORES[param](*args))
        got = np.asarray(_DBURR12_HESSIANS[param](*args))
        np.testing.assert_array_equal(got, -score * score)


def test_negative_information_wrapper_does_not_floor():
    """A tiny score must give ``-1e-18``, not the Sichel ``-1e-15`` floor."""
    from omnilss.distributions_discrete_compat import (
        _dburr12_negative_information,
    )

    def tiny_score(y, mu, sigma, nu):
        return jnp.asarray(1e-9) + 0.0 * jnp.asarray(y)

    hessian = _dburr12_negative_information(tiny_score)
    got = float(hessian(jnp.asarray(0.0), 1.0, 1.0, 1.0))
    assert got == pytest.approx(-1e-18, rel=0, abs=0)
    assert got != -1e-15


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_scores_match_their_own_density(pi):
    """Independent of R: the slots differentiate the density we score with."""
    profile = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    mu, sigma, nu = (float(profile[k]) for k in ("mu", "sigma", "nu"))
    logpdf = jax.vmap(_dburr12_log_pdf, in_axes=(0, None, None, None))
    h = 1e-5
    for param in ("mu", "sigma", "nu"):
        score = np.asarray(_DBURR12_SCORES[param](y, mu, sigma, nu))
        if param == "mu":
            plus = logpdf(y, mu + h, sigma, nu)
            minus = logpdf(y, mu - h, sigma, nu)
        elif param == "sigma":
            plus = logpdf(y, mu, sigma + h, nu)
            minus = logpdf(y, mu, sigma - h, nu)
        else:
            plus = logpdf(y, mu, sigma, nu + h)
            minus = logpdf(y, mu, sigma, nu - h)
        fd = (np.asarray(plus) - np.asarray(minus)) / (2 * h)
        np.testing.assert_allclose(score, fd, rtol=1e-6, atol=1e-9)


def test_density_is_a_normalised_pmf():
    """Independent anchor: the tail closes in closed form.

    A PMF truncated at ``N`` must sum to ``1 - Sur(N+1)`` -- the exact
    expression, not a hand-picked epsilon, so the residual is the arithmetic
    error only (measured <= 1e-15).  This also pins that ``d`` is a proper
    mass function rather than R's continuous difference.
    """
    y = jnp.arange(0, 400, dtype=jnp.float64)
    for profile in PROFILES:
        mu, sigma, nu = (float(profile[k]) for k in ("mu", "sigma", "nu"))
        total = float(jnp.sum(dDBURR12(y, mu, sigma, nu)))
        tail = float((1.0 + ((400.0 / mu) ** sigma)) ** (-nu))
        assert total == pytest.approx(1.0 - tail, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Family metadata and R's starting values
# ---------------------------------------------------------------------------


def test_family_metadata_matches_live_r(r_family_probe):
    assert r_family_probe["parameters"] == "mu TRUE,sigma TRUE,nu TRUE"
    assert r_family_probe["mu_link"] == "log"
    assert r_family_probe["sigma_link"] == "log"
    assert r_family_probe["nu_link"] == "log"
    assert r_family_probe["type"] == "Discrete"
    # R's expression, serialised by jsonlite as one string (auto_unbox).
    assert "mean(y)" in r_family_probe["mu_initial"], r_family_probe["mu_initial"]
    assert "rep(2" in r_family_probe["sigma_initial"]
    assert "rep(2" in r_family_probe["nu_initial"]

    fam = resolve(FAMILY)
    assert fam.parameters == ("mu", "sigma", "nu")
    assert fam.links == {"mu": "log", "sigma": "log", "nu": "log"}
    assert fam.type == "Discrete"


def test_initial_values_are_r_starting_points():
    """sigma and nu start at R's ``rep(2, length(y))``; mu is the roll-out."""
    from omnilss._fitting_init import _initial_parameter_value, _initial_sigma

    fam = resolve(FAMILY)
    y = np.asarray(FIT_Y, dtype=float)
    mu0 = (y + np.mean(y) / 2.0)
    w0 = np.ones(len(y))
    assert float(_initial_sigma(fam, y, mu0, w0)) == 2.0
    assert float(_initial_parameter_value(fam, "nu", y, mu0, w0)) == 2.0


# ---------------------------------------------------------------------------
# 5. End-to-end fit parity against live R
# ---------------------------------------------------------------------------


def test_fit_matches_live_r(r_fit_probe):
    """Intercept-only fit: deviance and link-scale coefficients.

    The family is still listed as a compatibility alias, so the fail-closed
    admission gate needs the explicit opt-in -- this test is exactly the
    evidence that retires that listing (measured: deviance rel 7.9e-7, we are
    *lower*, i.e. slightly further along the same optimum).
    """
    from omnilss.controls import GAMLSSControl
    from omnilss.fitting import gamlss

    ref = r_fit_probe
    assert ref["converged"] is True
    fit = gamlss(
        formula="y ~ 1",
        family=FAMILY,
        data={"y": FIT_Y},
        control=GAMLSSControl(n_cyc=200, c_crit=0.001, trace=False),
        allow_compat_alias=True,
    )
    assert fit.convergence_status() is True, fit.termination_reason
    assert float(fit.deviance) == pytest.approx(float(ref["deviance"]),
                                                rel=1e-5)
    for param in ("mu", "sigma", "nu"):
        ours = float(np.log(np.asarray(fit.fitted_values[param])[0]))
        assert ours == pytest.approx(float(ref[param]), rel=5e-3), (
            param, ours, ref[param])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
