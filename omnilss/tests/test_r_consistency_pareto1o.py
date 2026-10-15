"""PARETO1o (fixed-mu Pareto Type 1) parity against live R at digits=17.

What this replaces
------------------
The previous incarnation of this file queried RTestBus for the family name
``"PARETO1O"`` -- the *registry alias*, with a capital ``O``.  The bus
builds the R call with ``paste0(func_type, family_name)``, which asks R for
``dPARETO1O``; the installed gamlss.dist exports the lowercase-tail name
``dPARETO1o``, so every probe raised ``could not find function`` and the
test never compared a single number.  It "passed" only because RTestBus
turns probe failures into skips.  (The same ``paste0`` defect existed
independently in ``tools/r_parity_batch.R``; both were fixed in D2.42 by
resolving against the namespace exports with a case fallback.)

Four contract details live R 6.1.1 has, all measured
----------------------------------------------------
1. ``pPARETO1o`` ignores ``log.p``: the body has the ``cdf < - log(cdf)``
   typo (the same one ``pPARETO``/``pPARETO1`` have), so ``log.p = TRUE``
   returns the raw CDF.  We honour ``log.p``; the divergence is pinned
   here rather than hidden.
2. ``pPARETO1o`` applies ``ifelse(q <= mu, 0, cdf)`` *after* the
   ``lower.tail`` flip, so R's upper tail is ``0`` at and below the support
   edge where the mathematically correct value is ``1``.  We return ``1``.
3. ``dPARETO1o`` returns ``0`` on the *log* scale outside the support
   (``fy <- ifelse(x <= mu, 0, fy)`` sits after the ``log`` branch), so
   R's ``G.dev.incr = -2 * 0`` there, not ``+inf``.  The family's log-pdf
   reproduces that ``0``; the public ``d`` kernel returns ``-inf`` logged
   (i.e. a zero density).  Both halves are pinned below.
4. Live ``qPARETO1o`` stops for any ``p`` outside the open interval
   ``(0, 1)``; the vendored 6.1-8 source instead assigns ``q[p == 0] <- 0``,
   contradicting its own support (infimum ``mu``).  We keep the house
   convention: the support infimum at ``p <= 0``, ``Inf`` at ``p >= 1``.

The phantom score term (why the fit needs closed-form overrides)
---------------------------------------------------------------
R's ``dldd = 1/sigma + log(mu) - log(y)`` has **no support guard** while
R's density is zero for ``y <= mu``.  With ``mu`` fixed at ``min(y)`` the
minimum observation always sits exactly on the edge, so its phantom
``1/sigma`` term is inside every IRLS normal equation even though the
observation contributes nothing to the deviance.  The fixed point is
therefore ``sigma = n / sum(log(y/mu))`` (all ``n`` observations, including
the boundary one) -- measured to 10 digits against R's landing point, with
``c.crit = 1e-10`` not moving it (a fixed point, not an early stop).
An AD score that correctly vanishes at the boundary converges to
``n' / sum(...)`` with ``n' = n - 1`` instead, which is a different
distribution's MLE.  ``distributions_b16`` therefore installs R's closed
forms as ``score_overrides``/``hessian_overrides``; section 5 below pins
both the slots and the fixed point.

Methodology
-----------
d/p/q are compared element-wise against **live R** through ``digits=17``
subprocess probes (the ST1/ST2 house style).  Nothing is hard-coded: the
reference values come from the running R, and a mistyped constant cannot
make these tests pass.
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
from omnilss.distributions_b16 import (  # noqa: E402
    _PARETO1O_HESSIANS,
    _PARETO1O_SCORES,
    _pareto1o_log_pdf,
)

FAMILY = "PARETO1O"  # the registry alias RTestBus resolves (capital O)
RSCRIPT = shutil.which("Rscript")

# (mu, sigma) profiles.  mu is the fixed support infimum, sigma the only
# estimated parameter; the grid spans a tight and a heavy tail.
PROFILES = [
    dict(mu=0.5, sigma=0.5),
    dict(mu=1.0, sigma=1.0),
    dict(mu=2.0, sigma=2.0),
    dict(mu=5.0, sigma=0.3),
    dict(mu=0.5, sigma=2.0),
]

#: Relative offsets from mu -- the support is (mu, inf), so everything the
#: parity grid samples must be strictly above the profile's own mu.  1e-7
#: sits just inside the edge, where the CDF is O(1e-7) and only an
#: ``-expm1`` evaluation keeps full relative precision.
_REL_X = (1.0000001, 1.05, 1.3, 2.0, 5.0, 30.0, 200.0)

#: The out-of-support point every profile shares: half of its own mu.
_OUTSIDE_REL = 0.5

P_GRID = np.array(
    [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
)

# Measured agreement floors of the digits=17 probes, not round numbers.
RTOL = 1e-12
ATOL = 1e-300

#: The support-edge point (``mu * (1 + 1e-7)``) is where R's literal
#: ``1 - (mu/q)^sigma`` cancels away ~9 digits -- measured rel 5.4e-10 /
#: 8.0e-11 / 8.0e-11 on the p probes and 7.4e-10 on the log-density
#: probe, across the profiles.  That cancellation is exactly why our
#: kernel evaluates ``-expm1(sigma * log(mu/q))`` instead; the edge point
#: is therefore compared at the measured floor, and the rest of the grid
#: at RTOL.
EDGE_RTOL = 1e-8

# ---------------------------------------------------------------------------
# The two fixed fit samples.  Provenance (live R 4.6.0, gamlss.dist 6.1.1):
# generated with rPARETO1o / rNO and dumped with digits=17.  Embedded
# rather than regenerated so the Python and R fits provably see the same
# numbers -- a shared rng stream would be order-sensitive.
# ---------------------------------------------------------------------------

# Sample 1: 30 draws from the fitted model itself, one extreme (386.66).
FIT1_Y = np.array([
    1.1716792875855666, 1.0077277905971955, 1.7691668294922271,
    1.4005415464327804, 1.2815864958058178, 1.191282955156114,
    21.321304315553256, 1.7000369094633196, 1.5003309941744007,
    4.788287009822475, 386.662489836395, 1.1476481948024015,
    1.8690862028341337, 47.88858697664875, 11.837777842568803,
    17.94292545798145, 1.3658494572744542, 3.5791840922227864,
    1.2864861183532676, 1.400761274092471, 1.0100543772597037,
    4.9140870720669625, 3.3512345863696607, 2.5717731165035436,
    1.4148970172063513, 2.340584042499559, 2.5727635561934057,
    6.21690408167489, 6.880731501763041, 4.219432635631009,
])

# Sample 2: 40 draws plus a covariate (mu is fixed, so the covariate is
# inert -- but it must be inert *identically* on both sides).
FIT2_Y = np.array([
    3.282443055330715, 5.789844008035961, 2.0199301221223886,
    3.6967025886783054, 2.996975223220943, 2.4868965151910807,
    2.027272865081895, 2.4663148581122005, 2.8182845483180814,
    2.345395694016146, 3.147573758471636, 7.591114763570765,
    2.320492934820383, 6.029489935017784, 2.3139003262559332,
    4.497257127046155, 2.972200156104802, 2.0903558055044646,
    2.08084260713341, 2.974652078527572, 2.347700245051473,
    2.214289656224615, 5.832918560793476, 5.216322658911777,
    2.314878016368261, 2.148050732046946, 4.855921039666529,
    4.5298994748616135, 2.1562615812833457, 2.1318761411827567,
    2.1786949458714093, 2.1536926773985354, 13.03288627934154,
    4.41578803127169, 2.9844607357513153, 2.686542291784474,
    2.313694053115421, 30.940418058089392, 2.1395241249214467,
    2.2386532644337165,
])
FIT2_X = np.array([
    1.215408162057621, -1.2955735712157117, 0.8203896032126184,
    0.8674399791734169, 0.45094682107320044, 0.6885456552696072,
    -0.2186590323746389, 0.7680179875306749, -0.03197944031292938,
    -1.189500661764833, 0.4548605276626972, 0.5843511655939126,
    0.7726779404226064, -0.6405889536481759, 0.7584684910855557,
    0.4779235743831309, -0.24374951932134234, 1.0404954078766635,
    1.1920716759875507, 0.5698348147186554, 0.8350889473912342,
    -0.11873434383811736, -0.8947930344732505, -0.6795571772837643,
    0.015255282731024108, -0.05690326286349004, -0.8747090694728092,
    -2.7168289913317776, -0.5897164225780456, 0.7385500543861896,
    0.6142932075583741, -1.126683439436548, 0.926972690285982,
    1.7169665299648238, -1.3058335222461686, 0.11127581921667648,
    -0.28531257790983333, -0.5615361067540255, 0.3976100672507895,
    -0.27726066071513555,
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


def _y_grid(profile) -> np.ndarray:
    mu = float(profile["mu"])
    xs = [mu * rel for rel in _REL_X]
    xs.append(mu * _OUTSIDE_REL)
    return np.array(xs)


@pytest.fixture(scope="module")
def fam():
    return resolve(FAMILY)


@pytest.fixture(scope="module")
def r_dpqr_probe():
    """Element-wise live-R d/p/q at digits=17 (house style).

    ``d`` also samples the out-of-support point (half of mu): R's
    ``ifelse`` returns 0 there and so does our kernel, so the boundary is
    part of the comparison, not an exclusion.
    """
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']!r}, sigma={p['sigma']!r}"
        xs = _y_grid(p)
        lines.append(
            f"out[['{pi}.d']] <- vapply(c({_rvec(xs)}), function(xi) "
            f"suppressWarnings(as.numeric(gamlss.dist::dPARETO1o(xi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.p']] <- vapply(c({_rvec(xs)}), function(qi) "
            f"suppressWarnings(as.numeric(gamlss.dist::pPARETO1o(qi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.q']] <- vapply(c({_rvec(P_GRID)}), function(pp) "
            f"suppressWarnings(as.numeric(gamlss.dist::qPARETO1o(pp, {args}))), "
            f"numeric(1))"
        )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_family_probe():
    """The live family object: fixed-parameter flags, links, initial, slots."""
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "o <- get('PARETO1o', envir=asNamespace('gamlss.dist'))()",
        "out <- list(",
        "  parameters = names(o$parameters),",
        "  mu_estimable = as.numeric(o$parameters[['mu']]),",
        "  sigma_estimable = as.numeric(o$parameters[['sigma']]),",
        "  mu_link = o$mu.link, sigma_link = o$sigma.link, type = o$type,",
        "  mu_initial = deparse(o$mu.initial),",
        "  sigma_initial = deparse(o$sigma.initial)",
        ")",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_deriv_probe():
    """R's own dldd/d2ldd2 slots (what the overrides replace), digits=17."""
    lines = ["suppressMessages(library(gamlss.dist))",
             "o <- get('PARETO1o', envir=asNamespace('gamlss.dist'))()",
             "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']!r}, sigma={p['sigma']!r}"
        xs = _y_grid(p)
        for slot in ("dldd", "d2ldd2"):
            lines.append(
                f"out[['{pi}.{slot}']] <- vapply(c({_rvec(xs)}), function(yi) "
                f"suppressWarnings(as.numeric(o[['{slot}']](y=yi, {args}))), "
                f"numeric(1))"
            )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_fit_probe():
    """Live-R ``gamlss`` fits on both embedded samples, same RS control."""
    y1 = _rvec(FIT1_Y)
    y2 = _rvec(FIT2_Y)
    x2 = _rvec(FIT2_X)
    lines = [
        "suppressMessages({library(gamlss); library(gamlss.dist)})",
        "ctl <- " + R_FIT_CONTROL,
        f"y1 <- c({y1})",
        f"y2 <- c({y2})",
        f"xx <- c({x2})",
        "m1 <- gamlss(y1 ~ 1, family = PARETO1o, control = ctl)",
        "m2 <- gamlss(y2 ~ xx, family = PARETO1o, control = ctl)",
        # The closed form R's own IRLS fixed point must land on, computed
        # independently of either fit: sigma = n / sum(log(y/mu)).
        "closed1 <- length(y1) / sum(log(y1 / min(y1)))",
        "closed2 <- length(y2) / sum(log(y2 / min(y2)))",
        "out <- list(",
        "  s1 = list(deviance = m1$G.deviance, converged = isTRUE(m1$converged),",
        "            sigma_coef = as.numeric(coef(m1, what='sigma')),",
        "            mu_fitted = as.numeric(fitted(m1, what='mu')[1]),",
        "            mu_all_min = all(abs(fitted(m1, what='mu') - min(y1)) < 1e-12),",
        "            closed = closed1),",
        "  s2 = list(deviance = m2$G.deviance, converged = isTRUE(m2$converged),",
        "            sigma_coef = as.numeric(coef(m2, what='sigma')),",
        "            mu_fitted = as.numeric(fitted(m2, what='mu')[1]),",
        "            mu_all_min = all(abs(fitted(m2, what='mu') - min(y2)) < 1e-12),",
        "            closed = closed2)",
        ")",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return json.loads(_run_r(lines).strip())


def _ours_d(fam, xs, profile):
    return np.asarray(
        fam.d(jnp.asarray(xs, dtype=jnp.float64),
              float(profile["mu"]), float(profile["sigma"]))
    )


def _ours_p(fam, xs, profile, **kw):
    return np.asarray(
        fam.p(jnp.asarray(xs, dtype=jnp.float64),
              float(profile["mu"]), float(profile["sigma"]), **kw)
    )


def _ours_q(fam, ps, profile, **kw):
    return np.asarray(
        fam.q(jnp.asarray(ps, dtype=jnp.float64),
              float(profile["mu"]), float(profile["sigma"]), **kw)
    )


# ---------------------------------------------------------------------------
# 1. Density against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_d_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    xs = _y_grid(profile)
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    got = _ours_d(fam, xs, profile)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, r_d, rtol=RTOL, atol=ATOL)


def test_d_outside_the_support_is_zero_on_both_sides(r_dpqr_probe, fam):
    """``x <= mu`` gives 0 on both sides -- the one boundary R does not fumble.

    R's ``dPARETO1o`` applies ``fy <- ifelse(x <= mu, 0, fy)`` after the
    log branch; on the *density* scale that is the mathematically correct
    value, and our kernel agrees (docstring detail 3 is about the log
    scale, pinned below).  Asserted for every profile, not just one.
    """
    for pi, profile in enumerate(PROFILES):
        mu = float(profile["mu"])
        xs = _y_grid(profile)
        r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
        got = _ours_d(fam, xs, profile)
        i_out = len(xs) - 1  # the appended 0.5 * mu point
        assert xs[i_out] == pytest.approx(0.5 * mu, rel=0, abs=0)
        assert got[i_out] == 0.0, f"profile {pi}: our d(0.5mu) = {got[i_out]}"
        assert r_d[i_out] == 0.0, f"profile {pi}: R d(0.5mu) = {r_d[i_out]}"
        # ...and the edge itself (mu) is also outside: support is y > mu.
        at_mu = float(fam.d(jnp.asarray(mu), mu, float(profile["sigma"])))
        assert at_mu == 0.0, f"profile {pi}: our d(mu) = {at_mu}"


def test_log_density_outside_support_is_zero_in_r_and_in_the_logpdf(fam):
    """Docstring detail 3: R returns ``0`` (not ``-Inf``) on the log scale.

    Three values, three contracts, all pinned:
    * R's ``dPARETO1o(..., log = TRUE)`` at ``y <= mu`` is ``0``;
    * our ``_pareto1o_log_pdf`` (the scoring path) is ``0`` there too, so
      ``g_dev_inc = -2 * 0`` matches R's deviance contribution;
    * our public ``d(..., log = True)`` is ``-inf`` -- the mathematically
      honest log of a zero density.  The split is deliberate: the fitter
      must follow R, the public kernel must not lie.
    """
    mu, sigma = 0.5, 0.5
    y = np.array([0.25, 0.5])  # at and below the support edge
    lines = [
        "suppressMessages(library(gamlss.dist))",
        f"out <- vapply(c({_rvec(y)}), function(yi) "
        f"as.numeric(gamlss.dist::dPARETO1o(yi, mu={mu!r}, sigma={sigma!r}, "
        "log = TRUE)), numeric(1))",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    r_logd = np.asarray(json.loads(_run_r(lines).strip()), dtype=float)
    assert np.all(r_logd == 0.0), f"live R log-density outside support: {r_logd}"

    ours_logpdf = np.asarray(_pareto1o_log_pdf(jnp.asarray(y), mu, sigma))
    assert np.all(ours_logpdf == 0.0), ours_logpdf

    ours_logd = np.asarray(fam.d(jnp.asarray(y), mu, sigma, log=True))
    assert np.all(ours_logd == -np.inf), ours_logd
    # ...and the density scale is a plain zero, not exp(0) = 1.
    assert np.all(_ours_d(fam, y, dict(mu=mu, sigma=sigma)) == 0.0)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_g_dev_inc_matches_r_log_density_contract(r_dpqr_probe, fam, pi):
    """``g_dev_inc`` must be ``-2 *`` the *logpdf*, on both sides of the edge.

    Inside the support this is the ordinary identity; at the boundary it is
    the D2.42 point: R's deviance contribution is ``0`` (not ``+inf``), and
    with ``mu`` fixed at ``min(y)`` every fit evaluates that point.  The R
    reference is reconstructed from the same probe as the density:
    ``-2 * dPARETO1o(y, ..., log = TRUE)`` -- which for the out-of-support
    point is ``-2 * 0 = 0``, exactly what live R's ``G.dev.incr`` returns.
    """
    profile = PROFILES[pi]
    xs = _y_grid(profile)
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    mu, sigma = float(profile["mu"]), float(profile["sigma"])

    # R's log-scale density at the same points (the ifelse-after-log quirk
    # is what makes the boundary contribution 0).
    lines = [
        "suppressMessages(library(gamlss.dist))",
        f"out <- vapply(c({_rvec(xs)}), function(yi) "
        f"as.numeric(gamlss.dist::dPARETO1o(yi, mu={mu!r}, sigma={sigma!r}, "
        "log = TRUE)), numeric(1))",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    r_logd = np.asarray(json.loads(_run_r(lines).strip()), dtype=float)

    dev = np.asarray(fam.g_dev_inc(jnp.asarray(xs), mu, sigma))
    # The edge point (index 0) is R's cancellation zone on the log-density
    # formula too (sigma*log(mu) vs (sigma+1)*log(x) both ~0.69 for a
    # result ~1.5e-7): measured rel 7.4e-10, hence EDGE_RTOL there.
    np.testing.assert_allclose(dev[1:], -2.0 * r_logd[1:], rtol=RTOL,
                               atol=ATOL)
    np.testing.assert_allclose(dev[0], -2.0 * r_logd[0], rtol=EDGE_RTOL,
                               atol=ATOL)
    # The boundary observation is the whole point: zero contribution.
    assert dev[-1] == 0.0, dev[-1]
    # ...and the density-scale probe agrees with the deviance path inside.
    inside = slice(0, len(xs) - 1)
    np.testing.assert_allclose(np.exp(-0.5 * dev[inside]), r_d[inside],
                               rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. CDF against live R -- and the two places R fumbles the tails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_matches_live_r(r_dpqr_probe, fam, pi):
    """Lower tail at every grid point, including the support edge.

    R's ``ifelse(q <= mu, 0, cdf)`` and our ``cdf = 0`` outside agree on
    the *lower* tail, so the boundary is part of the comparison here.
    The edge point (index 0, ``mu * (1 + 1e-7)``) is compared at
    ``EDGE_RTOL``: R's literal ``1 - (mu/q)^sigma`` cancels there while
    our ``-expm1`` form does not, so the residual gap is R's measured
    loss, not ours.
    """
    profile = PROFILES[pi]
    xs = _y_grid(profile)
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    got = _ours_p(fam, xs, profile)
    np.testing.assert_allclose(got[1:], r_p[1:], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got[0], r_p[0], rtol=EDGE_RTOL, atol=ATOL)


def test_cdf_at_the_support_edge_keeps_full_precision(fam):
    """Independent of R: the edge value is the cancellation-free form.

    At ``q = mu * (1 + 1e-7)`` the CDF is O(1e-7); the naive
    ``1 - (mu/q)^sigma`` computes it as a difference of two numbers that
    are both 1 to machine precision (R's ~5e-10 relative loss, pinned in
    ``test_p_matches_live_r``).  Our value must be the ``-expm1``
    evaluation to full double precision -- this test does not need R.
    """
    mu, sigma = 0.5, 0.5
    q = mu * (1.0 + 1e-7)
    exact = -np.expm1(sigma * (np.log(mu) - np.log(q)))
    ours = float(fam.p(jnp.asarray(q), mu, sigma))
    assert ours == pytest.approx(exact, rel=1e-15, abs=0.0)
    # ...and it is genuinely tiny, i.e. the point really is at the edge.
    assert 1e-9 < ours < 1e-6, ours


def test_p_ignores_log_p_in_live_r(fam):
    """Docstring detail 1: R's ``cdf < - log(cdf)`` typo, pinned both sides.

    Live R returns the raw CDF when ``log.p = TRUE`` (measured here, not
    assumed); we honour the flag, so the two paths differ by ``log(cdf)``.
    A future gamlss.dist that fixes the typo would flip the first half of
    this test -- that is the point of pinning it.
    """
    mu, sigma = 0.5, 0.5
    q = np.array([0.6, 1.0, 5.0])
    lines = [
        "suppressMessages(library(gamlss.dist))",
        f"out <- vapply(c({_rvec(q)}), function(qi) "
        f"as.numeric(gamlss.dist::pPARETO1o(qi, mu={mu!r}, sigma={sigma!r}, "
        "log.p = TRUE)), numeric(1))",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    r_logp = np.asarray(json.loads(_run_r(lines).strip()), dtype=float)
    r_plain = np.asarray(json.loads(_run_r([
        "suppressMessages(library(gamlss.dist))",
        f"out <- vapply(c({_rvec(q)}), function(qi) "
        f"as.numeric(gamlss.dist::pPARETO1o(qi, mu={mu!r}, sigma={sigma!r})), "
        "numeric(1))",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]).strip()), dtype=float)
    # R ignores log.p: identical values, and they are probabilities, not logs.
    np.testing.assert_array_equal(r_logp, r_plain)
    assert np.all(r_logp > 0.0) and np.all(r_logp < 1.0)
    # We honour it.
    ours = _ours_p(fam, q, dict(mu=mu, sigma=sigma), log_p=True)
    np.testing.assert_allclose(ours, np.log(r_plain), rtol=RTOL, atol=ATOL)


def test_upper_tail_clamp_below_support_pins_r(fam):
    """Docstring detail 2: R's upper tail is 0 at the edge; ours is 1.

    ``pPARETO1o`` flips ``lower.tail`` *first* and only then applies
    ``ifelse(q <= mu, 0, cdf)`` -- so for ``q <= mu`` the upper tail is
    clamped to ``0`` where the correct survival function is ``1``.  Both
    halves asserted: our value is the mathematically right one, R's is
    measured here so the divergence cannot silently disappear (or spread).
    """
    mu, sigma = 0.5, 0.5
    for q in (0.25, 0.5):  # below, and exactly at, the support edge
        ours = float(fam.p(jnp.asarray(q), mu, sigma, lower_tail=False))
        assert ours == 1.0, f"our upper tail at q={q} is {ours}, want 1"
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "for (q in c(0.25, 0.5)) {",
        f"  r <- gamlss.dist::pPARETO1o(q, mu={mu!r}, sigma={sigma!r}, "
        "lower.tail = FALSE)",
        '  cat(sprintf("%.17g\\n", r))',
        "}",
    ]
    out = _run_r(lines).strip().splitlines()
    assert out == ["0", "0"], out
    # Inside the support the two tails are exact complements (no clamp).
    q = 5.0
    lower = float(fam.p(jnp.asarray(q), mu, sigma))
    upper = float(fam.p(jnp.asarray(q), mu, sigma, lower_tail=False))
    assert lower + upper == pytest.approx(1.0, rel=0, abs=1e-15)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_upper_tail_inside_support_matches_live_r(r_dpqr_probe, fam, pi):
    """Away from the edge the upper tail is R's own ``1 - cdf``.

    The last point (``0.5 * mu``) is the clamp zone -- excluded from the
    R comparison because R's 0 there is the bug pinned above -- and the
    first (the edge) is compared at ``EDGE_RTOL`` for the same
    cancellation reason as ``test_p_matches_live_r``.
    """
    profile = PROFILES[pi]
    xs = _y_grid(profile)
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    got = _ours_p(fam, xs[:-1], profile, lower_tail=False)
    np.testing.assert_allclose(got[1:], 1.0 - r_p[1:-1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got[0], 1.0 - r_p[0], rtol=EDGE_RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. Quantile against live R -- and the boundary stop we do not copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_q_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_q = np.asarray(r_dpqr_probe[f"{pi}.q"], dtype=float)
    got = _ours_q(fam, P_GRID, profile)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, r_q, rtol=RTOL, atol=ATOL)


def test_live_r_stops_for_p_outside_unit_interval():
    """Docstring detail 4: live qPARETO1o stops; the vendored source does not.

    The vendored 6.1-8 assigns ``q[p == 0] <- 0`` (below the support
    infimum ``mu``), ``q[p == 1] <- Inf`` and NaN outside ``[0, 1]``; the
    installed 6.1.1 rejects all four with ``p must be between 0 and 1``.
    Pinned so an upgrade that adopts the vendored boundary is visible.
    """
    lines = ["suppressMessages(library(gamlss.dist))"]
    for p in (0.0, 1.0, -0.5, 1.5):
        lines.append(
            f"r <- tryCatch(gamlss.dist::qPARETO1o({p!r}, mu=0.5, sigma=0.5), "
            "error = function(e) conditionMessage(e))"
        )
        lines.append(f'cat("[{p}]", r, "\\n")')
    out = _run_r(lines)
    assert out.count("p must be between 0 and 1") == 4, out


def test_q_boundaries_are_mu_and_inf(fam):
    """Our convention: support infimum at ``p <= 0``, ``Inf`` at ``p >= 1``.

    Consistent with ``qPARETO``/``qPARETO1`` (house convention) and with
    the support -- the vendored ``q[p == 0] <- 0`` would sit *below* the
    infimum and is the contradiction pinned above.
    """
    mu, sigma = 0.5, 0.5
    assert float(fam.q(jnp.asarray(0.0), mu, sigma)) == mu
    assert float(fam.q(jnp.asarray(-0.5), mu, sigma)) == mu
    assert float(fam.q(jnp.asarray(1.0), mu, sigma)) == np.inf
    assert float(fam.q(jnp.asarray(1.5), mu, sigma)) == np.inf
    assert np.isnan(float(fam.q(jnp.asarray(np.nan), mu, sigma)))
    # log_p / lower_tail feed the same transform.
    assert float(fam.q(jnp.asarray(-np.inf), mu, sigma, log_p=True)) == mu
    assert float(fam.q(jnp.asarray(0.0), mu, sigma, lower_tail=False)) == np.inf


def test_q_inverts_p(fam):
    """Independent of R: ``p(q(p)) == p`` and ``q(p(q)) == q``."""
    for profile in (PROFILES[0], PROFILES[2]):
        mu, sigma = float(profile["mu"]), float(profile["sigma"])
        p_of_y = _ours_p(fam, _y_grid(profile)[:-1], profile)
        y_back = _ours_q(fam, p_of_y, profile)
        np.testing.assert_allclose(y_back, _y_grid(profile)[:-1],
                                   rtol=1e-10, atol=1e-12)
        q_of_p = _ours_q(fam, P_GRID, profile)
        p_back = _ours_p(fam, q_of_p, profile)
        np.testing.assert_allclose(p_back, P_GRID, rtol=1e-10, atol=1e-12)


def test_r_is_quantile_of_uniforms(fam):
    """``rPARETO1o`` is ``qPARETO1o(runif(n))`` -- R's own construction."""
    key = jax.random.PRNGKey(20261022)
    n = 64
    u = jax.random.uniform(key, shape=(n,), dtype=jnp.float64)
    expected = np.asarray(fam.q(u, 0.5, 0.5))
    got = np.asarray(fam.r(key, n, 0.5, 0.5))
    np.testing.assert_array_equal(got, expected)
    assert np.all(got > 0.5)


# ---------------------------------------------------------------------------
# 4. The export-level gate fix: the capital-O alias
# ---------------------------------------------------------------------------


def test_rtestbus_resolves_the_capital_o_alias():
    """RTestBus must resolve ``"PARETO1O"`` to R's ``dPARETO1o``.

    The bus (and ``tools/r_parity_batch.R``) built the call with
    ``paste0(func_type, family_name)``, which asks R for ``dPARETO1O`` --
    a name the installed gamlss.dist does not export (the tail ``o`` is
    lowercase).  Eleven mixed-case families were silently unprobed; the
    D2.42 resolver (exact export match first, case-insensitive fallback
    second) is what this pins.  The comparison itself is at the bus's own
    ``digits = 10`` precision, so 1e-8 is the honest tolerance here.
    """
    from tests.rbus.core import RTestBus

    bus = RTestBus()
    x = np.array([2.1, 3.0, 10.0, 50.0])
    mu = np.full(4, 0.5)
    sigma = np.full(4, 0.5)
    r_d = np.asarray(
        bus.eval_family(FAMILY, "d", {
            "x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(),
        }),
        dtype=float,
    )
    assert np.all(np.isfinite(r_d)), r_d
    fam = resolve(FAMILY)
    ours = _ours_d(fam, x, dict(mu=0.5, sigma=0.5))
    np.testing.assert_allclose(ours, r_d, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. The IRLS inputs: R's closed forms and the fixed point they imply
# ---------------------------------------------------------------------------


def test_score_overrides_match_r_family_slots(r_deriv_probe, fam):
    """``dldd``/``d2ldd2`` must be R's own formulas -- no support guard.

    These are the slots ``score_overrides``/``hessian_overrides`` replace,
    so this is the direct evidence the IRLS inputs are R's.  The grid
    includes the out-of-support point on purpose: R's formulas evaluate
    the interior expression there (the phantom term), and so must ours.
    """
    for pi, profile in enumerate(PROFILES):
        xs = _y_grid(profile)
        mu = np.full(len(xs), float(profile["mu"]))
        sigma = np.full(len(xs), float(profile["sigma"]))
        for slot, table in (("dldd", _PARETO1O_SCORES),
                            ("d2ldd2", _PARETO1O_HESSIANS)):
            r_vals = np.asarray(r_deriv_probe[f"{pi}.{slot}"], dtype=float)
            ours = np.asarray(
                table["sigma"](jnp.asarray(xs), jnp.asarray(mu),
                               jnp.asarray(sigma))
            )
            np.testing.assert_allclose(ours, r_vals, rtol=RTOL, atol=ATOL)


def test_overrides_are_installed_on_the_family(fam):
    """The family must actually use the R closed forms, not just carry them."""
    assert fam.score_functions["sigma"] is _PARETO1O_SCORES["sigma"]
    assert fam.hessian_functions["sigma"] is _PARETO1O_HESSIANS["sigma"]


def test_phantom_score_term_is_visible_at_the_boundary(fam):
    """The D2.42 root cause, asserted directly.

    At ``y = mu`` the density is zero but R's ``dldd`` still returns
    ``1/sigma + log(mu) - log(mu) = 1/sigma`` -- a ghost observation
    inside the normal equation.  An AD score (or a guarded transcription)
    returns 0 there and converges to ``(n-1)/sum(log(y/mu))`` instead,
    a different MLE.  This is why the closed forms are overrides.
    """
    mu, sigma = 0.5, 0.5
    ghost = float(_PARETO1O_SCORES["sigma"](
        jnp.asarray(mu), jnp.asarray(mu), jnp.asarray(sigma)))
    assert ghost == pytest.approx(1.0 / sigma, rel=RTOL)
    # The same point through the deviance path contributes nothing.
    assert float(fam.g_dev_inc(y=mu, mu=mu, sigma=sigma)) == 0.0


def test_irls_fixed_point_is_reproduced(r_fit_probe):
    """Both fits must land on R's closed-form fixed point.

    R's IRLS with the unguarded ``dldd`` has the closed-form fixed point
    ``sigma = n / sum(log(y/mu))`` (all n observations -- the boundary
    one contributes its phantom ``1/sigma`` to the normal equation but
    nothing to the deviance).  Evidence for this being R's fixed point
    rather than an early stop: ``c.crit = 1e-10`` does not move R's
    landing point (measured offline, rel 6.6e-10 / 7e-9 from the closed
    form on the two samples).  We must land on the same closed form,
    deviance agreement being the operational statement.
    """
    for key, sample_y in (("s1", FIT1_Y), ("s2", FIT2_Y)):
        ref = r_fit_probe[key]
        closed = float(ref["closed"])
        eta_closed = np.log(closed)
        # R's own landing point is the closed form to its stopping slack.
        assert abs(float(ref["sigma_coef"]) - eta_closed) < 1e-8, (
            f"{key}: R eta {ref['sigma_coef']} vs closed {eta_closed}"
        )

        from omnilss.controls import GAMLSSControl
        from omnilss.fitting import gamlss

        fit = gamlss(
            formula="y ~ 1",
            family=FAMILY,
            data={"y": sample_y},
            control=GAMLSSControl(n_cyc=200, c_crit=0.001, trace=False),
        )
        assert fit.convergence_status() is True, fit.termination_reason
        sigma_fitted = float(np.asarray(fit.fitted_values["sigma"])[0])
        # ...and ours is the closed form itself, to full convergence.
        assert abs(np.log(sigma_fitted) - eta_closed) < 1e-8, (
            f"{key}: our eta {np.log(sigma_fitted)} vs closed {eta_closed}"
        )
        assert sigma_fitted == pytest.approx(closed, rel=1e-8)


# ---------------------------------------------------------------------------
# 6. End-to-end fit parity against live R
# ---------------------------------------------------------------------------


def test_fit_matches_live_r_sample1(r_fit_probe):
    """Intercept-only fit on sample 1 (30 obs, one extreme)."""
    from omnilss.controls import GAMLSSControl
    from omnilss.fitting import gamlss

    fit = gamlss(
        formula="y ~ 1",
        family=FAMILY,
        data={"y": FIT1_Y},
        control=GAMLSSControl(n_cyc=200, c_crit=0.001, trace=False),
    )
    ref = r_fit_probe["s1"]
    assert ref["converged"] is True
    assert fit.convergence_status() is True, fit.termination_reason
    np.testing.assert_allclose(fit.deviance, ref["deviance"],
                               rtol=1e-9, atol=0.0)
    # mu is pinned at min(y) on both sides -- the family-level fixed
    # parameter, not a fitted intercept.
    mu_fitted = np.asarray(fit.fitted_values["mu"], dtype=float)
    assert np.all(mu_fitted == pytest.approx(float(np.min(FIT1_Y)), rel=0,
                                             abs=0))
    assert ref["mu_all_min"] is True
    assert float(ref["mu_fitted"]) == pytest.approx(float(np.min(FIT1_Y)),
                                                    rel=0, abs=0)


def test_fit_matches_live_r_sample2_with_covariate(r_fit_probe):
    """``y ~ x`` fit on sample 2: the covariate must be inert, identically.

    ``mu`` is fixed, so R ignores the ``mu`` formula entirely (its
    ``fitted(mu)`` is the constant ``min(y)`` -- measured, see
    ``mu_all_min``); ``sigma`` stays intercept-only.  Our fitter must do
    the same, and land on the same deviance.
    """
    from omnilss.controls import GAMLSSControl
    from omnilss.fitting import gamlss

    fit = gamlss(
        formula="y ~ x",
        family=FAMILY,
        data={"y": FIT2_Y, "x": FIT2_X},
        control=GAMLSSControl(n_cyc=200, c_crit=0.001, trace=False),
    )
    ref = r_fit_probe["s2"]
    assert ref["converged"] is True
    assert fit.convergence_status() is True, fit.termination_reason
    np.testing.assert_allclose(fit.deviance, ref["deviance"],
                               rtol=1e-9, atol=0.0)
    mu_fitted = np.asarray(fit.fitted_values["mu"], dtype=float)
    assert np.all(mu_fitted == pytest.approx(float(np.min(FIT2_Y)), rel=0,
                                             abs=0))
    assert ref["mu_all_min"] is True


# ---------------------------------------------------------------------------
# 7. The fixed-mu mechanism itself
# ---------------------------------------------------------------------------


def test_family_marks_mu_non_estimable(r_family_probe, fam):
    """The family object's own flags, read from live R, not restated."""
    assert r_family_probe["mu_estimable"] == 0.0
    assert r_family_probe["sigma_estimable"] == 1.0
    assert fam.fixed_parameters == ("mu",)
    assert fam.links == {"mu": "log", "sigma": "log"}
    assert r_family_probe["mu_link"] == "log"
    assert r_family_probe["sigma_link"] == "log"
    assert r_family_probe["type"] == "Continuous"
    # R's mu.initial is the data-dependent expression we resolve.  jsonlite
    # serialises deparse()'s character *vector*, hence the line search.
    assert any("min(y)" in line for line in r_family_probe["mu_initial"]), (
        r_family_probe["mu_initial"]
    )


def test_resolve_fixed_parameter_values_pins_min_y():
    """``fitting._resolve_fixed_parameter_values`` must give R's ``min(y)``.

    The mechanism (D2.39) only fires when the family declares
    ``fixed_parameters``; PARETO1o was the first family to need a
    *data-dependent* default, which is why the resolver takes the
    response vector rather than a constant table.
    """
    from omnilss.fitting import _resolve_fixed_parameter_values

    fam = resolve(FAMILY)
    y = np.asarray(FIT1_Y, dtype=float)
    values = _resolve_fixed_parameter_values(fam, data={"y": y}, n_obs=len(y),
                                             y=y)
    mu = np.asarray(values["mu"], dtype=float)
    assert mu.shape == y.shape
    assert np.all(mu == pytest.approx(float(np.min(y)), rel=0, abs=0))


def test_initial_sigma_is_r_starting_value():
    """``sigma.initial = rep(0.5)`` (R's own), not the roll-out default."""
    from omnilss._fitting_init import _initial_sigma

    fam = resolve(FAMILY)
    y = np.asarray(FIT1_Y, dtype=float)
    mu0 = np.full(len(y), float(np.min(y)))
    w0 = np.ones(len(y))
    sigma0 = float(_initial_sigma(fam, y, mu0, w0))
    assert sigma0 == 0.5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
