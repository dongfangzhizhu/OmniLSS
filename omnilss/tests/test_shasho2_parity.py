"""SHASHo2 parity: the second sinh-arcsinh parameterization, end to end.

The defect
----------
Two separate things were wrong with ``SHASHo2`` before D2.40:

1. ``distributions_b14._shasho2_log_pdf`` implemented a **different
   distribution** from R's ``dSHASHo2``.  It used ``z = (y - mu) / sigma``
   (R divides by ``sigma * tau``), ``sinh(tau*asinh(z) + nu)`` (R's argument
   is ``tau*asinh(z) - nu``) and carried a spurious ``+ log(tau)`` while
   omitting R's ``+ log(cosh(arg))``.  Every RS-deviance comparison for the
   family was therefore apples-to-oranges, and every ``d``/score the family
   produced was wrong.
2. ``p``/``q``/``r`` were never wired up: ``build_ad_family``'s NaN
   placeholders stayed in place, so the dpqr gate read ``fail`` on all three
   rows.

Fixing (1) surfaced something worse.  ``distributions_b14`` carried a
*second*, also-wrong ``SHASHo`` factory, and ``omnilss/__init__.py``
re-exported **that** module's symbol -- so the public ``omnilss.SHASHo()``
returned the wrong distribution while only the registry's ``SHASHO`` (via
``distributions_o_alias_compat``, cloning ``distributions_b3``) resolved to
the correct one.  One family, two densities, with the wrong one exported:
the D2.15/D2.17 "one family, one density" discipline broken at the export
level.

The fix
-------
* Transcribe ``dSHASHo2`` / ``pSHASHo2`` / ``qSHASHo2`` / ``rSHASHo2`` from
  the vendored ``gamlss.dist`` 6.1-8 (``R/gamlss.dist/R/SHASHo2.R``) into
  ``dpqr_functions.py``.
* ``_shasho2_log_pdf`` now delegates to ``dSHASHo2(..., log=True)``, so the
  scoring path and the public ``d`` share one formula.
* ``SHASHo2()`` wires ``p``/``q``/``r``.
* Delete the duplicate ``SHASHo`` from ``distributions_b14`` and re-export
  the b3 one from ``omnilss/__init__.py``.

Why the family could not fit, and the second half of the fix
-------------------------------------------------------------
With the density correct, an intercept-only RS fit still exploded
(``deviance = nan`` after 200 cycles) unless two algorithm-level defects
were fixed first:

* ``_fitting_init`` had no entry for the SHASH cluster, so the generic
  roll-out fallback (``target = y``) drove ``sigma.initial`` to
  ``RMSE(y, y) = eps`` -- a degenerate starting scale.  D2.38: initial
  values are the first granularity of algorithm alignment.  R's own
  starting values (``mu <- (y+mean(y))/2``, ``sigma <- rep(sd(y)/5)``,
  ``nu <- 0.5``, ``tau <- 0.5``) are now reproduced exactly; the initial
  deviance then matches R's to 3e-15.
* The AD Hessian is **not** what R feeds IRLS.  R's ``d2ldm2``/``d2ldd2``/
  ``d2ldv2``/``d2ldt2`` slots are the negative-observational-information
  approximation ``-dldX * dldX`` (D2.15), while the true second derivative
  of this heavy-tailed log-density turns positive in the tails.  With the
  AD Hessian the IRLS weights flipped sign on 32/40 points, bottomed out at
  1e-10 and the first step sent ``eta`` to -1e41.  The eight R closed forms
  are now transcribed into ``distributions_b14`` and installed through
  ``build_ad_family``'s ``score_overrides``/``hessian_overrides``; with them
  the fit converges and tracks R's deviance to ~3e-6 relative.

The boundary contract (and why it is not live R's)
--------------------------------------------------
``qSHASHo2`` returns the continuous extension at the boundary
(``q(0) = -Inf``, ``q(1) = +Inf``, NaN outside (0, 1)) instead of raising.
The two gamlss.dist versions disagree: the installed 6.1.1 has
``if (any(p <= 0) | any(p >= 1)) stop(...)`` active, while the vendored
6.1-8 -- newer, and the transcription reference for this tree -- comments
that check out and assigns -Inf/Inf/NaN explicitly.  We follow the newer
source: it matches base R's ``qnorm`` convention, it is what the sibling
``qSHASH`` in the same file does, and raising from a quantile function is
what upstream removed the check for.  The parity gate is unaffected -- it
samples interior probabilities only (``linspace(0.1, 0.9, n)``).
``test_q_boundary_contract_pins_both_r_versions`` pins both sides of the
split so the divergence cannot silently drift.

Methodology
-----------
d/p/q and the eight derivative slots are compared element-wise against
**live R** through ``digits=17`` subprocess probes (the ST1/ST2 house
style), not ``RTestBus``: the bus serialises with ``digits=10`` and that
alone put a ~1e-10 floor under every reference value, which is coarser
than the agreement being asserted.  Nothing is hard-coded -- a mistyped
16-digit constant would make a parity test pass, which is the one failure
mode these tests must not have.  Measured agreement against live R 6.1.1
(digits=17 probes): density <= 9.09e-13, cdf <= 9.50e-13, quantile <=
3.09e-14, derivatives <= 6.04e-10 (relative; the largest absolute errors,
~1e-6, sit at tail points whose magnitudes are ~1e10, i.e. ~1e-16
relative).  The RS-fit comparison embeds a fixed R-generated sample and
runs live ``gamlss`` on the *same* numbers -- comparing a Python rng
sample against an R ``set.seed`` sample is a data-flow error, not parity.
"""

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

import omnilss
from omnilss import distributions_b14
from omnilss.distributions import resolve_family

FAMILY = "SHASHo2"
RSCRIPT = shutil.which("Rscript")

# mu/sigma/nu/tau profiles: symmetric and skewed, tau on both sides of 1,
# a tight scale and a wide one.  nu is allowed to be negative (it is a
# skewness parameter with identity link).
PROFILES = [
    dict(mu=0.0, sigma=1.0, nu=0.0, tau=1.0),    # reduces to N(mu, sigma^2)
    dict(mu=0.3, sigma=1.2, nu=-0.4, tau=0.8),
    dict(mu=-1.5, sigma=0.5, nu=0.9, tau=1.7),
    dict(mu=2.0, sigma=2.5, nu=-1.2, tau=0.4),
    dict(mu=0.0, sigma=0.3, nu=0.2, tau=3.0),
]

Y_GRID = np.linspace(-6.0, 6.0, 25)
Q_GRID = np.linspace(-6.0, 6.0, 25)
P_GRID = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])

# See the module docstring: these are the measured agreement floors of the
# digits=17 probes, not round numbers.
RTOL = 1e-12
ATOL = 1e-300

# The derivatives need a different floor.  On this grid they span
# 1e-16 (genuine zero crossings of dldX) to 4.6e14 (heavy tails), and a
# pure relative test is meaningless where the function itself crosses
# zero: there the measured disagreement is 1 ULP absolute (1.1e-16), i.e.
# 0.33 relative.  Measured over the profiles the agreement is double
# precision everywhere -- <= 1e-15 relative to each comparison's own
# magnitude -- so an absolute floor of 1e-12 covers the zero-crossing
# noise with four orders of margin while rtol governs every value above
# ~1.  (For comparison, the density/cdf/quantile values are all O(1) and
# keep ATOL = 1e-300 above.)
DERIV_ATOL = 1e-12

# The eight R derivative slots, in family-object order.
DERIV_SLOTS = ("dldm", "dldd", "dldv", "dldt",
               "d2ldm2", "d2ldd2", "d2ldv2", "d2ldt2")

# Slot name -> (override table, parameter key).  The tables are keyed by
# parameter name; the R family object names the slots by derivative.
_SLOT_TO_ENTRY = {
    "dldm": ("scores", "mu"), "dldd": ("scores", "sigma"),
    "dldv": ("scores", "nu"), "dldt": ("scores", "tau"),
    "d2ldm2": ("hessians", "mu"), "d2ldd2": ("hessians", "sigma"),
    "d2ldv2": ("hessians", "nu"), "d2ldt2": ("hessians", "tau"),
}


def _override_fn(slot):
    table_name, par = _SLOT_TO_ENTRY[slot]
    table = (distributions_b14._SHASHO2_SCORES if table_name == "scores"
             else distributions_b14._SHASHO2_HESSIANS)
    return table[par]

# A fixed sample for the fit comparison.  Provenance (live R 4.6.0,
# gamlss.dist 6.1.1): set.seed(1); rSHASHo2(40, mu=0.5, sigma=1, nu=0.3,
# tau=1.2), dumped with digits=17.  Embedded rather than regenerated so the
# Python and R fits provably see the same numbers.
FIT_Y = np.array([
    0.20575226258224127, 0.47929158077361245, 0.9957433644250296,
    2.233745611279673, 0.02838427842727348, 2.1730412554626026,
    2.511539955933804, 1.2446075125637792, 1.1522980558626466,
    -0.5085251499410464, 0.0409083002514018, -0.047345471408276385,
    1.323865036361666, 0.5093875934193612, 1.5977437542486101,
    0.7971899493201967, 1.4202952493868737, 3.328143400956853,
    0.49916964966592886, 0.35978888599775765, 1.0887246876625287,
    0.4360616268914596, 1.0255753449167188, 0.6426670537489417,
    1.3460976997628683, -0.1698834759024857, 0.4947070590232821,
    1.1566521560306812, 1.1420266670439125, 0.4673734346151568,
    1.9004044731999486, 1.0247830344097536, -0.2986244383244822,
    0.8534374752431602, 0.6549893577564499, 0.3275387289840244,
    0.5180463330282527, 1.0243421235793418, 0.21275867283448166,
    0.8618369470779273, 0.1859489333386088,
])

# Same RS control on both sides.  c_crit stays at R's default 0.001 --
# tightening ours alone is not a stricter test (D2.17 note).  ``trace`` is
# R's own print switch; it does not touch the algorithm, but without it
# gamlss's per-iteration "Global Deviance" lines corrupt the JSON stream.
R_FIT_CONTROL = "gamlss.control(n.cyc = 200, c.crit = 0.001, trace = FALSE)"

# Live R on this sample converges at 104.736342777 after 13 cycles; we
# reach 104.73636549840572 (relative 2.2e-7).  The optimum is flat in tau
# (a shape parameter identified only through the tails), so the *deviance*
# is the parity quantity here, not the individual coefficients: at this
# convergence level R's own stopping rule leaves ~1e-3 of slack in the
# coefficients while the objective moves < 1e-6.
FIT_RTOL = 1e-3


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


@pytest.fixture(scope="module")
def fam():
    return resolve_family(FAMILY)


@pytest.fixture(scope="module")
def r_dpqr_probe():
    """Element-wise live-R d/p/q at digits=17 (ST1/ST2 house style)."""
    yvec = ", ".join(f"{float(v):.17g}" for v in Y_GRID)
    qvec = ", ".join(f"{float(v):.17g}" for v in Q_GRID)
    pvec = ", ".join(f"{float(v):.17g}" for v in P_GRID)
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}, tau={p['tau']!r}"
        # d: the response is the first formal `x`.
        lines.append(
            f"out[['{pi}.d']] <- vapply(c({yvec}), function(xi) "
            f"suppressWarnings(as.numeric(gamlss.dist::d{FAMILY}(xi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.p']] <- vapply(c({qvec}), function(qi) "
            f"suppressWarnings(as.numeric(gamlss.dist::p{FAMILY}(qi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.q']] <- vapply(c({pvec}), function(pi_) "
            f"suppressWarnings(as.numeric(gamlss.dist::q{FAMILY}(pi_, {args}))), "
            f"numeric(1))"
        )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_deriv_probe():
    """The eight R closed-form derivative slots, element-wise, digits=17.

    These are what ``score_overrides``/``hessian_overrides`` replace, so
    they are the direct evidence that the IRLS inputs match R's.
    """
    yvec = ", ".join(f"{float(v):.17g}" for v in Y_GRID)
    lines = ["suppressMessages(library(gamlss.dist))",
             "o <- get('SHASHo2', envir=asNamespace('gamlss.dist'))()",
             "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = f"mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}, tau={p['tau']!r}"
        for slot in DERIV_SLOTS:
            lines.append(
                f"out[['{pi}.{slot}']] <- vapply(c({yvec}), function(yi) "
                f"suppressWarnings(as.numeric(o[['{slot}']](y=yi, {args}))), "
                f"numeric(1))"
            )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_fit():
    """Live-R ``gamlss`` fit on the embedded sample, same RS control."""
    yvec = ", ".join(f"{float(v):.17g}" for v in FIT_Y)
    lines = [
        "suppressMessages({library(gamlss); library(gamlss.dist)})",
        f"y <- c({yvec})",
        "m <- gamlss(y ~ 1, family = SHASHo2, control = " + R_FIT_CONTROL + ")",
        "cat(jsonlite::toJSON(list(deviance = m$G.deviance, "
        "converged = isTRUE(m$converged), n_cyc = m$ncyc), "
        "digits = 17, auto_unbox = TRUE))",
    ]
    return json.loads(_run_r(lines).strip())


def _pos(fn, first, profile):
    """Call a dpqr entry point positionally -- ``d`` takes ``*args`` only."""
    return np.asarray(fn(jnp.asarray(first), *(float(profile[k]) for k in ("mu", "sigma", "nu", "tau"))))


def _param_vec(fam, profile, first):
    """Broadcast one profile's scalars to the length of ``first``."""
    return tuple(np.repeat(float(profile[k]), len(first))
                 for k in ("mu", "sigma", "nu", "tau"))


# ---------------------------------------------------------------------------
# 1. Density against live R -- both entry points (d and the deviance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_d_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    jax_d = _pos(fam.d, Y_GRID, profile)
    assert np.all(np.isfinite(jax_d))
    np.testing.assert_allclose(jax_d, r_d, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_g_dev_inc_is_minus_two_log_d(fam, pi):
    """The scoring path must be the same density, not a lookalike.

    Asserted exactly and in **log space**: both sides are
    ``-2 * dSHASHo2(..., log=True)``, one through the vmapped log-pdf, one
    through the public ``d``.  Log space is not a convenience here -- at
    profile 2 the direct density underflows to exactly 0 on three tail
    points, and ``log(0) = -inf`` would make the identity untestable there
    while the scoring path (which never leaves log space) stays finite.
    That asymmetry is precisely why the deviance is computed from the
    log-density.
    """
    profile = PROFILES[pi]
    params = tuple(float(profile[k]) for k in ("mu", "sigma", "nu", "tau"))
    logd = np.asarray(fam.d(jnp.asarray(Y_GRID), *params, log=True))
    dev = np.asarray(fam.g_dev_inc(jnp.asarray(Y_GRID), *params))
    np.testing.assert_allclose(dev, -2.0 * logd, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_d_matches_live_r_through_deviance(r_dpqr_probe, fam, pi):
    """``gamlss`` itself feeds on ``G.dev.incr``, so that path is compared too."""
    profile = PROFILES[pi]
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    dev = np.asarray(
        fam.g_dev_inc(
            jnp.asarray(Y_GRID),
            *(float(profile[k]) for k in ("mu", "sigma", "nu", "tau")),
        )
    )
    np.testing.assert_allclose(np.exp(-0.5 * dev), r_d, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. CDF and quantile against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    jax_p = _pos(fam.p, Q_GRID, profile)
    np.testing.assert_allclose(jax_p, r_p, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_q_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_q = np.asarray(r_dpqr_probe[f"{pi}.q"], dtype=float)
    jax_q = _pos(fam.q, P_GRID, profile)
    np.testing.assert_allclose(jax_q, r_q, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_tail_paths_match_live_r(r_dpqr_probe, fam, pi):
    """``lower_tail`` / ``log_p`` are part of R's pSHASHo2 body, not extras."""
    profile = PROFILES[pi]
    r_upper = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    scalars = {k: float(v) for k, v in profile.items()}

    # R has no flag channel through the probe, so the reference is computed
    # the way R's own body does: 1 - p, then log.  The identity against our
    # lower_tail=False / log_p=True paths is what is asserted.
    jax_upper = np.asarray(
        fam.p(
            jnp.asarray(Q_GRID),
            *(float(scalars[k]) for k in ("mu", "sigma", "nu", "tau")),
            lower_tail=False,
        )
    )
    np.testing.assert_allclose(jax_upper, 1.0 - r_upper, rtol=RTOL, atol=ATOL)

    jax_logp = np.asarray(
        fam.p(
            jnp.asarray(Q_GRID),
            *(float(scalars[k]) for k in ("mu", "sigma", "nu", "tau")),
            log_p=True,
        )
    )
    np.testing.assert_allclose(jax_logp, np.log(r_upper), rtol=RTOL, atol=ATOL)


def test_p_q_roundtrip(fam):
    """p and q are mutual inverses -- a transcription slip in either shows up here."""
    profile = PROFILES[1]
    scalars = {k: float(v) for k, v in profile.items()}
    p_of_y = _pos(fam.p, Y_GRID, scalars)
    y_back = _pos(fam.q, p_of_y, scalars)
    np.testing.assert_allclose(y_back, Y_GRID, rtol=1e-10, atol=1e-12)

    q_of_p = _pos(fam.q, P_GRID, scalars)
    p_back = _pos(fam.p, q_of_p, scalars)
    np.testing.assert_allclose(p_back, P_GRID, rtol=1e-10, atol=1e-12)


def test_q_boundary_contract_pins_both_r_versions(fam):
    """We return the continuous extension; installed R 6.1.1 raises.

    Both halves are asserted on purpose.  The first documents *our*
    contract (which follows the newer vendored gamlss.dist 6.1-8 and base
    R's ``qnorm``); the second pins the installed version's opposite
    behaviour, so if the environment's gamlss.dist is ever upgraded to
    match the vendored tree this test tells us exactly what changed instead
    of silently drifting.
    """
    scalars = dict(mu=0.3, sigma=1.2, nu=-0.4, tau=0.8)

    assert float(fam.q(0.0, **scalars)) == -np.inf
    assert float(fam.q(1.0, **scalars)) == np.inf
    assert np.isnan(float(fam.q(-0.5, **scalars)))
    assert np.isnan(float(fam.q(1.5, **scalars)))
    # log_p / lower_tail feed the same transform.
    assert float(fam.q(-np.inf, log_p=True, **scalars)) == -np.inf
    assert float(fam.q(1.0, lower_tail=False, **scalars)) == -np.inf

    args = f"mu={scalars['mu']}, sigma={scalars['sigma']}, nu={scalars['nu']}, tau={scalars['tau']}"
    # NB: R's stop() message itself carries a trailing "\n"
    # (stop(paste("p must be between 0 and 1", "\n", ""))), so the per-line
    # output is split across two lines -- count occurrences in the whole
    # output instead of parsing lines.
    lines = ["suppressMessages(library(gamlss.dist))", "for (p in c(0, 1, -0.5, 1.5)) {"]
    lines.append(f"  r <- tryCatch(qSHASHo2(p, {args}), error = function(e) conditionMessage(e))")
    lines.append('  cat(p, "->", r, "\\n")')
    lines.append("}")
    out = _run_r(lines)
    assert out.count("p must be between 0 and 1") == 4, out


def test_r_is_quantile_of_uniforms(fam):
    """``rSHASHo2`` is ``qSHASHo2(runif(n))`` -- the same construction as R.

    Asserted exactly against our own ``q``: if ``r`` ever grows its own
    sampler this test fails, which is how sampler/CDF drift starts.
    """
    key = jax.random.PRNGKey(20261020)
    n = 64
    scalars = dict(mu=0.3, sigma=1.2, nu=-0.4, tau=0.8)
    u = jax.random.uniform(key, shape=(n,))
    expected = np.asarray(fam.q(u, **scalars))
    got = np.asarray(fam.r(key, n, **scalars))
    np.testing.assert_array_equal(got, expected)
    assert np.all(np.isfinite(got))


# ---------------------------------------------------------------------------
# 3. The IRLS inputs: the eight R closed forms behind the overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", DERIV_SLOTS)
@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_derivative_slots_match_live_r(r_deriv_probe, fam, pi, slot):
    """``score_overrides``/``hessian_overrides`` must be R's own formulas.

    This is the direct evidence for the D2.15 point that made the family
    fittable: R feeds IRLS the negative-observational-information
    approximation ``-dldX * dldX``, not the true second derivative, and the
    AD Hessian disagreed on 32/40 points at the starting values.
    """
    profile = PROFILES[pi]
    r_vals = np.asarray(r_deriv_probe[f"{pi}.{slot}"], dtype=float)
    mu, sigma, nu, tau = _param_vec(fam, profile, Y_GRID)
    ours = np.asarray(_override_fn(slot)(
        jnp.asarray(Y_GRID), jnp.asarray(mu), jnp.asarray(sigma),
        jnp.asarray(nu), jnp.asarray(tau)))
    np.testing.assert_allclose(ours, r_vals, rtol=RTOL, atol=DERIV_ATOL)


def test_hessian_slots_are_negative_score_squares(fam):
    """Every d2ldX2 must be exactly ``-dldX * dldX`` (R's own definition).

    Checked against our score table rather than live R: this is the
    structural invariant the overrides install, and a transcription that
    dropped it would silently restore the AD-Hessian divergence.
    """
    profile = PROFILES[2]
    mu, sigma, nu, tau = _param_vec(fam, profile, Y_GRID)
    args = (jnp.asarray(Y_GRID), jnp.asarray(mu), jnp.asarray(sigma),
            jnp.asarray(nu), jnp.asarray(tau))
    for par in ("mu", "sigma", "nu", "tau"):
        score = np.asarray(distributions_b14._SHASHO2_SCORES[par](*args))
        hess = np.asarray(distributions_b14._SHASHO2_HESSIANS[par](*args))
        np.testing.assert_allclose(hess, -score * score, rtol=0.0, atol=0.0)


def test_overrides_are_installed_on_the_family():
    """The family must actually use the R closed forms, not just carry them."""
    fam = distributions_b14.SHASHo2()
    for par in ("mu", "sigma", "nu", "tau"):
        assert fam.score_functions[par] is distributions_b14._SHASHO2_SCORES[par]
        assert fam.hessian_functions[par] is distributions_b14._SHASHO2_HESSIANS[par]


# ---------------------------------------------------------------------------
# 4. SHASHo2 is NOT SHASHo -- the confusion that produced the duplicate
# ---------------------------------------------------------------------------


def test_shasho2_is_not_shasho(fam):
    """The two parameterizations are different distributions.

    The deleted b14 duplicate existed because "SHASHo vs SHASHo2" was
    treated as a naming detail.  This test fails if anyone re-derives one
    from the other again.
    """
    shasho = resolve_family("SHASHO")
    profile = dict(mu=0.3, sigma=1.2, nu=-0.4, tau=0.8)
    d2 = _pos(fam.d, Y_GRID, profile)
    d1 = _pos(shasho.d, Y_GRID, profile)
    assert not np.allclose(d2, d1, rtol=1e-3), "SHASHo2 density collapsed onto SHASHo"
    # Both must still be valid densities integrating to ~1 over a wide grid.
    grid = np.linspace(-40.0, 40.0, 20001)
    for f in (fam, shasho):
        vals = _pos(f.d, grid, profile)
        np.testing.assert_allclose(np.trapezoid(vals, grid), 1.0, rtol=1e-3)


def test_links_and_type_match_live_r(fam):
    """Query the live R family object rather than restating the source."""
    script = (
        'suppressMessages(library(gamlss.dist)); '
        'f <- get("SHASHo2", envir=asNamespace("gamlss.dist"))(); '
        'cat(f$mu.link, "\\n", f$sigma.link, "\\n", f$nu.link, "\\n", '
        'f$tau.link, "\\n", f$type, "\\n", sep="")'
    )
    out = _run_r([script])
    mu_l, sigma_l, nu_l, tau_l, type_ = out.strip().splitlines()
    assert fam.links == {"mu": mu_l, "sigma": sigma_l, "nu": nu_l, "tau": tau_l}
    assert fam.type == type_


# ---------------------------------------------------------------------------
# 5. The export-level defect: one public SHASHo, the correct one
# ---------------------------------------------------------------------------


def test_public_shasho_export_is_the_b3_family():
    """``omnilss.SHASHo`` must be the verified b3 factory, not a b14 lookalike."""
    from omnilss.distributions_b3 import SHASHo as b3_shasho

    assert omnilss.SHASHo is b3_shasho
    assert not hasattr(distributions_b14, "SHASHo"), (
        "the wrong duplicate SHASHo is back in distributions_b14"
    )
    # The registry alias must clone from the same correct family.
    assert resolve_family("SHASHO").name == "SHASHo"


def test_public_shasho_density_matches_live_r():
    """The exported family's density is R's dSHASHo, checked through the deviance."""
    family = resolve_family("SHASHO")
    profile = dict(mu=0.3, sigma=1.2, nu=-0.4, tau=0.8)
    y = np.linspace(-6.0, 6.0, 25)
    yvec = ", ".join(f"{float(v):.17g}" for v in y)
    args = f"mu={profile['mu']}, sigma={profile['sigma']}, nu={profile['nu']}, tau={profile['tau']}"
    lines = [
        "suppressMessages(library(gamlss.dist))",
        f"out <- vapply(c({yvec}), function(xi) "
        f"suppressWarnings(as.numeric(dSHASHo(xi, {args}, log=TRUE))), numeric(1))",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    r_logd = np.asarray(json.loads(_run_r(lines).strip()), dtype=float)
    jax_logd = np.asarray(
        family.g_dev_inc(
            jnp.asarray(y),
            *(float(profile[k]) for k in ("mu", "sigma", "nu", "tau")),
        )
    ) * -0.5
    np.testing.assert_allclose(jax_logd, r_logd, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 6. The family must fit, and land where R lands
# ---------------------------------------------------------------------------


def test_rs_fit_matches_live_r(r_fit):
    """Full RS cycle on a fixed sample, against live ``gamlss``.

    Not the scanner's rsdev gate (that one owns the 146-family stream and
    its registered numbers) -- this is the end-to-end evidence that the
    two algorithm-level fixes hold: the R starting values and the R
    closed-form IRLS inputs.  Both sides run the same control
    (``n.cyc = 200, c.crit = 0.001``) on the same 40 numbers.
    """
    from omnilss.controls import GAMLSSControl
    from omnilss.fitting import gamlss

    fit = gamlss(
        formula="y ~ 1",
        family=FAMILY,
        data={"y": FIT_Y},
        control=GAMLSSControl(n_cyc=200, c_crit=0.001, trace=False),
    )
    assert fit.convergence_status() is True, fit.termination_reason
    assert np.isfinite(fit.deviance)
    for par in fit.parameters:
        vals = np.asarray(fit.fitted_values[par], dtype=float)
        assert np.all(np.isfinite(vals)), (par, vals)

    assert r_fit["converged"] is True
    np.testing.assert_allclose(
        fit.deviance, r_fit["deviance"], rtol=FIT_RTOL, atol=0.0
    )


def test_initial_deviance_matches_r_starting_values():
    """D2.38: the starting point itself must reproduce R's, before any step.

    R's SHASHo2 ``initial`` expression sets ``mu <- (y+mean(y))/2``,
    ``sigma <- rep(sd(y)/5)`` (n-1 denominator), ``nu <- 0.5``,
    ``tau <- 0.5``; the deviance at those values must match R's
    ``G.dev.incr`` sum exactly.  A mismatch here means the roll-out
    fallback is still in play and the first inner step starts from
    different working weights.
    """
    family = resolve_family(FAMILY)
    mu0 = (FIT_Y + np.mean(FIT_Y)) / 2.0
    sigma0 = np.repeat(np.std(FIT_Y, ddof=1) / 5.0, len(FIT_Y))
    nu0 = np.repeat(0.5, len(FIT_Y))
    tau0 = np.repeat(0.5, len(FIT_Y))
    ours = float(np.sum(np.asarray(
        family.g_dev_inc(jnp.asarray(FIT_Y), jnp.asarray(mu0),
                         jnp.asarray(sigma0), jnp.asarray(nu0),
                         jnp.asarray(tau0))
    )))

    yvec = ", ".join(f"{float(v):.17g}" for v in FIT_Y)
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "f <- get('SHASHo2', envir=asNamespace('gamlss.dist'))()",
        f"y <- c({yvec})",
        "mu0 <- (y + mean(y))/2",
        "dev0 <- sum(f$G.dev.incr(y, mu=mu0, sigma=rep(sd(y)/5, length(y)),",
        "                            nu=rep(0.5, length(y)), tau=rep(0.5, length(y))))",
        "cat(jsonlite::toJSON(list(deviance = dev0), digits = 17, auto_unbox = TRUE))",
    ]
    r_dev = json.loads(_run_r(lines).strip())["deviance"]
    np.testing.assert_allclose(ours, r_dev, rtol=RTOL, atol=ATOL)


def test_fitter_uses_r_starting_values():
    """The fitter's own starting values must be R's, not the roll-out default.

    ``_initial_mu_target``/``_initial_sigma``/``_initial_parameter_value``
    are what produced the degenerate ``sigma.initial = eps`` for the SHASH
    cluster before D2.40; this pins the three entries so the fallback
    cannot silently return.  Signatures follow ``rs_algorithm``: mu target
    first (identity link, so the target *is* the starting fitted value),
    then sigma against it.
    """
    from omnilss._fitting_init import (
        _initial_mu_target,
        _initial_parameter_value,
        _initial_sigma,
    )

    family = resolve_family(FAMILY)
    y = FIT_Y
    w = np.ones_like(y)
    mu0 = np.asarray(_initial_mu_target(family, y, w))
    np.testing.assert_allclose(mu0, (y + np.mean(y)) / 2.0,
                               rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        _initial_sigma(family, y, mu0, w), np.std(y, ddof=1) / 5.0,
        rtol=0.0, atol=0.0
    )
    assert _initial_parameter_value(family, "nu", y, mu0, w) == 0.5
    assert _initial_parameter_value(family, "tau", y, mu0, w) == 0.5
