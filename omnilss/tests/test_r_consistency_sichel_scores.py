"""SICHEL / ZISICHEL / ZASICHEL IRLS slots against live R at digits=17 (D2.44).

What this replaces
------------------
Gate 2 of ``tools/rs_family_parity_scan.py`` measured the derivative slots of
these three families off by 0.55-1.27 rel.  The root cause is *not* the
densities -- they are parity-verified -- but ``bessel.py``'s custom JVP for
``log K_nu``: ``_log_bessel_kv_jvp`` substitutes the *asymptotic* tangent
``-1 - nu/(x + |nu| + 1)`` for the true ``d/dx log K_nu = -(K_{nu+1}/K_nu) -
nu/x`` and drops the ``nu`` tangent entirely.  The primal is exact (scipy),
so every density is right while every AD score is wrong -- and SICHEL's
log-density reaches ``log(K_nu(1/sigma))``, so all three families inherit
the defect (SI and GIG still do; repairing the common JVP is a separate
task, not this delivery).  The fix installs R's closed forms as
``score_overrides``/``hessian_overrides`` -- the D2.40/D2.42 pattern -- which
keeps the IRLS working weights on R's own formulas, R's approximations
included, which is what parity means here.

Five contract details live R 6.1.1 has, all measured
----------------------------------------------------
1. ``dldv`` is a *forward* difference.  gamlss's ``numeric.deriv`` -- dumped
   from the installed namespace -- is ``(f(x+delta) - f(x))/delta``, not
   stats' central difference; ``delta = 0.001``.  Section 1 pins the
   convention on live R; section 6 pins our replication of it.
2. ``d2ldX2 = -dldX^2`` clamped at ``-1e-15`` (R's negative-information
   approximation) for every diagonal slot -- with one exception:
   ZASICHEL's ``d2ldt2`` is ``-1/(tau*(1-tau))``.
3. Each SICHEL slot first applies
   ``sigma <- ifelse(sigma > 10000 & nu > 0, 10000, sigma)``; ZASICHEL's
   ``dldd``/``d2ldd2`` additionally floor ``sigma`` at ``1e-6``.
4. The zero variants branch at ``y == 0``.  ZISICHEL scales the base slot by
   ``(1-tau)*f0/(tau + (1-tau)*f0)`` with ``f0 = dSICHEL(0, ...)``, and its
   ``dldt`` becomes ``(1-f0)/(tau+(1-tau)*f0)``; ZASICHEL *adds*
   ``f0*base(0)/(1-f0)`` to the base slot, forces ``dldm = dldd = 0`` at
   ``y == 0``, and its ``dldt`` becomes ``1/tau`` there.
5. DBURR12 is *not* SICHEL: it is the discrete Burr XII with survival
   ``Sur(y) = (1+(y/mu)^sigma)^(-nu)`` and ``nu > 0`` required, while the
   registry currently aliases it onto the SICHEL kernel.  That mismatch is
   registered as a follow-up delivery, not pinned here -- the alias tests
   live in ``test_dburr12_compat.py``.

Methodology
-----------
Slot values are compared element-wise against **live R** through
``digits=17`` subprocess probes (the ST1/ST2 house style): the scripts build
the family objects (``SICHEL()``, ``ZISICHEL()``, ``ZASICHEL()``) and call
their own ``dldm``/``dldd``/``dldv``/``dldt``/``d2ldX2`` slots.  Nothing is
hard-coded; a mistyped constant cannot make these tests pass.  Cross terms
(``d2ldmdd`` etc.) are not transcribed -- RS consumes the diagonal Hessian
only -- so they are not probed either.

Known gap, registered rather than hidden: the scan's gate 3 (RS fit
deviance vs R) stays ``skip`` for all three families because **live R
itself** aborts with ``NA/NaN/Inf in foreign function call`` on the scan's
sample -- there is no R reference to compare against (measured: R fits
SICHEL at dev 128.2298 and ZASICHEL at 93.3715 on a clean sample, but
ZISICHEL aborts on a zero-heavy one).  Fit-level parity for SICHEL /
ZASICHEL is additionally blocked on our side by the 100k-step masked scan
in ``_sichel_log_pdf``/``tofySICHEL``, which makes an IRLS fit take
10-20 minutes; both are follow-up work, so this file stops at the slots.
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
from omnilss.distributions_b6 import (  # noqa: E402
    _SICHEL_HESSIANS,
    _SICHEL_SCORES,
    _sichel_log_pdf,
)
from omnilss.distributions_b10_zero_variants import (  # noqa: E402
    _ZASICHEL_HESSIANS,
    _ZASICHEL_SCORES,
    _ZISICHEL_HESSIANS,
    _ZISICHEL_SCORES,
    _zero_sichel_log_pdf,
)

RSCRIPT = shutil.which("Rscript")

#: (mu, sigma, nu) profiles.  SICHEL accepts any real nu (unlike DBURR12,
#: which requires nu > 0), so the grid spans negative and positive shape.
PROFILES = [
    dict(mu=1.0, sigma=0.5, nu=-0.5),
    dict(mu=2.0, sigma=1.2, nu=0.3),
    dict(mu=0.7, sigma=0.3, nu=-1.5),
    dict(mu=3.0, sigma=2.0, nu=2.0),
]

#: The zero-inflation probability shared by the two zero-variant profiles.
TAU = 0.3

#: Counts 0..9 -- y = 0 first, because the zero variants branch there and
#: the branch column is asserted separately from the interior columns.
Y_GRID = np.arange(10, dtype=float)

#: (R slot name, our parameter key) for the score slots.
SCORE_SLOTS = (("dldm", "mu"), ("dldd", "sigma"), ("dldv", "nu"))
#: ...and for the diagonal Hessian slots.
HESS_SLOTS = (("d2ldm2", "mu"), ("d2ldd2", "sigma"), ("d2ldv2", "nu"))
#: The zero variants add the tau slot to both tables.
ZERO_SCORE_SLOTS = SCORE_SLOTS + (("dldt", "tau"),)
ZERO_HESS_SLOTS = HESS_SLOTS + (("d2ldt2", "tau"),)

#: Measured agreement floors of the digits=17 probes, not round numbers:
#: mu <= 4.2e-14, sigma <= 1.1e-12, tau <= 1.6e-14 across the profiles.
#: nu is a forward difference on *both* sides, so the cross-library noise
#: is amplified by 2/delta = 2000: measured <= 7.6e-8 (score) and
#: <= 1.5e-7 (hessian) -- hence 1e-6 for the nu slots.
SLOT_RTOL = {"mu": 1e-12, "sigma": 1e-11, "nu": 1e-6, "tau": 1e-12}
ATOL = 1e-300

#: gamlss's ``numeric.deriv`` default step, as used by every ``dldv`` slot.
_FORWARD_DELTA = 0.001


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


def _slot_probe_lines(r_name, slots, profiles, tau=None):
    """Build the R lines that dump ``r_name``'s own family-object slots."""
    lines = [
        "suppressMessages(library(gamlss.dist))",
        f"o <- get('{r_name}', envir=asNamespace('gamlss.dist'))()",
        "out <- list()",
    ]
    ys = ", ".join(str(int(v)) for v in Y_GRID)
    for pi, p in enumerate(profiles):
        args = f"mu={p['mu']!r}, sigma={p['sigma']!r}, nu={p['nu']!r}"
        if tau is not None:
            args += f", tau={tau!r}"
        for slot in slots:
            lines.append(
                f"out[['{pi}.{slot}']] <- vapply(c({ys}), function(yi) "
                f"suppressWarnings(as.numeric(o[['{slot}']](yi, {args}))), "
                f"numeric(1))"
            )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return lines


def _ours_slots(table, profile, tau=None):
    """Evaluate one of our override tables on the shared grid."""
    args = [jnp.asarray(Y_GRID), float(profile["mu"]), float(profile["sigma"]),
            float(profile["nu"])]
    if tau is not None:
        args.append(float(tau))
    return {key: np.asarray(fn(*args)) for key, fn in table.items()}


def _compare(probe, pi, slot_pairs, table, profile, tau=None):
    ours = _ours_slots(table, profile, tau=tau)
    for slot, param in slot_pairs:
        r_vals = np.asarray(probe[f"{pi}.{slot}"], dtype=float)
        np.testing.assert_allclose(ours[param], r_vals,
                                   rtol=SLOT_RTOL[param], atol=ATOL)


# ---------------------------------------------------------------------------
# Fixtures: the live-R probes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_numeric_deriv_probe():
    """gamlss's ``numeric.deriv`` convention, measured on ``exp``.

    The function is not exported by stats in R 4.6.0, so it is pulled from
    the gamlss namespace exactly as the family slots see it.
    """
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "ND <- get('numeric.deriv', envir=asNamespace('gamlss'))",
        "out <- list()",
    ]
    for nu, tag in ((0.0, "exp0"), (0.7, "exp7")):
        # ``numeric.deriv`` differentiates by re-evaluating its expression in
        # the calling frame, so the variable it names must actually exist:
        # ``ND(exp(0.0), "nu", ...)`` dies with "object 'nu' not found"
        # (measured) -- the symbol has to be bound first.
        lines += [
            f"nu <- {nu!r}",
            "nd <- as.vector(attr(ND(exp(nu), 'nu', delta=0.001), 'gradient'))",
            f"out[['{tag}']] <- nd",
            f"out[['{tag}.fwd']] <- (exp(nu+0.001) - exp(nu))/0.001",
            f"out[['{tag}.cen']] <- (exp(nu+0.001) - exp(nu-0.001))/0.002",
        ]
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_sichel_slots():
    """SICHEL's own dldm/dldd/dldv/d2ldm2/d2ldd2/d2ldv2, digits=17."""
    slots = [s for s, _ in SCORE_SLOTS + HESS_SLOTS]
    return json.loads(_run_r(_slot_probe_lines("SICHEL", slots,
                                               PROFILES)).strip())


@pytest.fixture(scope="module")
def r_zisichel_slots():
    """ZISICHEL's eight diagonal slots (tau = TAU), digits=17."""
    slots = [s for s, _ in ZERO_SCORE_SLOTS + ZERO_HESS_SLOTS]
    return json.loads(_run_r(_slot_probe_lines("ZISICHEL", slots, PROFILES,
                                               tau=TAU)).strip())


@pytest.fixture(scope="module")
def r_zasichel_slots():
    """ZASICHEL's eight diagonal slots (tau = TAU), digits=17."""
    slots = [s for s, _ in ZERO_SCORE_SLOTS + ZERO_HESS_SLOTS]
    return json.loads(_run_r(_slot_probe_lines("ZASICHEL", slots, PROFILES,
                                               tau=TAU)).strip())


@pytest.fixture(scope="module")
def r_family_probe():
    """The live family objects: parameter flags, links, type."""
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for r_name in ("SICHEL", "ZISICHEL", "ZASICHEL"):
        lines += [
            f"o <- get('{r_name}', envir=asNamespace('gamlss.dist'))()",
            f"out[['{r_name}.parameters']] <- "
            "paste(names(o$parameters), o$parameters, collapse=',')",
            f"out[['{r_name}.mu_link']] <- o$mu.link",
            f"out[['{r_name}.sigma_link']] <- o$sigma.link",
            f"out[['{r_name}.nu_link']] <- o$nu.link",
            f"out[['{r_name}.type']] <- o$type",
        ]
        if r_name != "SICHEL":
            lines.append(f"out[['{r_name}.tau_link']] <- o$tau.link")
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


# ---------------------------------------------------------------------------
# 1. The numeric.deriv convention: forward, not central
# ---------------------------------------------------------------------------


def test_gamlss_numeric_deriv_is_a_forward_difference(r_numeric_deriv_probe):
    """``numeric.deriv`` from the gamlss namespace is a forward difference.

    stats::numeric.deriv is a *central* difference; gamlss's own copy --
    dumped from the installed namespace and measured here -- is
    ``(f(x+delta) - f(x))/delta``.  Every ``dldv`` slot in this delivery
    replicates that choice, so the convention itself is pinned first: the
    measured gradient equals the forward difference to ~4e-14 relative and
    differs from the central one by ~5e-4 relative, three orders above the
    probe noise.
    """
    for tag in ("exp0", "exp7"):
        nd = float(r_numeric_deriv_probe[tag])
        fwd = float(r_numeric_deriv_probe[f"{tag}.fwd"])
        cen = float(r_numeric_deriv_probe[f"{tag}.cen"])
        assert nd == pytest.approx(fwd, rel=1e-13, abs=0.0)
        assert abs(nd - cen) > 1e-4 * abs(nd), (tag, nd, cen)


# ---------------------------------------------------------------------------
# 2. SICHEL's slots against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_sichel_score_slots_match_live_r(r_sichel_slots, pi):
    """dldm/dldd/dldv must be R's own formulas at every profile."""
    _compare(r_sichel_slots, pi, SCORE_SLOTS, _SICHEL_SCORES, PROFILES[pi])


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_sichel_hessian_slots_match_live_r(r_sichel_slots, pi):
    """d2ldm2/d2ldd2/d2ldv2 must be R's own negative information."""
    _compare(r_sichel_slots, pi, HESS_SLOTS, _SICHEL_HESSIANS, PROFILES[pi])


# ---------------------------------------------------------------------------
# 3. The zero variants' slots against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_zisichel_score_slots_match_live_r(r_zisichel_slots, pi):
    _compare(r_zisichel_slots, pi, ZERO_SCORE_SLOTS, _ZISICHEL_SCORES,
             PROFILES[pi], tau=TAU)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_zisichel_hessian_slots_match_live_r(r_zisichel_slots, pi):
    _compare(r_zisichel_slots, pi, ZERO_HESS_SLOTS, _ZISICHEL_HESSIANS,
             PROFILES[pi], tau=TAU)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_zasichel_score_slots_match_live_r(r_zasichel_slots, pi):
    _compare(r_zasichel_slots, pi, ZERO_SCORE_SLOTS, _ZASICHEL_SCORES,
             PROFILES[pi], tau=TAU)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_zasichel_hessian_slots_match_live_r(r_zasichel_slots, pi):
    _compare(r_zasichel_slots, pi, ZERO_HESS_SLOTS, _ZASICHEL_HESSIANS,
             PROFILES[pi], tau=TAU)


# ---------------------------------------------------------------------------
# 4. The y == 0 branches: the whole point of the wrappers
# ---------------------------------------------------------------------------


def test_zero_variant_branches_at_y_zero_in_live_r(r_sichel_slots,
                                                   r_zisichel_slots,
                                                   r_zasichel_slots):
    """The y == 0 column follows R's branch formulas, not the base ones.

    A wrapper that "simplified" the zero branch into the base formula would
    still pass the interior columns; these structural assertions are what
    catch it.  R's own values are read from the probes, so the branch
    identities are measured, not restated.
    """
    for pi in range(len(PROFILES)):
        base_mu0 = float(r_sichel_slots[f"{pi}.dldm"][0])
        zi_mu0 = float(r_zisichel_slots[f"{pi}.dldm"][0])
        za_mu0 = float(r_zasichel_slots[f"{pi}.dldm"][0])
        za_dt0 = float(r_zasichel_slots[f"{pi}.dldt"][0])
        zi_dt0 = float(r_zisichel_slots[f"{pi}.dldt"][0])
        # ZISICHEL scales the base slot by (1-tau)*f0/(tau+(1-tau)*f0) --
        # with tau = 0.3 and these f0 the factor is far from 1.
        assert zi_mu0 != pytest.approx(base_mu0, rel=1e-6), (
            pi, zi_mu0, base_mu0)
        # ZASICHEL forces the mu/sigma scores to exactly 0 at y == 0...
        assert za_mu0 == 0.0, (pi, za_mu0)
        assert float(r_zasichel_slots[f"{pi}.dldd"][0]) == 0.0, pi
        # ...and its dldt at y == 0 is exactly 1/tau (ZISICHEL's is the
        # f0-dependent hurdle branch: positive here, and never the y > 0
        # value -1/(1-tau)).
        assert za_dt0 == pytest.approx(1.0 / TAU, rel=1e-12), (pi, za_dt0)
        assert zi_dt0 > 0.0, (pi, zi_dt0)
        assert zi_dt0 != pytest.approx(-1.0 / (1.0 - TAU), rel=1e-6), (pi, zi_dt0)
        # The interior columns are the base formulas for ZISICHEL
        # (R: ifelse(y==0, dldm0, SICHEL()$dldm(y, ...))).
        np.testing.assert_allclose(
            np.asarray(r_zisichel_slots[f"{pi}.dldm"][1:], dtype=float),
            np.asarray(r_sichel_slots[f"{pi}.dldm"][1:], dtype=float),
            rtol=1e-12, atol=ATOL)


def test_our_zero_variant_branches_at_y_zero():
    """Our wrappers branch the same way -- independent of the R probes."""
    profile = PROFILES[1]
    y = jnp.asarray(Y_GRID)
    base0 = float(_SICHEL_SCORES["mu"](jnp.asarray(0.0), profile["mu"],
                                       profile["sigma"], profile["nu"]))
    zi = _ours_slots(_ZISICHEL_SCORES, profile, tau=TAU)
    za = _ours_slots(_ZASICHEL_SCORES, profile, tau=TAU)
    assert zi["mu"][0] != pytest.approx(base0, rel=1e-6)
    assert za["mu"][0] == 0.0
    assert za["sigma"][0] == 0.0
    assert za["tau"][0] == pytest.approx(1.0 / TAU, rel=1e-12)
    assert zi["tau"][0] > 0.0
    assert zi["tau"][0] != pytest.approx(-1.0 / (1.0 - TAU), rel=1e-6)


# ---------------------------------------------------------------------------
# 5. The Hessian conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_negative_information_convention(pi):
    """``d2ldX2 = -dldX^2`` clamped at ``-1e-15`` -- every family, every slot.

    R's negative-information approximation: the diagonal Hessian is the
    negated squared score, never a true second derivative, and never more
    negative than ``-1e-15``.  Asserted against our own scores, so it holds
    whether or not R is reachable.
    """
    profile = PROFILES[pi]
    for scores, hessians, tau in (
        (_SICHEL_SCORES, _SICHEL_HESSIANS, None),
        (_ZISICHEL_SCORES, _ZISICHEL_HESSIANS, TAU),
        (_ZASICHEL_SCORES, _ZASICHEL_HESSIANS, TAU),
    ):
        for param in ("mu", "sigma", "nu"):
            args = [jnp.asarray(Y_GRID), float(profile["mu"]),
                    float(profile["sigma"]), float(profile["nu"])]
            if tau is not None:
                args.append(tau)
            score = np.asarray(scores[param](*args))
            # R: ifelse(d2ldX2 < -1e-15, d2ldX2, -1e-15) -- i.e. keep the
            # value when it is *more* negative than -1e-15.
            expected = np.minimum(-score * score, -1e-15)
            got = np.asarray(hessians[param](*args))
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_zasichel_d2ldt2_is_the_binomial_form(r_zasichel_slots):
    """The one slot that is *not* ``-dldt^2``: R writes ``-1/(tau(1-tau))``.

    R's ZASICHEL source carries the comment ``# or -1/((1-tau)^2)`` next to
    the formula, i.e. the authors knew the slot was an approximation.  Both
    halves pinned: R's dumped value is the binomial form (measured here),
    and ours matches it -- while the ``-dldt^2`` value differs by O(1), so a
    future "consistency cleanup" cannot pass silently.
    """
    expected = np.minimum(-1.0 / (TAU * (1.0 - TAU)), -1e-15)
    for pi in range(len(PROFILES)):
        r_vals = np.asarray(r_zasichel_slots[f"{pi}.d2ldt2"], dtype=float)
        np.testing.assert_allclose(r_vals, expected, rtol=1e-12, atol=0.0)
        ours = _ours_slots(_ZASICHEL_HESSIANS, PROFILES[pi], tau=TAU)["tau"]
        np.testing.assert_allclose(ours, r_vals, rtol=SLOT_RTOL["tau"],
                                   atol=ATOL)
        score = _ours_slots(_ZASICHEL_SCORES, PROFILES[pi], tau=TAU)["tau"]
        neg_info = np.minimum(-score * score, -1e-15)
        assert not np.allclose(ours, neg_info, rtol=1e-3), (
            pi, ours[0], neg_info[0])


# ---------------------------------------------------------------------------
# 6. Internal consistency: the closed forms differentiate our own densities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_mu_and_sigma_scores_are_the_density_gradients(pi):
    """dldm/dldd against a central difference of our own log-density.

    Independent of R: the closed forms must differentiate the density the
    family actually scores with.  h = 1e-5 lands at <= 6.2e-8 relative
    (measured across the profiles) -- the finite-difference floor, not a
    round number, hence rtol 1e-6.
    """
    profile = PROFILES[pi]
    y = jnp.asarray(Y_GRID)
    mu, sigma, nu = (float(profile[k]) for k in ("mu", "sigma", "nu"))
    logpdf = jax.vmap(_sichel_log_pdf, in_axes=(0, None, None, None))
    h = 1e-5
    dldm = np.asarray(_SICHEL_SCORES["mu"](y, mu, sigma, nu))
    dldd = np.asarray(_SICHEL_SCORES["sigma"](y, mu, sigma, nu))
    fd_mu = (np.asarray(logpdf(y, mu + h, sigma, nu))
             - np.asarray(logpdf(y, mu - h, sigma, nu))) / (2 * h)
    fd_sd = (np.asarray(logpdf(y, mu, sigma + h, nu))
             - np.asarray(logpdf(y, mu, sigma - h, nu))) / (2 * h)
    np.testing.assert_allclose(dldm, fd_mu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dldd, fd_sd, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("family", ["SICHEL", "ZISICHEL", "ZASICHEL"])
def test_nu_score_is_the_forward_difference_of_our_density(family):
    """dldv replicates R's forward difference -- and is not the central one.

    The forward/central choice is worth 5.5e-3..5.7e-3 relative on this
    grid (measured), three orders above the reproduction accuracy, so a
    future "cleanup" to the central difference cannot pass silently.  The
    zero variants' dldv differences their *own* wrapper densities, which is
    what R does too (numeric.deriv over dZISICHEL/dZASICHEL).
    """
    profile = PROFILES[0]
    y = jnp.asarray(Y_GRID)
    mu, sigma, nu = (float(profile[k]) for k in ("mu", "sigma", "nu"))
    d = _FORWARD_DELTA
    if family == "SICHEL":
        table = _SICHEL_SCORES
        logpdf = jax.vmap(_sichel_log_pdf, in_axes=(0, None, None, None))
        fwd_args = (y, mu, sigma, nu + d), (y, mu, sigma, nu)
        cen_args = (y, mu, sigma, nu + d), (y, mu, sigma, nu - d)
        slot_args = (y, mu, sigma, nu)
    else:
        table = (_ZISICHEL_SCORES if family == "ZISICHEL" else _ZASICHEL_SCORES)
        altered = family == "ZASICHEL"

        def logpdf(yv, muv, sigv, nuv, tauv):
            return jax.vmap(
                lambda a, b, c, dd, e: _zero_sichel_log_pdf(
                    a, b, c, dd, e, altered=altered),
                in_axes=(0, None, None, None, None),
            )(yv, muv, sigv, nuv, tauv)

        fwd_args = (y, mu, sigma, nu + d, TAU), (y, mu, sigma, nu, TAU)
        cen_args = (y, mu, sigma, nu + d, TAU), (y, mu, sigma, nu - d, TAU)
        slot_args = (y, mu, sigma, nu, TAU)

    dldv = np.asarray(table["nu"](*slot_args))
    fwd = (np.asarray(logpdf(*fwd_args[0]))
           - np.asarray(logpdf(*fwd_args[1]))) / d
    cen = (np.asarray(logpdf(*cen_args[0]))
           - np.asarray(logpdf(*cen_args[1]))) / (2 * d)
    # Bit-identical: the slot is literally this computation.
    np.testing.assert_array_equal(dldv, fwd)
    gap = np.max(np.abs(fwd - cen) / np.maximum(np.abs(fwd), 1e-3))
    assert gap > 1e-3, gap


# ---------------------------------------------------------------------------
# 7. R's clamps, transcribed
# ---------------------------------------------------------------------------


def test_sigma_clamp_above_10000_fires_only_for_positive_nu():
    """R's ``sigma <- ifelse(sigma>10000 & nu>0, 10000, sigma)``, both halves.

    With ``nu > 0`` the slot must evaluate exactly as if sigma were the cap
    (bit-identical, since the clamp replaces the value before any formula
    runs); with ``nu <= 0`` the same oversized sigma must give a different
    value, i.e. no clamp.  The parity grid deliberately stays inside the
    clamp-free region, so this is the only place the guard is pinned.
    """
    y = jnp.asarray(Y_GRID)
    mu = 1.0
    for nu in (0.5, -0.5):
        huge = np.asarray(_SICHEL_SCORES["sigma"](y, mu, 20000.0, nu))
        cap = np.asarray(_SICHEL_SCORES["sigma"](y, mu, 10000.0, nu))
        if nu > 0:
            np.testing.assert_array_equal(huge, cap)
        else:
            assert not np.allclose(huge, cap, rtol=1e-6)


def test_zasichel_dldd_floors_sigma_at_1e6():
    """ZASICHEL's ``dldd``/``d2ldd2`` floor sigma at 1e-6 -- and only those.

    R applies ``sigma <- ifelse(sigma<0.000001, 0.000001, sigma)`` inside
    ZASICHEL's sigma slots only; ``dldm`` has no such floor.  Both halves
    asserted on our transcription: the floored evaluation equals the value
    at the floor (bit-identical), while the unfloored ``dldm`` at the same
    tiny sigma differs.
    """
    y = jnp.asarray(Y_GRID)
    mu, nu, tau = 2.0, 0.5, TAU
    tiny = 1e-9
    floored = np.asarray(_ZASICHEL_SCORES["sigma"](y, mu, tiny, nu, tau))
    at_floor = np.asarray(_ZASICHEL_SCORES["sigma"](y, mu, 1e-6, nu, tau))
    np.testing.assert_array_equal(floored, at_floor)
    dldm_tiny = np.asarray(_ZASICHEL_SCORES["mu"](y, mu, tiny, nu, tau))
    dldm_floor = np.asarray(_ZASICHEL_SCORES["mu"](y, mu, 1e-6, nu, tau))
    assert not np.allclose(dldm_tiny, dldm_floor, rtol=1e-6)


# ---------------------------------------------------------------------------
# 8. Installation and call contract
# ---------------------------------------------------------------------------


def test_overrides_are_installed_on_all_three_families():
    """The families must actually use the R closed forms, not just carry them."""
    tables = (
        ("SICHEL", _SICHEL_SCORES, _SICHEL_HESSIANS),
        ("ZISICHEL", _ZISICHEL_SCORES, _ZISICHEL_HESSIANS),
        ("ZASICHEL", _ZASICHEL_SCORES, _ZASICHEL_HESSIANS),
    )
    for name, scores, hessians in tables:
        fam = resolve(name)
        for param, fn in scores.items():
            assert fam.score_functions[param] is fn, (name, param)
        for param, fn in hessians.items():
            assert fam.hessian_functions[param] is fn, (name, param)


def test_family_metadata_matches_live_r(r_family_probe):
    """Parameter flags, links and type read from the live family objects."""
    expected = {
        "SICHEL": ("mu TRUE,sigma TRUE,nu TRUE",
                   {"mu": "log", "sigma": "log", "nu": "identity"}),
        "ZISICHEL": ("mu TRUE,sigma TRUE,nu TRUE,tau TRUE",
                     {"mu": "log", "sigma": "log", "nu": "identity",
                      "tau": "logit"}),
        "ZASICHEL": ("mu TRUE,sigma TRUE,nu TRUE,tau TRUE",
                     {"mu": "log", "sigma": "log", "nu": "identity",
                      "tau": "logit"}),
    }
    for name, (params, links) in expected.items():
        # jsonlite serialises the flat list() as dotted keys.
        probe = {key.split(".", 1)[1]: value
                 for key, value in r_family_probe.items()
                 if key.startswith(f"{name}.")}
        assert probe["parameters"] == params, (name, probe["parameters"])
        assert probe["mu_link"] == links["mu"]
        assert probe["sigma_link"] == links["sigma"]
        assert probe["nu_link"] == links["nu"]
        if "tau" in links:
            assert probe["tau_link"] == links["tau"]
        assert probe["type"] == "Discrete"
        fam = resolve(name)
        assert fam.parameters == tuple(links)
        assert fam.links == links
        assert fam.type == "Discrete"


def test_slots_accept_kwargs_and_positional_calls_identically():
    """rs_step calls the slots with keywords; the parity tests call them
    positionally -- both spellings must be the same functions."""
    profile = PROFILES[1]
    y = jnp.asarray(Y_GRID)
    for table, tau in ((_SICHEL_SCORES, None), (_ZISICHEL_SCORES, TAU),
                       (_ZASICHEL_SCORES, TAU)):
        for fn in table.values():
            if tau is None:
                pos = np.asarray(fn(y, profile["mu"], profile["sigma"],
                                    profile["nu"]))
                kw = np.asarray(fn(y=y, mu=profile["mu"],
                                   sigma=profile["sigma"],
                                   nu=profile["nu"]))
            else:
                pos = np.asarray(fn(y, profile["mu"], profile["sigma"],
                                    profile["nu"], tau))
                kw = np.asarray(fn(y=y, mu=profile["mu"],
                                   sigma=profile["sigma"],
                                   nu=profile["nu"], tau=tau))
            np.testing.assert_array_equal(pos, kw)


def test_scalar_calls_return_zero_dimensional():
    """The scalar path (``_prepare_jax_args``) unwraps to a 0-d array."""
    y0 = jnp.asarray(0.0)
    assert np.asarray(_SICHEL_SCORES["mu"](y0, 1.0, 0.5, -0.5)).shape == ()
    assert np.asarray(_ZASICHEL_SCORES["tau"](y0, 1.0, 0.5, -0.5, 0.3)).shape == ()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
