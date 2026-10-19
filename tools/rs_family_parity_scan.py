#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch family parity scan against live R.

Motivation
----------
``converged=True`` does **not** imply parity.  LOGNO reported
``termination_reason="relative_deviance_converged"`` with a deviance of
8.9e12 while R sat at 173.05 (see commit 7aaea8a).  Catching that class of
defect requires comparing against live R family by family, not trusting the
fit's own convergence flag.

Doing this per family by hand is slow, so this tool runs three gates over
every family in one pass and prints a coverage report:

    gate 1  dpqr      d/p/q values vs live R (gamlss.dist)
    gate 2  deriv     score functions vs central finite differences
    gate 3  rsdev     RS fit deviance vs live R gamlss()

Gate 1 and 3 need R; gate 2 is pure Python and always runs.

Design notes
------------
All R work happens in a **single** Rscript invocation
(:mod:`tools.r_parity_batch`).  Spawning Rscript per family costs ~2s, which
makes a 100+ family scan impractical.

Parameter counts and argument names differ per family, so the probe builds
each job from the family's own ``parameters`` tuple rather than hardcoding a
signature.  Discrete families get integer-ish grids; unit-interval families
get (0,1) grids; the rest get a positive grid.

Usage
-----
    python tools/rs_family_parity_scan.py                 # all families
    python tools/rs_family_parity_scan.py --families NO,GA,LOGNO
    python tools/rs_family_parity_scan.py --json out.json # machine-readable
    python tools/rs_family_parity_scan.py --no-r          # gate 2 only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Grid size for derivative probing.  Kept small: the finite-difference gate
# runs three density evaluations per parameter over the whole grid.
N_GRID = 6

# IRLS cycles allowed per RS fit, applied to *both* sides so the deviance
# comparison stays like-for-like.  A well-spread response sends many families
# all the way to the cycle limit; 50 cycles pushed a full scan past 30 minutes.
N_CYC = 20
# Observations per synthetic fit.  Large enough for a stable deviance, small
# enough that the 146-family scan stays runnable.
N_OBS = 48

ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "tools" / "r_parity_batch.R"
RSCRIPT_CANDIDATES = [
    r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe",
    r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe",
    r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe",
    "Rscript",
]

# --- probe calibration -------------------------------------------------------
#
# The first version of this scanner carried hand-maintained UNIT_FAMILIES /
# COUNT_FAMILIES whitelists.  They were the single largest source of *false*
# failures: LOGITNO lives on (0, 1) but was probed on 0.3..4.0, and the
# zero-inflated count families (ZAGA, ZIBI, ...) were probed on a continuous
# grid.  A scanner that cries wolf teaches you to ignore it, so the grids are
# now derived from the family object itself, which already carries everything
# needed:
#
#   ``type``   Continuous / Discrete / Mixed  -> bounds the response
#   ``links``  log / logit / identity / ...   -> bounds each parameter
#
# A family that no candidate grid can probe (density never finite and
# positive) is reported as ``skip`` rather than ``fail``: "we could not check
# this" and "this is wrong" are different statements and must not be conflated.

# Parameter sampling range implied by each link.  ``log`` means the parameter
# lives on (0, inf); ``logit`` means (0, 1); ``identity`` means the real line.
_LINK_PARAM_RANGE = {
    "log": (0.5, 3.0),
    "logit": (0.25, 0.75),
    "probit": (0.05, 0.95),
    "cloglog": (0.05, 0.95),
    "identity": (-2.0, 3.0),
    "inverse": (0.5, 3.0),
    "sqrt": (0.25, 4.0),
    "logshiftto2": (2.5, 8.0),
}
_DEFAULT_PARAM_RANGE = (0.5, 3.0)

# Parameters that must stay strictly positive no matter what link the family
# declares.  ``identity`` maps to (-2, 3) because most GAMLSS scale/location
# parameters genuinely live on the real line -- but ``mu`` and ``sigma`` are
# positive for every continuous family, and probing them at negative values
# makes the density non-positive, so the calibration scores every candidate
# grid as unusable and the family is reported ``skip`` rather than measured.
#
# This is not hypothetical: it is exactly what happened to BCT after D2.13
# corrected ``mu.link`` from ``log`` to ``identity``.
_POSITIVE_PARAMS = {"mu", "sigma"}


def _param_range(param: str, link: str) -> tuple[float, float]:
    """Sampling range for one parameter, honouring positivity constraints."""
    lo, hi = _LINK_PARAM_RANGE.get(link, _DEFAULT_PARAM_RANGE)
    if param in _POSITIVE_PARAMS and lo <= 0.0:
        # Shift the whole grid into the positive half-line, keeping its width.
        lo = 0.5
        hi = max(hi, lo + 1.0)
    return lo, hi


# Per-(family, parameter) tolerance overrides for gate 2.
#
# The gate compares our score against a central finite difference of our own
# density.  For most parameters that is the right yardstick.  For a few, R's
# own score is a *hand-rolled approximation*, and R's value is the further of
# the two from the true derivative -- so a tight bound here reports R's
# roughness as if it were our bug.
#
# BCT's tau score is the confirmed case.  ``gamlss.dist::BCT()$dldt``
# differentiates ``log(pt(...))`` with a hard-coded 0.01 forward step:
#
#     j <- (log(pt(..., tau+0.01)) - log(pt(..., tau))) / 0.01
#
# Measured against live R at mu=0.8, sigma=1.5, nu=0.7, tau=4.0:
#
#     R's dldt                   0.0094931303529283695
#     our analytic score         0.009493130352900836    |diff to R| 2.8e-14
#     our central difference     0.0094747620749124906   |diff to R| 1.8e-05
#
# i.e. our score is ~9 orders of magnitude closer to R than the finite
# difference is.  Tightening the derivative itself would move us *away* from
# R, which is the opposite of parity.
_DERIV_TOLERANCE: dict[tuple[str, str], float] = {
    ("BCT", "tau"): 1e-2,
    ("ST2", "tau"): 5e-1,
    ("DBI", "sigma"): 1e-2,
}

_DERIV_TOLERANCE_NOTE: dict[tuple[str, str], str] = {
    ("BCT", "tau"): "R's dldt uses a 0.01 forward step on log pt; ours is 2.8e-14 from R",
    ("ST2", "tau"): (
        "R's dldt holds w fixed while differencing pt(w, df=tau+1), so it omits "
        "dw/dtau -- w = nu*sqrt(lam)*z depends on tau through lam.  Measured on "
        "live R at y=0.3, mu=sigma=1, nu=0.5: R's own dldt is 0.929927 / "
        "0.298691 / 0.083972 at tau=0.5/1/2 while a central difference of "
        "log dST2 gives 0.990414 / 0.328777 / 0.096027, i.e. R is itself 6-12% "
        "off (ratio 0.939/0.909/0.875).  The gate's central difference is fully "
        "converged -- identical from h=1e-3 down to 1e-7 -- so this is R's "
        "discretisation, not the step size.  Ours matches R to 1.2e-6; fixing "
        "it would move us away from parity (D2.15)."
    ),
    ("DBI", "sigma"): (
        "R's dDBI switches to plain dbinom when |sigma - 1| < 0.001, so inside "
        "that window the log-density is *exactly constant* in sigma and any "
        "central difference of it is 0 -- yet R's own dldd does not switch and "
        "keeps returning the general-branch value.  R is inconsistent with "
        "itself there.  Measured on live R at y=4, mu=0.45: R's dldd is "
        "-0.480365218462 / -0.480176057746 / -0.479987093435 / -0.479421170164 "
        "at sigma = 0.9995 / 1 / 1.0005 / 1.002, while a central difference of "
        "R's own dDBI gives 0.000000000000 / 0.000000000000 / -0.479421170829.  "
        "Ours reproduces R's dldd to 2e-8 (R's numeric.deriv precision), so "
        "'fixing' this would mean branching our score where R does not -- the "
        "same trap as Box-Cox: matching R is the goal (D2.23)."
    ),
}

# dpqr gate tolerances.  The default is 1e-6 relative; a family/type pair
# listed here has a *measured* R-side error larger than that, and the entry
# documents the live-R evidence (same discipline as _DERIV_TOLERANCE: the
# gate compares against the number R actually returns, not the mathematically
# ideal one).
_DPQR_TOLERANCE: dict[tuple[str, str], float] = {
    ("SIMPLEX", "q"): 1e-4,
    ("IG", "q"): 1e-4,
    ("SEP1", "q"): 1e-4,
    ("SEP2", "q"): 1e-4,
    ("SHASH", "q"): 1e-4,
    ("SN1", "q"): 1e-4,
    ("ST1", "q"): 1e-4,
    ("ST2", "q"): 1e-4,
    ("ZAIG", "q"): 1e-4,
    # R's pSEP1 integrates dSEP1 with adaptive `integrate` at its default
    # settings per call; measured against a subdivisions=2000, rel.tol=1e-13
    # re-solve of R's own integrand, R's returned values carry up to 1.13e-06
    # absolute error on the probe grid (omnilss/tmp/d232_sep1_measure.R) --
    # exactly the 1.134e-06 the gate saw.  Our values track the re-solve to
    # ~1e-10, so the residual is R's own quadrature error (D2.32).
    ("SEP1", "p"): 1e-5,
    # SEP's p/q residuals (D2.33).  Live-R sources: pSEP evaluates adaptive
    # integrate(dSEP, -Inf, q) at its DEFAULT settings per point -- measured
    # against a subdivisions=2000, rel.tol=1e-13 re-solve of the same
    # integrand, R's values carry up to 1.36e-05 absolute error on the probe
    # grid (omnilss/tmp/d233_sep_measure.R), exactly the 1.364e-05 the gate
    # reported.  qSEP is a GOLDEN-SECTION search on (F-p)^2 with tol=1e-06
    # under a RELATIVE criterion (|up-lo| > tol*(|val1|+|val2|)), bracketed
    # in [mu-5*sigma, mu+5*sigma]; re-solving R's own pSEP at uniroot
    # tol=1e-14 inside the same bracket puts R's fallpoints up to 3e-07 from
    # the true root, and the gate's 4.96e-06 max rel is that tolerance plus
    # R's p-quadrature noise carried through 1/f.  Our brentq-based q lands
    # within 8e-07 of R everywhere on the grid.  CORRECTION of a D2.32 note:
    # the earlier claim "R's qSEP is off by 20 (7.28 vs a true root 27.39)"
    # was an artifact of the high-precision re-solve script -- R's pSEP is
    # itself non-monotone far in the tail there (pSEP(27.39) = 6.96e-05,
    # an integrate failure on this sharp density), so the "27.39 root" was
    # a spurious sign change of a broken CDF.  R's qSEP fallpoint 7.28 is
    # consistent with R's own pSEP(7.28) = 0.9, and ours agrees to 8e-07.
    ("SEP", "p"): 1e-4,
    ("SEP", "q"): 1e-4,
}

# Shared evidence block for the eight families whose R quantile function is a
# uniroot solve at the DEFAULT tolerance (D2.32).  Live-R sources: qIG /
# qSEP1 / qSEP2 / qSHASH / qSN1 / qST1 / qST2 each build h1 = p<FM>(q) - p and
# call uniroot(h1, interval) with no tol argument (default
# .Machine$double.eps^0.25 ~ 1.2e-4); qZAIG composes qIG, inheriting the same
# error (and answers 0 for p <= nu by an explicit ifelse).  Measured on the
# probe grids (omnilss/tmp/d232_measure.R): re-solving R's OWN objective at
# tol=1e-14 puts R's returned roots up to these absolute errors from the true
# root, and our bisection/ndtri-based values track the tol=1e-14 re-solve to
# ~1e-9 everywhere (worst: ST1's extreme-tail point at p=0.1, mu=sigma=0.5,
# nu=-2, tau=0.5 where ours sits 8.5e-5 absolute / 4.4e-6 relative from it --
# CDF tail precision amplified through 1/f, still well inside 1e-4).  The
# residual against R is therefore R's own quantile error, not a defect on our
# side -- same disposition as the DBI deriv registration (D2.23) and SIMPLEX
# q (D2.31).  Measured max rel on the probe grid per family:
_Q_UNIROOT_MEASURED = {
    "IG": "1.96e-05",
    "SEP1": "2.48e-05",
    "SEP2": "1.51e-05",
    "SHASH": "2.34e-05",
    "SN1": "1.47e-05",
    "ST1": "1.34e-05",
    "ST2": "8.67e-06",
    "ZAIG": "8.58e-06",
}

_DPQR_TOLERANCE_NOTE: dict[tuple[str, str], str] = {
    ("SIMPLEX", "q"): (
        "R's qSIMPLEX solves uniroot(F(x)-p, c(0.001, 0.999)) with the DEFAULT "
        "tolerance (.Machine$double.eps^0.25 ~ 1.2e-4).  Measured on live R "
        "(mu=0.4, sigma=0.8, p=0.1/0.3/0.5/0.7/0.9) against a tol=1e-14 "
        "re-solve of R's own objective: R's returned roots carry up to 2.1e-05 "
        "absolute error (1.18e-05 / 9.3e-08 / 2.67e-06 / 7.07e-06 / 2.04e-05).  "
        "Our bisection converges to the true root of our trapezoid CDF (which "
        "matches R's pSIMPLEX to ~8e-09 absolute), so the residual vs R is R's "
        "own quantile error, not a defect on our side (D2.31)."
    ),
}

for _fm, _meas in _Q_UNIROOT_MEASURED.items():
    _DPQR_TOLERANCE_NOTE.setdefault((_fm, "q"), (
        f"R's q{_fm} is a uniroot solve at the DEFAULT tolerance "
        "(.Machine$double.eps^0.25 ~ 1.2e-4) -- live-R source, D2.32; "
        "qZAIG composes qIG and inherits the same error.  Measured max rel "
        f"on the probe grid: {_meas} (see the shared evidence block above "
        "the registry: our values track a tol=1e-14 re-solve of R's own "
        "objective to ~1e-9, so the residual is R's own quantile error)."
    ))
_DPQR_TOLERANCE_NOTE[("SEP1", "p")] = (
    "R's pSEP1 evaluates adaptive integrate(dSEP1, -Inf, z) at its DEFAULT "
    "settings per point; measured against a subdivisions=2000, rel.tol=1e-13 "
    "re-solve of the same integrand, R's values carry up to 1.13e-06 absolute "
    "error on the probe grid (omnilss/tmp/d232_sep1_measure.R) -- exactly the "
    "1.134e-06 the gate reported.  Ours matches the re-solve to ~1e-10, so "
    "the residual is R's own quadrature error, not a defect (D2.32)."
)
_DPQR_TOLERANCE_NOTE[("SEP", "p")] = (
    "R's pSEP evaluates adaptive integrate(dSEP, -Inf, q) at its DEFAULT "
    "settings per point; measured self-error up to 1.36e-05 on the probe "
    "grid (omnilss/tmp/d233_sep_measure.R) -- exactly the gate's 1.364e-05. "
    "Ours (scipy quad, limit=300) tracks the high-precision re-solve; the "
    "residual is R's own quadrature error (D2.33)."
)
_DPQR_TOLERANCE_NOTE[("SEP", "q")] = (
    "R's qSEP is a golden-section search on (F-p)^2, tol=1e-06 under the "
    "RELATIVE criterion |up-lo| > tol*(|val1|+|val2|), bracketed in "
    "[mu-5*sigma, mu+5*sigma].  Re-solving R's own pSEP at uniroot tol=1e-14 "
    "inside the same bracket puts R's fallpoints up to 3e-07 from the true "
    "root; the gate's 4.96e-06 max rel is that tolerance plus R's "
    "p-quadrature noise carried through 1/f.  Ours (bracket-expanding "
    "brentq, xtol=1e-10) lands within 8e-07 of R on the whole grid.  "
    "CORRECTION of the D2.32 note: 'R off by 20 (7.28 vs 27.39)' was an "
    "artifact of that re-solve script -- R's pSEP is non-monotone far in "
    "that tail (pSEP(27.39)=6.96e-05, an integrate failure), so 27.39 was "
    "a spurious sign change of a broken CDF; R's fallpoint 7.28 is "
    "consistent with R's own pSEP(7.28)=0.9 and ours agrees to 8e-07 "
    "(D2.33)."
)

# Binomial denominator used when a family declares a fixed ``bd``.  R's
# gamlss() takes the binomial response as counts with `bd` as the trial total
# (a two-column cbind(success, failure) response); our d/p/q take the same.
_PROBE_BD = 10.0

# ---------------------------------------------------------------------------
# rsdev: families where R's RS halts before it reaches the optimum we find
# ---------------------------------------------------------------------------
#
# A `rel` gap is only a defect if we are the side that is wrong.  When OUR
# deviance is the *lower* one the gap can instead mean R stopped early -- and
# R's stopping rule is an absolute one (`abs(G.dev.old - G.dev) > c.crit`,
# c.crit = 0.001 by default), which a deviance around 140 satisfies after a
# couple of cycles no matter how flat the remaining directions are.
#
# That is a real possibility, not a comfortable one, so each family listed here
# carries the evidence that was gathered before letting it through:
#
#   (family, n_cyc_ours, dev_ours, dev_r, what settles the question)
#
# The cross-check is the same in every case: re-evaluate OUR coefficients with
# R's own `d*` function.  If R's density returns our deviance, the two sides
# minimise the same objective and the gap is R's early stop; if it does not,
# our density disagrees with R and the family is a genuine defect.  Comparing
# two deviance numbers alone settles nothing.
_RSDEV_R_STOPPED_EARLY: dict[str, dict] = {
    # "ST1" was registered here through D2.30 (n_cyc=1000, dev_ours=134.604639,
    # dev_r=138.379017, "R stops early": R froze at iteration 5 with tau at the
    # normal limit while our RS descended to a deviance 3.8 lower, cross-checked
    # bit-for-bit with R's own dST1).  D2.36 removed it: once the inner RS loop
    # ran with R's glim.fit semantics (cc=0.001, cyc=50) our n_cyc=1000 fit
    # lands at 138.434 against R's 138.379 -- rel 3.9e-4, a straight gate pass
    # with no registration needed.  The historical evidence is preserved in
    # the D2.36 log.
    # "BCPE" was registered here through D2.34 (n_cyc=20, dev_ours=94.153575,
    # dev_r=122.82555373916798, "R stops early").  D2.36 removed it: once the
    # inner RS loop ran with R's glim.fit semantics (cc=0.001, cyc=50) and the
    # Box-Cox families adopted R's sigma.initial = 0.1, our n_cyc=20 fit lands
    # at 122.816234 against R's 122.825553739 -- rel 8e-5, a straight gate pass
    # with no registration needed.  The historical evidence (R's own dBCPE
    # returning our coefficients' deviance bit-for-bit at the old 94.11
    # plateau) is preserved in the D2.36 log and omnilss/tmp/d236_inner.log.
    "ZIPIG": dict(
        n_cyc=20,
        dev_ours=207.7863230779,
        dev_r=211.7673906379,
        note=(
            "R's early stop is the c_crit pattern again, and its trajectory is "
            "the clearest case yet because it *rises* monotonically: 210.6150995933 "
            "at n.cyc=2, 211.5829342519 at 5, 211.7576497286 at 10, 211.7673906379 "
            "at 14 and unchanged at 60/200/600.  R's lowest value is behind its "
            "starting point; it is not descending at all when it stops.\n"
            "Holding R's own mu and nu and sweeping sigma on the link scale, R's "
            "own G.dev.incr gives 209.0949097249 at sigma_eta = -2.5 against "
            "211.7673906379 at R's reported sigma_eta = -1.26766160 -- 2.67 lower, "
            "three orders of magnitude above the 1e-3 gate tolerance.  At our "
            "point the same slice has its minimum exactly where we sit: 207.7863230084 "
            "at sigma_eta = -3.54717651 against 207.8652291058 at -3.0 and "
            "208.2121591839 at -2.5.  R's ZIPIG()$G.dev.incr evaluated at our "
            "coefficients returns 207.7863230780 against the 207.7863230779 we "
            "report -- 1e-10 apart, so the densities agree and both sides "
            "minimise the same objective.\n"
            "Reaching that point needed R's own nu.initial.  ZIPIG was absent "
            "from every branch of _initial_parameter_value and fell through to "
            "the 1.0 default, i.e. logit(1.0) = +inf into the eta clip at +15 -- "
            "seventeen logit units from R's logit(0.1) = -2.197.  From there the "
            "RS iteration parked on a sigma plateau and cycled with period four "
            "(deviance 3251.14 -> 214.6458 over 18 monotone steps, then 788.22, "
            "455.26, 816.69, 1450.21, after which iterations 23-30 replayed "
            "iterations 2-9 to five digits).  With nu.initial = 0.1 the fit "
            "converges to 207.7863230779 at n.cyc = 20 and stays there at 40, 60 "
            "and 120, with eta_gradient_norm_by_param = {mu: 0.0165, "
            "sigma: 0.00148, nu: 8.88e-07}."
        ),
    ),
    "ZINBI": dict(
        n_cyc=20,
        dev_ours=214.3401622108,
        dev_r=214.75738739,
        note=(
            "Full-family-stream adjudication (D2.30, the two-step recipe).\n"
            "Step 1: R's trajectory is the plateau pattern -- 214.75738739 at "
            "n.cyc = 14, unchanged at 20 and 60 (before that it reports "
            "'Algorithm RS has not yet converged' warnings at 2/5/10 cycles), "
            "so it halts the moment the absolute change drops under c_crit.\n"
            "Step 2: R's own ZINBI()$G.dev.incr evaluated at OUR coefficients "
            "returns 214.340162211 -- identical to the 214.3401622108 we "
            "report, so the densities agree to the last digit and both sides "
            "minimise the same objective; our point is 0.417 below R's halt, "
            "which makes R's plateau an early stop rather than an optimum.  "
            "The self-check (R's G.dev.incr at R's own coefficients vs R's "
            "reported G.deviance) differs by 0.0091: R's stored G.deviance is "
            "the value from the iteration where it stopped, one half-step "
            "behind its final coefficients.\n"
            "Stream discipline: the earlier full-family baseline numbers for "
            "this family bind to the full-family sample stream; running the "
            "scanner on a sublist shifts every later family's sample and "
            "produces different numbers on BOTH sides -- that is a stream "
            "move, not a regression (D2.18 criterion 1, and the same trap "
            "that made ZIPIG's registration look broken on a 10-family run)."
        ),
    ),
    # "JSU" was registered here through D2.34 (n_cyc=200, dev_ours=149.1870706002586,
    # dev_r=149.21527701, "R stops early on the nu plateau", two-step
    # adjudication with R's own G.dev.incr agreeing to ~1e-10 at our
    # coefficients).  D2.36 removed it: with the glim.fit-aligned inner loop
    # (cc=0.001, cyc=50) our n_cyc=200 fit lands at 149.183 against R's
    # 149.215 -- rel 2.1e-4, a straight gate pass on the full-family stream
    # (omnilss/tmp/d236_full_scan.json).
    "PE2": dict(
        n_cyc=20,
        dev_ours=122.176634,
        dev_r=125.9294,
        note=(
            "D2.38 two-step adjudication on the full-family-stream sample.\n"
            "Step 1: R's trajectory is the c.crit pattern -- 125.9303 at cycle 6 "
            "then 125.9294 at cycle 7, where the cycle-end change (9e-4) drops "
            "under c.crit = 0.001 and R halts (converged: TRUE).  A long-budget "
            "run (c.crit = 1e-9, n.cyc = 200) only descends to 125.7922 -- 0.14 "
            "in 193 extra cycles -- so R is nowhere near our 122.176634 and the "
            "gap is not a stopping-budget artifact; R's glim.fit trajectory "
            "from R's own initial values simply never enters the basin we reach "
            "once the per-observation mu.initial/sigma.initial rollout (D2.38) "
            "starts us where R starts.\n"
            "Step 2: R's own dPE2 evaluated at OUR coefficients returns "
            "122.176634 -- identical to the 122.176634 we report, so the "
            "densities agree to the last digit and both sides minimise the "
            "same objective; our point is 3.7 below R's halt, all inside "
            "PE2's legal parameter domain.\n"
            "Initial state (both sides agree bit-for-bit): deviance 82.877441 "
            "at mu = (y+mean(y))/2, sigma = (|y-mean|+sd)/2 per observation "
            "(R's sd on the n-1 denominator), nu = 1.8.\n"
            "Stream discipline: numbers bind to the full-family sample stream "
            "(D2.30); a sublist run shifts them on BOTH sides."
        ),
    ),
}


def _response_candidates(kind: str, n: int) -> list[np.ndarray]:
    """Response grids to try, most plausible first, for a family ``type``."""
    if kind == "Discrete":
        return [
            np.arange(n, dtype=np.float64),          # 0, 1, 2, ...
            np.arange(n, dtype=np.float64) + 1.0,    # 1, 2, 3, ...
            np.linspace(0.3, 4.0, n),
            np.linspace(0.05, 0.95, n),
        ]
    if kind == "Mixed":
        # Zero-inflated / zero-adjusted: the response mixes a point mass at 0
        # with a positive (continuous or count) part.
        return [
            np.arange(n, dtype=np.float64),
            np.linspace(0.3, 4.0, n),
            np.linspace(0.05, 0.95, n),
        ]
    return [
        np.linspace(0.3, 4.0, n),
        np.linspace(0.05, 0.95, n),
        np.linspace(-3.0, 3.0, n),
        np.linspace(1.5, 8.0, n),
    ]


class Calibration(NamedTuple):
    """A probe point set that keeps the family's density finite and positive."""

    y: np.ndarray
    params: list[np.ndarray]
    bd: float | None
    ok: bool


def calibrate(family: str, n: int = 6) -> Calibration:
    """Pick a response grid and parameter values suited to ``family``.

    Candidates are scored by how many grid points yield a finite, strictly
    positive density.  ``ok`` is False when no candidate reaches ``n - 1``
    points, in which case the caller should skip rather than report a failure.
    """
    _add_src()
    from omnilss.distributions import resolve_family  # noqa: PLC0415

    fam = resolve_family(family)
    params = list(fam.parameters)
    links = fam.links or {}
    parrays = [
        np.linspace(*_param_range(p, links.get(p, "")), n)
        for p in params
    ]
    fixed = tuple(getattr(fam, "fixed_parameters", ()) or ())
    bd = _PROBE_BD if "bd" in fixed else None

    def score(y: np.ndarray) -> int:
        """Count grid points that look like they are inside the support.

        The test is *relative* to the largest density on the same grid.  A
        plain ``d > 0`` test is not enough: LOGITNO's density evaluates to a
        denormal-positive number above 1 (where it should be zero), so the
        (0.3, 4.0) grid scored a perfect 6 while only its first point was
        really in support.  Requiring a floor of ``1e-6 * max(d)`` separates
        "genuinely in support" from "numerical dust outside it".
        """
        extra = {"bd": bd} if bd is not None else {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dens = np.asarray(fam.d(y, *parrays, **extra), dtype=np.float64)
        except Exception:
            return -1
        good = np.isfinite(dens) & (dens > 0.0)
        if not good.any():
            return 0
        floor = 1e-6 * np.max(dens[good])
        return int(np.sum(good & (dens >= floor)))

    candidates = _response_candidates(getattr(fam, "type", "Continuous"), n)
    if bd is not None:
        # Binomial responses are integer counts in 0..bd.  The obvious spelling
        # -- np.arange(n) * (bd / (n - 1)) -- produces 0, 1.43, 2.86, ... 10 for
        # n=8, bd=10, and R's dbinom answers every non-integer x with -Inf plus
        # a "non-integer x" warning.  Half the grid was therefore unevaluable on
        # the R side while ours happily rounded it, which made the comparison
        # meaningless in both directions.  Integer grid instead:
        step = max(int(round(bd / max(n - 1, 1))), 1)
        int_grid = np.arange(n, dtype=np.float64) * step
        candidates.insert(0, int_grid[int_grid <= bd])
        candidates.insert(1, np.arange(n, dtype=np.float64))

    scored = [(score(y), i, y) for i, y in enumerate(candidates)]
    best = max(scored, key=lambda item: (item[0], -item[1]))
    return Calibration(y=best[2], params=parrays, bd=bd, ok=best[0] >= n - 1)


def _call_args(fam, cal: Calibration, first: np.ndarray) -> list:
    """Positional arguments for a d/p/q call, in R's argument order."""
    args = [first]
    if cal.bd is not None:
        args.append(np.full(len(first), cal.bd))
    return args + list(cal.params)


def _fit_payload(family: str, jobs: list[dict], index: dict):
    """Recover the fitting data used for this family's R fit job."""
    fit_idx = (index.get(family) or {}).get("fit")
    if fit_idx is None:
        return None
    return {
        k: [float(v) for v in vals] for k, vals in jobs[fit_idx]["data"].items()
    }


def _rscript_path() -> str | None:
    for candidate in RSCRIPT_CANDIDATES:
        if candidate == "Rscript":
            import shutil

            found = shutil.which("Rscript")
            if found:
                return found
            continue
        if Path(candidate).exists():
            return candidate
    return None


def _add_src() -> None:
    src = ROOT / "omnilss" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src()

from omnilss.distributions import resolve_family  # noqa: E402  (needs sys.path)


def discover_families() -> list[str]:
    """Return every family name the Python registry can resolve.

    Scanning ``dir(omnilss.distributions)`` only finds the ~19 names
    re-exported by that module; the rest live in ``distributions_b*.py``.
    Asking ``resolve_family`` for an unknown name makes it list all
    registered families, which is the complete source.
    """
    _add_src()
    import re

    from omnilss.distributions import resolve_family  # noqa: PLC0415

    try:
        resolve_family("__NO_SUCH_FAMILY__")
    except Exception as exc:
        names = re.findall(r"'([A-Z][A-Z0-9_]*)'", str(exc))
        if names:
            return sorted(set(names))
    # Fallback: whatever the module exports.
    mod = _dist_module()
    return sorted(a for a in dir(mod) if a.isupper() and not a.startswith("_"))


def _dist_module():
    import omnilss.distributions as d  # noqa: PLC0415

    return d


def _family_grids(family: str, n: int) -> tuple[np.ndarray, list, float | None]:
    """Backwards-compatible tuple view of :func:`calibrate`."""
    cal = calibrate(family, n)
    return cal.y, cal.params, cal.bd


# ---------------------------------------------------------------------------
# Gate 2: derivatives vs finite differences (pure Python, always runs)
# ---------------------------------------------------------------------------

def gate_derivatives(family: str) -> dict:
    """Compare each parameter's score function against central differences.

    A mismatch here means the RS/IRLS working weights are built from wrong
    derivatives -- which is exactly what drove LOGNO's sigma to 1e-18.
    """
    _add_src()
    from omnilss.distributions import resolve_family  # noqa: PLC0415

    try:
        fam = resolve_family(family)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "skip", "detail": f"resolve failed: {exc}"}

    score_fns = getattr(fam, "score_functions", None)
    d_fn = getattr(fam, "d", None)
    if not score_fns or d_fn is None:
        return {"status": "skip", "detail": "no score_functions or density"}

    params = list(fam.parameters)
    try:
        cal = calibrate(family, N_GRID)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "skip", "detail": f"calibration failed: {exc}"[:90]}
    if not cal.ok:
        return {"status": "skip", "detail": "no support grid yields a finite density"}

    y = cal.y
    base = list(cal.params)
    extra = {"bd": cal.bd} if cal.bd is not None else {}
    failures: list[str] = []

    for k, pname in enumerate(params):
        fn = score_fns.get(pname)
        if fn is None:
            continue

        def loglik_at(value: np.ndarray) -> np.ndarray:
            """Log-likelihood at parameter k = ``value``.

            ``fam.d`` returns a *density*, so it must be logged before
            differencing -- differencing the density itself yields the
            derivative of the density, not the score.
            """
            args = list(base)
            args[k] = value
            dens = np.asarray(d_fn(y, *args, **extra), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(dens)

        h = 1e-5
        grid = base[k]
        try:
            fd = (loglik_at(grid + h) - loglik_at(grid - h)) / (2.0 * h)
        except Exception as exc:
            # A family whose density raises (e.g. DPO1 used jnp.lgamma, which
            # JAX does not provide) must not abort the whole scan -- report it
            # as a failure for that family and keep going.
            failures.append(f"{pname}: density raised {type(exc).__name__}: {exc}"[:110])
            continue

        try:
            analytic = np.asarray(fn(y, *base, **extra), dtype=np.float64)
        except Exception as exc:
            failures.append(f"{pname}: raised {type(exc).__name__}")
            continue

        finite = np.isfinite(fd) & np.isfinite(analytic)
        if not finite.any():
            failures.append(f"{pname}: no finite comparison points")
            continue
        scale = np.maximum(np.abs(fd[finite]), 1.0)
        rel = np.abs(analytic[finite] - fd[finite]) / scale
        tol = _DERIV_TOLERANCE.get((family, pname), 1e-3)
        if np.max(rel) > tol:
            note = _DERIV_TOLERANCE_NOTE.get((family, pname), "")
            failures.append(
                f"{pname}: max rel diff {np.max(rel):.3e} (tol {tol:.0e}"
                + (f", {note}" if note else "")
                + f"; analytic {np.round(analytic[finite][:2], 5)} vs "
                f"fd {np.round(fd[finite][:2], 5)})"
            )

    if failures:
        return {"status": "fail", "detail": "; ".join(failures)}
    return {"status": "pass", "detail": ""}


# ---------------------------------------------------------------------------
# Gates 1 and 3: live R comparison
# ---------------------------------------------------------------------------

def _build_jobs(families: list[str], n: int) -> tuple[list[dict], dict]:
    """Build R jobs for d/p/q plus an RS fit, one entry per family."""
    _add_src()
    rng = np.random.default_rng(20261006)
    jobs: list[dict] = []
    index: dict[str, dict] = {}

    for family in families:
        entry: dict[str, dict] = {}
        params = _family_parameters(family)
        if params is None:
            index[family] = {"dpqr": None}
            continue

        try:
            y, parrays, bd = _family_grids(family, n)
        except Exception:
            index[family] = {"dpqr": None}
            continue

        # Argument names follow gamlss.dist convention.  NOTE: the first
        # argument is named differently per type -- d* takes `x`, p* takes `q`,
        # q* takes `p`.  Getting this wrong yields "unused argument" from R.
        argnames_by_type = {"d": "x", "p": "q", "q": "p"}
        params = list(params)
        extra = ["bd"] if bd is not None else []

        for type_char in ("d", "p", "q"):
            # q* takes a probability as its first argument, so its grid must
            # be in (0, 1); d*/p* take quantiles from the family's support.
            first = np.linspace(0.1, 0.9, n) if type_char == "q" else y
            vals = [first.tolist()]
            if bd is not None:
                vals.append(np.full(n, bd).tolist())
            for parray in parrays:
                vals.append(np.asarray(parray, dtype=np.float64).tolist())
            jobs.append(
                {
                    "kind": "dpqr",
                    "family": family,
                    "type": type_char,
                    "argnames": [argnames_by_type[type_char], *extra, *params],
                    "args": vals,
                }
            )
            entry[type_char] = len(jobs) - 1

        # RS fit job
        fit_data = _fit_data(family, rng, y)
        fixed = tuple(
            getattr(_safe_family(family), "fixed_parameters", ()) or ()
        )
        jobs.append(
            {
                "kind": "fit",
                "family": family,
                "data": fit_data,
                # R needs cbind(success, failure) for a binomial family; see
                # _fit_data's docstring.  Omitted otherwise so the R side keeps
                # the plain single-column formula.
                **({"response_formula": "cbind(y, fail) ~ x"} if "bd" in fixed else {}),
                "formula": "y ~ x",
                "sigma_formula": "~1",
                "n_cyc": N_CYC,
            }
        )
        entry["fit"] = len(jobs) - 1
        index[family] = entry

    return jobs, index


def _safe_family(family: str):
    """``resolve_family`` or None -- probe helpers must not raise."""
    try:
        return resolve_family(family)
    except Exception:
        return None


def _family_parameters(family: str) -> tuple[str, ...] | None:
    try:
        return tuple(resolve_family(family).parameters)
    except Exception:
        return None


def _fit_data(family: str, rng: np.random.Generator, y_grid: np.ndarray) -> dict[str, list[float]]:
    """Build a fitting sample whose response lies in the family's support.

    The calibrated grid fixes the *range* of a legal response; the sample is
    then drawn from that range in a way that matches the family's declared
    type.  Tiling the grid instead (the first attempt) produced a response with
    only six distinct values, which is degenerate enough to distort the fit --
    SST reported a deviance 80x R's and TF2 21x, and there was no way to tell a
    real defect from a data artifact.

    Binomial families (D2.22)
    -------------------------
    R's ``gamlss()`` does **not** accept a single response column plus a ``bd``
    column for a binomial family -- it rejects the fit outright::

        Error in gamlss(y ~ x, family = BI(), data = dfb, ...) :
          y values must be 0 <= y <= 1

    ``BI()$y.valid`` is ``all(y >= 0)``, yet the failure comes from
    ``gamlss()``'s own binomial branch, which only recognises the two-column
    ``cbind(success, failure)`` form.  So the sample is emitted that way:
    ``y`` holds success counts and ``fail`` holds ``bd - y``.  Before this,
    BI and DBI reported ``rsdev: skip`` with exactly that R error and their
    RS parity went unmeasured.

    Our own ``gamlss()`` takes ``y`` as the success count plus an explicit
    ``bd``, so the R-side and Python-side payloads necessarily differ in shape.
    They must not differ in *content*: ``_python_fit`` below reconstructs the
    same ``bd`` so both sides minimise the identical binomial likelihood.
    """
    n = N_OBS
    x = rng.normal(0.0, 1.0, n)
    lo, hi = float(np.min(y_grid)), float(np.max(y_grid))

    try:
        fam = resolve_family(family)
        kind = getattr(fam, "type", "Continuous")
        fixed = tuple(getattr(fam, "fixed_parameters", ()) or ())
    except Exception:
        kind, fixed = "Continuous", ()

    if kind == "Discrete":
        # Integers spanning the calibrated range, inclusive of the upper end.
        y = rng.integers(int(np.floor(lo)), int(np.ceil(hi)) + 1, n).astype(float)
    else:
        y = rng.uniform(lo, hi, n)
        if kind == "Mixed":
            # Zero-inflated / zero-adjusted: keep a point mass at zero so the
            # mixture component is actually exercised.
            mask = rng.random(n) < 0.3
            y[mask] = 0.0

    if "bd" in fixed:
        # Binomial: y is a success count in 0..bd.  The calibrated grid for
        # these families is already integral (see ``calibrate``), but clamp
        # anyway -- a non-integer count makes R's dbinom answer -Inf and the
        # comparison meaningless in both directions.
        bd = _PROBE_BD
        y = np.clip(np.round(y), 0.0, bd)
        return {"y": y.tolist(), "x": x.tolist(), "fail": (bd - y).tolist()}

    return {"y": y.tolist(), "x": x.tolist()}


def _run_r(jobs: list[dict]) -> list[dict]:
    rscript = _rscript_path()
    if rscript is None or not R_SCRIPT.exists():
        raise RuntimeError("Rscript or tools/r_parity_batch.R unavailable")

    # Pass the job list through a file: a full-family scan produces a payload
    # far beyond the Windows command-line limit (WinError 206).
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(jobs, handle)
        payload_path = handle.name
    try:
        proc = subprocess.run(
            [rscript, str(R_SCRIPT), payload_path],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        Path(payload_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"R failed ({proc.returncode}): {proc.stderr[:400]}")
    return json.loads(proc.stdout)


def gate_dpqr(family: str, results: list[dict], index: dict) -> dict:
    entry = index.get(family) or {}
    if entry.get("d") is None:
        return {"status": "skip", "detail": "no parameter metadata"}
    failures: list[str] = []
    missing_in_r = True
    # Functions that R does not attach to the family object at all.  These are
    # `skip`, not `fail`: "R has no such function" and "our value disagrees"
    # are different statements, and conflating them is what makes a scanner
    # cry wolf.  Two independent reasons a function lands here:
    #
    #   * the family never had one -- DBI ships `pDBI` in the namespace but
    #     `DBI()$q` is NULL, so `q` has no R counterpart to compare against;
    #   * the family's slot is a *parameterisation marker* rather than a
    #     function -- `DBI()$p` is the list `list(mu = TRUE, sigma = TRUE)`,
    #     which R uses to mean "this function is parameterised", so calling it
    #     would fail rather than answer.
    no_r_counterpart = False
    for type_char in ("d", "p", "q"):
        r = results[entry[type_char]]
        if "error" in r:
            # "missing function dXXX" means R has no such family at all
            # (e.g. Python's DPO1 has no R counterpart -- only DPO exists).
            # That is a registry question, not a parity defect, so it skips.
            if "missing function" in r["error"]:
                no_r_counterpart = True
                continue
            failures.append(f"{type_char}: R error {r['error'][:60]}")
            continue
        missing_in_r = False
        r_vals = np.asarray(r["values"], dtype=np.float64)
        py_vals = _python_dpqr(family, type_char, entry.get("__args"))
        if py_vals is None:
            failures.append(f"{type_char}: no Python function")
            continue
        finite = np.isfinite(r_vals) & np.isfinite(py_vals)
        if not finite.any():
            failures.append(f"{type_char}: no finite overlap")
            continue
        scale = np.maximum(np.abs(r_vals[finite]), 1.0)
        rel = np.abs(py_vals[finite] - r_vals[finite]) / scale
        # _DPQR_TOLERANCE: a registered pair means R's own value carries a
        # measured error larger than the default (see the note table); the
        # residual against R is then R's error, not ours (D2.31).
        tol = _DPQR_TOLERANCE.get((family, type_char), 1e-6)
        if np.max(rel) > tol:
            failures.append(f"{type_char}: max rel {np.max(rel):.3e}")
    if missing_in_r:
        return {"status": "skip", "detail": "no d/p/q counterpart in R"}
    if failures:
        return {"status": "fail", "detail": "; ".join(failures)}
    if no_r_counterpart:
        return {"status": "pass", "detail": "R attaches no counterpart for some slots"}
    return {"status": "pass", "detail": ""}


def _python_dpqr(family: str, type_char: str, args: list | None):
    """Evaluate the Python d/p/q for the same grid used to build the R job.

    Arguments are passed **by name**, never positionally.  The R side already
    does this (``names(aa) <- argnames`` in ``r_parity_batch.R``) because R's
    ``dBI`` signature is ``(x, bd, mu, log)`` -- ``bd`` comes *before* ``mu``.
    OurPython signatures do not agree on that order (``dBI(x, mu, bd, log)``),
    so a positional call silently hands ``mu`` the value 10 and ``bd`` the
    probability.  Measured on BI at n=8 with bd=10 and mu in 0.35..0.65:

        positional  [3.2e-03, 2.0e+06, 7.5e+24, 9.2e+33, 8.5e+52,
                     1.1e+62, 1.1e+81, 1.2e+90]
        by name     [0.0563, 0.0980, 0.2213, 0.2307, 0.2307, 0.2213,
                     0.0980, 0.0563]

    The 1.2e+90 is exactly the ``d: max rel 1.246e+90`` the scanner reported for
    BI and DBI -- so that finding was a probe defect, not an implementation one.
    ``mu`` was clipped to 1 - 1e-10 and the binomial coefficient grew without
    bound.
    """
    _add_src()
    from omnilss.distributions import resolve_family  # noqa: PLC0415

    try:
        fam = resolve_family(family)
    except Exception:
        return None
    fn = {"d": fam.d, "p": fam.p, "q": fam.q}.get(type_char)
    if fn is None:
        return None
    n = 8
    _, parrays, bd = _family_grids(family, n)
    # Match the grid used for the R job: q* takes probabilities, d*/p* take
    # quantiles from the family's support.
    if type_char == "q":
        first_arg = "p"
        first = np.linspace(0.1, 0.9, n)
    else:
        first_arg = "x" if type_char == "d" else "q"
        first = _family_grids(family, n)[0]
    kwargs = {first_arg: first}
    if bd is not None:
        kwargs["bd"] = np.full(n, bd)
    params = list(getattr(fam, "parameters", ()) or ())
    for name, p in zip(params, parrays):
        kwargs[name] = np.asarray(p, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(fn(**kwargs), dtype=np.float64).ravel()
    except TypeError:
        # Family whose d/p/q does not accept `bd` as a keyword (e.g. wrappers
        # with their own signature). Retry positionally so the gate still runs
        # rather than reporting "no Python function".
        call_args = [first]
        if bd is not None:
            call_args.append(np.full(n, bd))
        call_args += [np.asarray(p, dtype=np.float64) for p in parrays]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.asarray(fn(*call_args), dtype=np.float64).ravel()
        except Exception:
            return None
    except Exception:
        return None


def gate_rs_deviance(family: str, results: list[dict], index: dict, fit_data) -> dict:
    entry = index.get(family) or {}
    if entry.get("fit") is None:
        return {"status": "skip", "detail": "no fit job"}
    r = results[entry["fit"]]
    if "error" in r:
        return {"status": "skip", "detail": f"R fit error: {r['error'][:60]}"}

    _add_src()
    from omnilss import gamlss  # noqa: PLC0415

    # A family registered as "R stops early" has to be run at the budget its
    # case was established at.  Comparing our 20-cycle plateau against R's
    # 5-cycle halt measures two different stopping points, which is the very
    # thing the registration is about -- and it would report a number that
    # matches none of the evidence.
    rec = _RSDEV_R_STOPPED_EARLY.get(family)
    n_cyc = int(rec["n_cyc"]) if rec else N_CYC

    # Binomial families need `bd` on our side: R gets cbind(success, failure)
    # and infers the denominator from the row total, while our gamlss() takes
    # the success count plus an explicit bd (D2.22).  Reconstruct it from the
    # same two columns so both sides minimise the identical likelihood --
    # comparing R's fit against ours without bd would be scoring a Bernoulli.
    data = dict(fit_data)
    if "fail" in data and "bd" not in data:
        data["bd"] = (np.asarray(data["y"], dtype=np.float64)
                      + np.asarray(data.pop("fail"), dtype=np.float64))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss(
                "y ~ x",
                family=family,
                data=data,
                sigma_formula="~1",
                method="RS",
                max_iter=n_cyc,
                # The scanner is migration/smoke-test work by definition: it
                # is the tool that produces the evidence a compatibility
                # alias needs before it can leave the fail-closed set.  Without
                # the opt-in, every implemented-but-still-listed family would
                # report ``fail: python raised FamilyCapabilityError`` -- a
                # capability-policy answer masquerading as a parity failure.
                allow_compat_alias=True,
            )
    except Exception as exc:
        return {"status": "fail", "detail": f"python raised {type(exc).__name__}: {exc}"[:90]}

    py_dev = float(model.deviance)
    r_dev = float(r["deviance"])
    if not np.isfinite(py_dev):
        return {"status": "fail", "detail": f"python deviance non-finite ({py_dev})"}
    denom = max(abs(r_dev), 1.0)
    rel = abs(py_dev - r_dev) / denom
    if rel > 1e-3:
        # Being *below* R is the one direction that can be innocent: R's
        # absolute c.crit can halt it on a plateau.  Admit that only for
        # families whose case was checked against R's own density -- and only
        # when the numbers still match what was recorded, so a later
        # regression cannot quietly ride in on the old note.
        rec = _RSDEV_R_STOPPED_EARLY.get(family)
        if rec is not None and py_dev < r_dev:
            same = (
                abs(py_dev - rec["dev_ours"]) <= 1e-3 * max(abs(rec["dev_ours"]), 1.0)
                and abs(r_dev - rec["dev_r"]) <= 1e-3 * max(abs(rec["dev_r"]), 1.0)
            )
            if same:
                return {
                    "status": "warn",
                    "detail": (
                        f"ours {py_dev:.6g} is below R {r_dev:.6g} (rel {rel:.3e}); "
                        f"R stopped early -- cross-checked with R's own density"
                    ),
                }
            return {
                "status": "fail",
                "detail": (
                    f"deviance {py_dev:.6g} vs R {r_dev:.6g} (rel {rel:.3e}); "
                    f"registered as R-stopping-early but the numbers moved "
                    f"(recorded {rec['dev_ours']:.6g} / {rec['dev_r']:.6g}) "
                    f"-- re-verify before assuming the same cause"
                ),
            }
        return {
            "status": "fail",
            "detail": f"deviance {py_dev:.6g} vs R {r_dev:.6g} (rel {rel:.3e})",
        }
    if not r.get("converged", True):
        return {"status": "warn", "detail": f"R did not converge (py {py_dev:.6g})"}
    return {"status": "pass", "detail": f"{py_dev:.6g} vs R {r_dev:.6g}"}


# ---------------------------------------------------------------------------
# Per-family evaluation, isolated behind a wall-clock timeout
# ---------------------------------------------------------------------------


def evaluate_family(family: str, results, index, jobs, r_ok: bool, n: int) -> dict:
    """Run all applicable gates for one family and return its report row."""
    row: dict = {"family": family}
    gates = (
        ("deriv", lambda: gate_derivatives(family)),
        ("dpqr", lambda: gate_dpqr(family, results, index) if r_ok else None),
        (
            "rsdev",
            lambda: gate_rs_deviance(family, results, index, _fit_payload(family, jobs, index))
            if r_ok
            else None,
        ),
    )
    for gate_name, gate_fn in gates:
        try:
            info = gate_fn()
        except Exception as exc:  # never let one family abort the scan
            info = {"status": "fail", "detail": f"scanner raised {type(exc).__name__}: {exc}"[:110]}
        if info is not None:
            row[gate_name] = info
    return row


def _timeout_row(family: str, seconds: float) -> dict:
    return {
        "family": family,
        "deriv": {
            "status": "skip",
            "detail": f"probe exceeded {seconds:g}s and was killed",
        },
    }


def _worker_loop(conn) -> None:  # pragma: no cover - runs in a child process
    """Serve family evaluations over a pipe until the parent closes it.

    A dedicated process is the only way to bound a family that does not
    terminate: ``signal.alarm`` does not exist on Windows, and a thread cannot
    interrupt a running loop.  Terminating the worker is safe because each
    family is evaluated independently and the parent simply restarts it.
    """
    setup = conn.recv()
    results, index, jobs, r_ok, n = setup
    while True:
        try:
            family = conn.recv()
        except (EOFError, OSError):
            return  # parent went away or shut the pipe: exit quietly
        try:
            conn.send(evaluate_family(family, results, index, jobs, r_ok, n))
        except Exception as exc:  # pragma: no cover - defensive
            conn.send({"family": family, "deriv": {"status": "fail", "detail": f"{exc}"[:110]}})


class _Worker:
    """A reusable child process that evaluates one family at a time.

    The worker is restarted only when a family blows the timeout, so the
    (expensive) JAX import is paid once in the common case.
    """

    def __init__(self, results, index, jobs, r_ok: bool, n: int) -> None:
        self._setup = (results, index, jobs, r_ok, n)
        self._proc = None
        self._conn = None
        self._start()

    def _start(self) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_worker_loop, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        parent_conn.send(self._setup)
        self._conn, self._proc = parent_conn, proc

    def _restart(self) -> None:
        self.close()
        self._start()

    def evaluate(self, family: str, timeout: float) -> dict:
        try:
            self._conn.send(family)
        except (BrokenPipeError, OSError):
            self._restart()
            self._conn.send(family)
        if self._conn.poll(timeout):
            try:
                return self._conn.recv()
            except EOFError:
                self._restart()
                return _timeout_row(family, timeout)
        # The family is wedged inside the child; the only remedy is to kill it.
        self._restart()
        return _timeout_row(family, timeout)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._proc is not None:
            # Closing the pipe first lets the child leave its recv loop on
            # EOF and exit on its own; terminate is only the fallback for a
            # child that is already wedged inside a family.
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
            if self._proc.is_alive():  # pragma: no cover - stubborn child
                self._proc.kill()
            self._proc = None

    def __enter__(self) -> "_Worker":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def scan_families(
    families: list[str],
    results,
    index,
    jobs,
    r_ok: bool,
    n: int,
    timeout: float,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate every family, isolating hangs behind ``timeout`` seconds."""
    try:
        worker = _Worker(results, index, jobs, r_ok, n)
    except Exception as exc:  # pragma: no cover - no multiprocessing available
        print(f"[warn] worker unavailable ({exc}); running in-process", file=sys.stderr)
        return [evaluate_family(f, results, index, jobs, r_ok, n) for f in families]

    rows: list[dict] = []
    with worker:
        for i, family in enumerate(families, 1):
            row = worker.evaluate(family, timeout)
            rows.append(row)
            if verbose:
                cells = " ".join(
                    f"{g}={(row.get(g) or {}).get('status', '-'):5s}"
                    for g in ("deriv", "dpqr", "rsdev")
                    if g in row
                )
                print(f"[{i:3d}/{len(families)}] {family:12s} {cells}", flush=True)
    return rows


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", help="comma-separated family names")
    parser.add_argument("--json", help="write machine-readable results here")
    parser.add_argument("--no-r", action="store_true", help="run gate 2 only")
    parser.add_argument("--n", type=int, default=8, help="grid size")
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="per-family wall-clock limit in seconds (0 disables isolation)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-family progress")
    opts = parser.parse_args(argv)

    families = (
        [f.strip() for f in opts.families.split(",") if f.strip()]
        if opts.families
        else discover_families()
    )

    jobs, index = _build_jobs(families, opts.n)
    results: list[dict] = []
    r_ok = False
    if not opts.no_r:
        try:
            results = _run_r(jobs)
            r_ok = True
        except Exception as exc:
            print(f"[warn] R unavailable, running gate 2 only: {exc}", file=sys.stderr)

    if opts.timeout and opts.timeout > 0:
        rows = scan_families(
            families, results, index, jobs, r_ok, opts.n, opts.timeout, not opts.quiet
        )
    else:
        rows = [evaluate_family(f, results, index, jobs, r_ok, opts.n) for f in families]

    _report(rows, r_ok)
    if opts.json:
        Path(opts.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {opts.json}")
    return 0


def _report(rows: list[dict], r_ok: bool) -> None:
    gates = ["deriv"] + (["dpqr", "rsdev"] if r_ok else [])
    print(f"\n{'family':10s} " + "  ".join(f"{g:8s}" for g in gates))
    print("-" * (10 + 10 * len(gates)))
    for row in rows:
        cells = []
        for g in gates:
            info = row.get(g, {"status": "skip", "detail": ""})
            cells.append(f"{info['status']:8s}")
        print(f"{row['family']:10s} " + "  ".join(cells))

    print("\n=== coverage ===")
    for g in gates:
        counts: dict[str, int] = {}
        for row in rows:
            s = row.get(g, {"status": "skip"})["status"]
            counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        passed = counts.get("pass", 0)
        print(
            f"{g:6s}: {passed}/{total} pass"
            + "".join(f", {k}={v}" for k, v in sorted(counts.items()) if k != "pass")
        )

    print("\n=== failures ===")
    any_fail = False
    for row in rows:
        for g in gates:
            info = row.get(g)
            if info and info["status"] in {"fail", "warn"}:
                any_fail = True
                print(f"  {row['family']:10s} {g:6s} {info['status']:5s} {info['detail'][:100]}")
    if not any_fail:
        print("  (none)")


if __name__ == "__main__":
    raise SystemExit(main())
