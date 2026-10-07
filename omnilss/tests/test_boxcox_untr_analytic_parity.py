"""Untruncated Box-Cox parity: BCTUNTR / BCCGUNTR / BCPEUNTR against live R.

What D2.16 fixed
----------------
The previous round's recommendation was **wrong about which families these
were**, and this module records the correction so it does not have to be
re-derived:

* R's ``BCTo`` / ``BCCGo`` / ``BCPEo`` are **not** the untruncated variants.
  ``identical(deparse(body(dBCTo)), deparse(body(dBCT)))`` is ``TRUE`` in
  ``gamlss.dist`` -- their densities are byte-identical to the truncated
  families'; only ``mu.link`` swaps between ``log`` and ``identity``.  Those
  three are already implemented separately (``distributions_b14.py`` /
  ``distributions_b16.py``).
* The real untruncated families are ``BCTuntr`` / ``BCCGuntr`` / ``BCPEuntr``
  -- a third group that exists in R and that we were **missing entirely**.
  ``distributions_boxcox_compat.py`` did not implement them; it aliased the
  *truncated* ``BCT`` / ``BCCG`` / ``BCPE`` under the names ``BCTUNTR`` /
  ``BCCGUNTR`` / ``BCPEUNTR``.  The name said untruncated, the behaviour was
  truncated.

What actually differs
---------------------
Truncation enters in three places, all of which the aliases got wrong:

* the log-density carries ``-log(F(1/(sigma*|nu|)))``
* ``dldd`` / ``dldv`` / ``dldt`` carry the matching hazard
  ``h = f(1/(sigma*|nu|)) / F(1/(sigma*|nu|))`` -- ``dldm`` never does, because
  the truncation point does not involve ``mu``
* ``p`` / ``q`` rescale through the truncation constant

The links are the same as the truncated families' (``mu`` is ``identity``, not
``log`` -- the ``log`` link belongs to ``BCTo``), and the second derivatives
turn out to be identical too: none of ``d2ldm2`` / ``d2ldd2`` / ``d2ldv2`` /
``d2ldt2`` involved the truncation constant, which
``identical(deparse(body(BCT()$d2ldm2)), deparse(body(BCTuntr()$d2ldm2)))``
confirms at the R level.  ``BCPEuntr``'s ``d2ldm2`` *is* different: it drops
the ``if (any(tau < 1.05)) -dldm^2`` branch that ``BCPE`` carries.

Measured parity (60 strictly-positive points, RS on byte-identical data)
    BCTUNTR   ours 109.154038   R 109.154072   rel 3.11e-07
    BCCGUNTR  ours 109.154041   R 109.154184   rel 1.31e-06
    BCPEUNTR  ours 108.313741   R 108.313843   rel 9.46e-07
All scores and diagonal second derivatives match live R to rel <= 9.9e-15.

A note on the R probe
---------------------
R's untruncated ``G.dev.incr`` bodies start with
``z <- ifelse(nu != 0, ..., log(y/mu)/sigma)``.  With a scalar ``nu`` that
``ifelse`` collapses to length 1, so a *vectorised* call silently evaluates
every observation's kernel at ``y[1]``'s ``z``.  Only the first element comes
out right (measured: 3.714485 matches, the rest are off by up to 0.95).  Every
R value below is therefore obtained one ``y`` at a time.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


RSCRIPT = Path("C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe")

# R names its second derivatives d2ldm2/d2ldd2/d2ldv2/d2ldt2 -- the trailing
# letter is the *suffix* (mu, d=sigma, v=nu, t=tau), not the parameter name.
R_SUFFIX = {"mu": "m", "sigma": "d", "nu": "v", "tau": "t"}

PROFILES = {
    "BCTUNTR": {"mu": 2.4, "sigma": 0.7, "nu": 1.3, "tau": 3.5},
    "BCCGUNTR": {"mu": 2.4, "sigma": 0.7, "nu": 1.3},
    "BCPEUNTR": {"mu": 2.4, "sigma": 0.7, "nu": 1.3, "tau": 2.6},
}
R_NAMES = {
    "BCTUNTR": "BCTuntr",
    "BCCGUNTR": "BCCGuntr",
    "BCPEUNTR": "BCPEuntr",
}

# Strictly positive: R stops with "response variable out of range" otherwise,
# while our d* functions silently clamp -- the two would not be comparable.
FIT_Y = np.abs(2.0 + 1.2 * np.random.default_rng(2026).normal(size=40)) + 0.15

R_DEV = {
    "BCTUNTR": 109.15407229047376,
    "BCCGUNTR": 109.15418390842414,
    "BCPEUNTR": 108.31384329558718,
}


@pytest.fixture(scope="module")
def families():
    from omnilss.distributions_boxcox_untr import BCCGuntr, BCPEuntr, BCTuntr

    return {
        "BCTUNTR": BCTuntr(),
        "BCCGUNTR": BCCGuntr(),
        "BCPEUNTR": BCPEuntr(),
    }


@pytest.fixture(scope="module")
def r_probe(families):
    """One element-wise call into live R per y -- see the module docstring."""
    if not RSCRIPT.exists():
        pytest.skip(f"Rscript not found at {RSCRIPT}")

    y_path = Path(tempfile.gettempdir()) / "omnilss_untr_y.txt"
    y_path.write_text("\n".join(f"{v:.17g}" for v in FIT_Y), encoding="utf-8")

    lines = [
        "suppressMessages(library(gamlss.dist))",
        f'y <- scan("{y_path.as_posix()}", quiet=TRUE)',
        "out <- list()",
    ]
    for ours, rname in R_NAMES.items():
        p = PROFILES[ours]
        tau_arg = f", tau={p['tau']}" if "tau" in p else ""
        tau_val = f", p$tau" if "tau" in p else "NULL"
        lines += [
            f'o <- suppressWarnings(get("{rname}", envir=asNamespace("gamlss.dist")))()',
            f"p <- list(mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}{tau_arg})",
            # Element-wise: vectorised G.dev.incr collapses z to length 1.
            f"out[['{ours}.gdev']] <- vapply(y, function(yi) "
            f"suppressWarnings(as.numeric(o$G.dev.incr(yi, mu=p[['mu']], sigma=p[['sigma']], "
            f"nu=p[['nu']], tau={tau_val}))), numeric(1))",
        ]
        lines.append(
            f"""for (fn in c("dldm","d2ldm2","dldd","d2ldd2","dldv","d2ldv2","dldt","d2ldt2")) {{
  f <- o[[fn]]
  if (is.null(f)) next
  fm <- names(formals(f))
  out[[paste0("{ours}.", fn)]] <- vapply(y, function(yi) {{
    args <- list(y=yi, mu=p[['mu']], sigma=p[['sigma']], nu=p[['nu']]{', tau=p$tau' if 'tau' in p else ''})
    args <- args[intersect(fm, names(args))]
    v <- tryCatch(do.call(f, args), error=function(e) NA_real_)
    if (is.numeric(v)) as.numeric(v)[1] else NA_real_
  }}, numeric(1))
}}"""
        )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), path], capture_output=True, text=True, timeout=900
        )
    finally:
        Path(path).unlink(missing_ok=True)
        y_path.unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.strip())


# ---------------------------------------------------------------------------
# 1. The aliases were truncated.  Pin the difference.
# ---------------------------------------------------------------------------


def test_untruncated_density_differs_from_truncated_by_a_constant(families):
    """The old aliases returned ``BCT``/``BCCG``/``BCPE`` verbatim.

    Dropping the truncation term changes ``log d`` by exactly
    ``log(F(1/(sigma*|nu|)))`` -- a quantity that depends on sigma and nu but
    not on y.  If this test fails, the untruncated families have picked up a
    truncation term (or the truncated ones lost theirs); both are defects, and
    they are invisible in the deviance alone.
    """
    from omnilss import BCCG, BCPE, BCT
    from omnilss.distributions_boxcox_untr import dBCCGuntr, dBCPEuntr, dBCTuntr

    y = np.linspace(0.4, 4.0, 15)
    mu, sigma, nu = 2.4, 0.7, 1.3

    untr = np.asarray(dBCTuntr(y, mu, sigma, nu, 3.5, log=True))
    tr = -0.5 * np.asarray(BCT().g_dev_inc(y, mu, sigma, nu, 3.5))
    assert np.ptp(untr - tr) < 1e-12, "BCTUNTR differs from BCT by more than a constant"
    assert abs(np.mean(untr - tr)) > 1e-3, "difference too small to be the truncation term"

    untr = np.asarray(dBCCGuntr(y, mu, sigma, nu, log=True))
    tr = -0.5 * np.asarray(BCCG().g_dev_inc(y, mu, sigma, nu))
    assert np.ptp(untr - tr) < 1e-12
    assert abs(np.mean(untr - tr)) > 1e-3

    untr = np.asarray(dBCPEuntr(y, mu, sigma, nu, 2.6, log=True))
    tr = -0.5 * np.asarray(BCPE().g_dev_inc(y, mu, sigma, nu, 2.6))
    assert np.ptp(untr - tr) < 1e-12
    assert abs(np.mean(untr - tr)) > 1e-3


def test_truncation_term_is_recovered_against_a_normal_cdf(families):
    """The constant offset must equal ``log F(1/(sigma*|nu|))``, not anything else."""
    from scipy.stats import norm

    from omnilss import BCCG
    from omnilss.distributions_boxcox_untr import dBCCGuntr

    y = np.linspace(0.4, 4.0, 11)
    mu, sigma, nu = 2.4, 0.7, 1.3
    untr = np.asarray(dBCCGuntr(y, mu, sigma, nu, log=True))
    tr = -0.5 * np.asarray(BCCG().g_dev_inc(y, mu, sigma, nu))
    offset = float(np.mean(untr - tr))
    assert abs(offset - np.log(norm.cdf(1.0 / (sigma * abs(nu))))) < 1e-12


# ---------------------------------------------------------------------------
# 2. Links: identity on mu, same as the truncated families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours,expected", [
    ("BCTUNTR", {"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"}),
    ("BCCGUNTR", {"mu": "identity", "sigma": "log", "nu": "identity"}),
    ("BCPEUNTR", {"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"}),
])
def test_links_match_r(families, ours, expected):
    """R's untruncated families use ``mu.link = identity``.

    This is the assertion that catches the round's original (wrong) premise:
    swapping in ``log`` here would make the family match ``BCTo``, a different
    family that happens to share its density.
    """
    assert families[ours].links == expected


def test_mu_link_is_not_log_unlike_bcto(families):
    from omnilss import BCTo

    assert families["BCTUNTR"].links["mu"] == "identity"
    assert BCTo().links["mu"] == "log", (
        "BCTo is the log-link family; if this ever changes the two have merged "
        "and BCTUNTR's identity link needs re-deriving from R"
    )


# ---------------------------------------------------------------------------
# 3. One density: g_dev_inc == -2 * log d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours", list(PROFILES))
def test_density_and_deviance_agree(families, ours):
    from omnilss.distributions_boxcox_untr import (
        dBCCGuntr,
        dBCPEuntr,
        dBCTuntr,
    )

    p = PROFILES[ours]
    y = np.linspace(0.35, 4.2, 13)
    dfn = {"BCTUNTR": dBCTuntr, "BCCGUNTR": dBCCGuntr, "BCPEUNTR": dBCPEuntr}[ours]
    log_d = np.asarray(dfn(y, log=True, **p), dtype=np.float64)
    log_dev = np.asarray(families[ours].g_dev_inc(y, **p), dtype=np.float64) / -2.0
    assert np.max(np.abs(log_d - log_dev)) < 1e-12


@pytest.mark.parametrize("ours", list(PROFILES))
def test_non_positive_response_is_rejected(families, ours):
    """Support is ``y > 0`` -- R declares it via ``y.valid = all(y > 0)``.

    R's own ``G.dev.incr(0, ...)`` returns ``NaN`` (``log(0)``) and lets
    ``y.valid`` gate the response; we return a non-finite deviance, which is
    the same rejection expressed through the scoring path.  What matters for
    parity is that it is *not* a finite number: silently clamping to ``eps``
    (which the ``d`` helpers do internally) would fit a different distribution.
    """
    p = PROFILES[ours]
    gdev = np.asarray(families[ours].g_dev_inc(np.array([0.0, -1.0]), **p))
    assert np.all(~np.isfinite(gdev))


# ---------------------------------------------------------------------------
# 4. p / q round-trip and the CDF's meaning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours,dfun,qfun", [
    ("BCTUNTR", "pBCTuntr", "qBCTuntr"),
    ("BCCGUNTR", "pBCCGuntr", "qBCCGuntr"),
    ("BCPEUNTR", "pBCPEuntr", "qBCPEuntr"),
])
def test_p_q_round_trip(families, ours, dfun, qfun):
    """Round trip on probabilities that lie inside the Box-Cox support.

    With ``nu = 1.3, sigma = 0.7`` the transform ``z = ((y/mu)^nu - 1)/(nu
    * sigma)`` only reaches down to ``-1/(nu*sigma) = -1.10``, so a Student-t
    quantile below that (t(3.5) at p=0.01 is -4.06) has no preimage in ``y``.
    The lower tail of the probability axis is genuinely unreachable, which is
    why the grid starts at 0.25; :func:`test_quantile_reports_the_support_boundary`
    pins what happens out there.
    """
    import omnilss.dpqr_functions as dpqr

    p = PROFILES[ours]
    probs = np.array([0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99])
    x = np.asarray(getattr(dpqr, qfun)(probs, **p), dtype=np.float64)
    back = np.asarray(getattr(dpqr, dfun)(x, **p), dtype=np.float64)
    assert np.max(np.abs(back - probs)) < 1e-8
    assert np.all(np.diff(x) > 0), "quantiles must be increasing in p"


@pytest.mark.parametrize("ours,dfun,qfun", [
    ("BCTUNTR", "pBCTuntr", "qBCTuntr"),
    ("BCCGUNTR", "pBCCGuntr", "qBCCGuntr"),
    ("BCPEUNTR", "pBCPEuntr", "qBCPEuntr"),
])
def test_quantile_reports_the_support_boundary(families, ours, dfun, qfun):
    """Below the support, the quantile function must return 0 -- not ``eps``.

    Clamping to ``eps`` (what the truncated ``qBCT`` does) reports a tiny
    positive quantile whose CDF comes back around 0.17, i.e. the round trip
    silently reports a probability nobody asked for.
    """
    import omnilss.dpqr_functions as dpqr

    p = PROFILES[ours]
    probs = np.array([1e-6, 0.01, 0.1])
    x = np.asarray(getattr(dpqr, qfun)(probs, **p), dtype=np.float64)
    assert np.all(x == 0.0), f"expected the y=0 boundary, got {x}"
    back = np.asarray(getattr(dpqr, dfun)(x, **p), dtype=np.float64)
    assert np.all(back == 0.0), f"p(0) must be 0, got {back}"


@pytest.mark.parametrize("ours,dfun", [
    ("BCTUNTR", "pBCTuntr"),
    ("BCCGUNTR", "pBCCGuntr"),
    ("BCPEUNTR", "pBCPEuntr"),
])
def test_cdf_is_untruncated(families, ours, dfun):
    """The untruncated CDF must *exceed* the truncated one at the same point.

    Dropping the renormalisation means mass above ``1/(sigma*|nu|)`` is kept, so
    the untruncated CDF is larger everywhere by a non-negative amount.  This is
    the assertion that would have failed on the old aliases, whose ``p`` was the
    truncated CDF.
    """
    import omnilss.dpqr_functions as dpqr

    p = PROFILES[ours]
    y = np.linspace(0.5, 3.5, 9)
    trunc_fn = {"BCTUNTR": "pBCT", "BCCGUNTR": "pBCCG", "BCPEUNTR": "pBCPE"}[ours]
    untr = np.asarray(getattr(dpqr, dfun)(y, **p), dtype=np.float64)
    tr = np.asarray(getattr(dpqr, trunc_fn)(y, **p), dtype=np.float64)
    assert np.all(untr >= tr - 1e-12), "untruncated CDF must dominate the truncated one"
    assert np.max(untr - tr) > 1e-6, "difference too small to be the truncation"


# ---------------------------------------------------------------------------
# 5. Derivatives against live R, element-wise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours", list(PROFILES))
def test_scores_match_live_r(families, r_probe, ours):
    p = PROFILES[ours]
    worst = 0.0
    for param in families[ours].parameters:
        key = f"{ours}.dld{R_SUFFIX[param]}"
        if key not in r_probe:
            continue
        r_vals = np.asarray(r_probe[key], dtype=np.float64)
        ours_vals = np.asarray(
            families[ours].score_functions[param](FIT_Y, **p), dtype=np.float64
        )
        rel = np.max(np.abs(ours_vals - r_vals) / np.maximum(np.abs(r_vals), 1e-30))
        worst = max(worst, float(rel))
    assert worst < 1e-10, f"{ours}: worst score rel {worst:.3e}"


@pytest.mark.parametrize("ours", list(PROFILES))
def test_hessians_match_live_r(families, r_probe, ours):
    p = PROFILES[ours]
    worst = 0.0
    for param in families[ours].parameters:
        key = f"{ours}.d2ld{R_SUFFIX[param]}2"
        if key not in r_probe:
            continue
        r_vals = np.atleast_1d(np.asarray(r_probe[key], dtype=np.float64))
        ours_vals = np.asarray(
            families[ours].hessian_functions[param](FIT_Y, **p), dtype=np.float64
        )
        rel = np.max(np.abs(ours_vals - r_vals) / np.maximum(np.abs(r_vals), 1e-30))
        worst = max(worst, float(rel))
    assert worst < 1e-10, f"{ours}: worst Hessian rel {worst:.3e}"


@pytest.mark.parametrize("ours", list(PROFILES))
def test_gdev_inc_matches_live_r(families, r_probe, ours):
    p = PROFILES[ours]
    r_vals = np.asarray(r_probe[f"{ours}.gdev"], dtype=np.float64)
    ours_vals = np.asarray(families[ours].g_dev_inc(FIT_Y, **p), dtype=np.float64)
    assert np.max(np.abs(ours_vals - r_vals)) < 1e-12


# ---------------------------------------------------------------------------
# 6. Scores are self-consistent with the untruncated density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours", list(PROFILES))
def test_scores_match_central_difference_of_untruncated_density(families, ours):
    """Differentiate ``-g_dev_inc/2`` -- *not* ``g_dev_inc``.

    Differentiating the deviance gives exactly twice the score, which reads as
    a uniform rel error of 1.5 (i.e. |fd - s| / |fd| where fd = -2s) and looks
    like every score being wrong.  It is the classic GAMLSS-convention trap.
    """
    p = PROFILES[ours]
    y = FIT_Y[:1]
    for param in families[ours].parameters:
        h = 1e-6 * max(abs(p[param]), 1.0)

        def neg_loglik(value):
            pp = dict(p)
            pp[param] = value
            return -0.5 * float(
                np.asarray(families[ours].g_dev_inc(y, **pp), dtype=np.float64)[0]
            )

        fd = (neg_loglik(p[param] + h) - neg_loglik(p[param] - h)) / (2.0 * h)
        analytic = float(
            np.asarray(families[ours].score_functions[param](y[0], **p))
        )
        rel = abs(fd - analytic) / max(abs(fd), 1e-30)
        assert rel < 1e-6, f"{ours}/{param}: score vs central difference rel {rel:.3e}"


# ---------------------------------------------------------------------------
# 7. RS end-to-end deviance against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours", list(PROFILES))
def test_rs_deviance_matches_live_r(families, ours):
    import pandas as pd

    from omnilss import gamlss

    frame = pd.DataFrame({"y": FIT_Y, "x": np.arange(FIT_Y.size, dtype=float)})
    model = gamlss("y ~ 1", family=ours, data=frame, method="RS", trace=False)
    dev = float(model.deviance)
    expected = R_DEV[ours]
    rel = abs(dev - expected) / abs(expected)
    assert rel < 1e-5, f"{ours}: deviance {dev} vs R {expected} (rel {rel:.3e})"


# ---------------------------------------------------------------------------
# 8. No longer a compatibility alias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ours", list(PROFILES))
def test_families_are_not_compat_aliases(ours):
    from omnilss.family_capabilities import ImplementationStatus, implementation_status_for_family

    status = implementation_status_for_family(ours)
    assert status is not ImplementationStatus.COMPAT_ALIAS, (
        f"{ours} is a full implementation now; leaving it in "
        "_KNOWN_COMPAT_ALIAS_FAMILIES keeps it out of the RS route"
    )


def test_module_no_longer_named_compat():
    """``distributions_boxcox_compat.py`` held clones; the new module does not."""
    from omnilss.family_capabilities import _COMPAT_FACTORY_MODULES

    assert "omnilss.distributions_boxcox_untr" not in _COMPAT_FACTORY_MODULES
    assert "omnilss.distributions_boxcox_compat" not in _COMPAT_FACTORY_MODULES