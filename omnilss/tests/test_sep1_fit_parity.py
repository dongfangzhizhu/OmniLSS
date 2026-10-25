"""SEP1 fit parity: the family now fits at all (D2.61).

What the scanner found
----------------------
SEP1's rsdev gate reported "python raised NotImplementedError:
Differentiation rule for 'igamma_grad_a' not implemented" on every sample --
the family could not be fitted, period.  The deriv gate passed, which is the
tell: SEP1's density reaches gammainc(1/tau, s), JAX *can* differentiate that
once (d/da goes through igamma_grad_a), so the score functions build -- but
the second derivative re-enters igamma_grad_a, which has no rule, and the RS
fit needs the Hessian.

This is the BCT/BCPE pattern documented in build_ad_family: hand in R's own
closed-form derivatives rather than teach JAX another special function.  Live
R's SEP1() carries dldm/dldd/dldv closed forms (built from lpdf, lcdf and
pgamma), a dldt that is analytic plus a two-sided difference in tau, and
d2ldX2 = -dldX^2 for all four -- so the whole family leaves the AD path.

Fit parity after the fix (full-family-stream sample, n_cyc = 20): ours
124.842834 against R's 124.842999 (rel 1.3e-06).  Coefficients:
mu=[2.11014, 0.08969] vs R [2.10954, 0.09048], sigma=0.403384 vs 0.403319,
nu=-0.235024 vs -0.236274, tau=2.743305 vs 2.743568.  Both sides report
converged=False (R: "Algorithm RS has not yet converged").

One derivative note: R's dldt writes the CDF term as log(1 + pgamma(s, 1/tau))
-- without the sign(w) the real lcdf carries.  It agrees with a central
difference of R's own dSEP1 to ~1e-10 wherever the negative tail is
negligible and is off by 57% deep in it; we transcribe R's formula verbatim
(parity means using what R feeds the IRLS, D2.15), and the deriv gate
carries a registered tolerance for that (tools/rs_family_parity_scan.py).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss
from omnilss.distributions import resolve_family

FAMILY = "SEP1"
RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream SEP1 fit sample (n = 48).
FIT_Y = [
    1.7805957573832893, 1.5993473427248357, 2.2996146699001683, 1.2897415257067606, 2.630125484433294, 3.74848201587994,
    0.37688865202626415, 3.304770471649778, 3.9333245636316136, 1.0206553855084182, 1.228023414680066, 2.5513875813348084,
    1.6406041155371205, 1.5472238247639487, 0.4940580682560771, 2.9106639984560183, 0.6892504716972429, 0.3434336147872511,
    1.77868074970141, 3.3245069768353233, 1.072699063891225, 2.242300160932334, 2.1192358061936236, 1.413079401493451,
    0.6787210931662979, 1.958054355220548, 3.507547535230229, 3.003346741160701, 0.8808168687638473, 3.3354581352511223,
    1.744608611409948, 3.07476173550104, 2.397978409298252, 2.6868399281537716, 1.464943161477456, 1.774633767663208,
    0.7918074331726034, 0.9349167193279688, 2.854959200158156, 0.38411125983464756, 1.2103725644057732, 1.2493682071767205,
    3.076217094452548, 3.6657828791252656, 2.6182094345590916, 0.8890710556506458, 3.439585851373146, 1.4040668371622265,
]
FIT_X = [
    -0.037758453696819384, 1.0633387162271675, 0.014799748557025082, 0.845339849783006, -0.14736632972187597, -0.19915786429899582,
    -1.1016096417238803, -1.1125116913417976, 0.5166347651479046, 0.1596449992578282, -0.7653956571556015, 0.8489495973875361,
    1.7784493089665565, 0.6941318164341637, -0.2320396885799987, 1.2598431934727112, 0.39342275257213166, -0.9247221660972128,
    -1.0329993302028826, -2.41220014739695, 0.7770797299213104, 1.1588068884559692, 1.3084689503778018, 0.026695362502374073,
    -0.9603109399415474, 0.24514038300497593, -0.3917444809517082, 0.7745881296901997, -0.37105671288379777, 0.6351991300636125,
    0.6214746242465163, 0.03939605301940656, 0.6110903145704288, -1.3320609929166352, -0.8225617267362837, -0.5153179055909297,
    0.3279968469212851, 0.11305865162141779, 0.41909583621280666, 0.9072862139520191, -0.30228197977855553, -1.797182549679918,
    -2.1222488247429734, 2.8652886320350097, -0.17899664598408263, -1.378726682950985, 0.9363759714750994, -0.6134777409556683,
]


# Probe grid for the score comparison -- spans both signs of w and both tails.
PROBE_GRID = [
    (0.5, 2.0, 1.0, 0.0, 2.0),
    (1.0, 2.0, 1.0, -0.5, 1.0),
    (2.5, 2.0, 0.5, 0.5, 2.0),
    (4.0, 2.0, 0.5, -1.0, 1.5),
    (0.1, 2.0, 1.5, -0.8, 2.5),
    (3.0, 1.5, 0.8, -0.3, 0.8),
]

_FAMILY_CACHE: dict[str, object] = {}


def _family(name: str):
    if name not in _FAMILY_CACHE:
        _FAMILY_CACHE[name] = resolve_family(name)
    return _FAMILY_CACHE[name]


def _r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def _run_r(lines, timeout=900) -> str:
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(chr(10).join(lines))
        path = fh.name
    try:
        proc = subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout


def _r_probe_script() -> str:
    nl = chr(10)
    rows = []
    for y, mu, sigma, nu, tau in PROBE_GRID:
        args = f"y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r}"
        rows.append(
            f"  list(y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r},"
            f" dldm=SEP1()$dldm({args}), dldd=SEP1()$dldd({args}),"
            f" dldv=SEP1()$dldv({args}), dldt=SEP1()$dldt({args}))"
        )
    probe_block = "list(" + nl + ("," + nl).join(rows) + nl + ")"
    return (
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})" + nl +
        "out <- list()" + nl +
        "y <- " + _r_vector(FIT_Y) + nl +
        "x <- " + _r_vector(FIT_X) + nl +
        "out$scores <- " + probe_block + nl +
        "m <- gamlss(y ~ x, family = SEP1(), sigma.formula = ~1," + nl +
        "            control = gamlss.control(n.cyc = 20, trace = FALSE))" + nl +
        "out$fit <- list(" + nl +
        "  deviance = m$G.deviance," + nl +
        '  mu_coef = as.numeric(coef(m, parameter = "mu")),' + nl +
        '  sigma_coef = as.numeric(coef(m, parameter = "sigma")),' + nl +
        '  nu_coef = as.numeric(coef(m, parameter = "nu")),' + nl +
        '  tau_coef = as.numeric(coef(m, parameter = "tau")),' + nl +
        "  converged = m$converged" + nl +
        ")" + nl +
        "cat(toJSON(out, auto_unbox = TRUE, digits = 17))"
    )


@pytest.fixture(scope="module")
def r_probe():
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    return json.loads(_run_r(_r_probe_script().splitlines()))


# ---------------------------------------------------------------------------
# The pre-fix failure: the family could not be fitted
# ---------------------------------------------------------------------------


class TestFamilyIsFittable:
    def test_fit_does_not_raise_not_implemented(self):
        """Pre-fix this raised NotImplementedError on igamma_grad_a."""
        py = gamlss("y ~ x", family=FAMILY,
                    data={"y": np.asarray(FIT_Y), "x": np.asarray(FIT_X)},
                    method="RS", max_iter=5, allow_compat_alias=True)
        assert np.isfinite(float(py.deviance))

    def test_hessians_are_r_squared_score_form(self):
        fam = _family(FAMILY)
        for y, mu, sigma, nu, tau in PROBE_GRID:
            args = (y, mu, sigma, nu, tau)
            for pname in fam.parameters:
                d = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                h = float(np.ravel(np.asarray(
                    fam.hessian_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(h, -d * d, rtol=1e-12,
                                           err_msg=f"{pname} hessian form")


# ---------------------------------------------------------------------------
# Scores vs live R
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestScoresAgainstLiveR:
    def test_scores_match_r_slots(self, r_probe):
        fam = _family(FAMILY)
        for row in r_probe["scores"]:
            args = (row["y"], row["mu"], row["sigma"], row["nu"], row["tau"])
            for pname, slot in (("mu", "dldm"), ("sigma", "dldd"),
                                ("nu", "dldv"), ("tau", "dldt")):
                ours = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(
                    ours, float(row[slot]), rtol=1e-9, atol=1e-11,
                    err_msg=(f"{pname} score at y={row['y']}, mu={row['mu']}, "
                             f"sigma={row['sigma']}, nu={row['nu']}, "
                             f"tau={row['tau']} vs R's {slot}"),
                )

    def test_mu_sigma_nu_scores_are_true_derivatives(self):
        """dldm/dldd/dldv are exact on both sides; only tau carries R's
        approximation (registered in the scanner's _DERIV_TOLERANCE)."""
        fam = _family(FAMILY)
        h = 1e-6
        for y, mu, sigma, nu, tau in PROBE_GRID:
            for k, pname in enumerate(("mu", "sigma", "nu"), start=1):
                base = [y, mu, sigma, nu, tau]
                plus = list(base)
                minus = list(base)
                plus[k] += h
                minus[k] -= h
                lp = float(np.ravel(np.asarray(fam.d(*plus, log=True), dtype=float))[0])
                lm = float(np.ravel(np.asarray(fam.d(*minus, log=True), dtype=float))[0])
                fd = (lp - lm) / (2.0 * h)
                analytic = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*base), dtype=float))[0])
                if not np.isfinite(analytic):
                    # R's dldv is exp(lpdf-lcdf) * (w/nu), i.e. 0/0 = NaN when
                    # nu is exactly 0 and w = nu*z = 0.  Ours reproduces R's
                    # formula bit for bit, so there is nothing to difference
                    # here; the fit never reaches nu = 0 (it starts at 0.1 on
                    # an identity link).
                    assert pname == "nu" and nu == 0.0
                    continue
                np.testing.assert_allclose(
                    analytic, fd, rtol=1e-6, atol=1e-8,
                    err_msg=f"{pname} score vs central difference of the density",
                )


# ---------------------------------------------------------------------------
# End-to-end fit parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParity:
    def test_rs_fit_matches_r_at_scanner_budget(self, r_probe):
        """n_cyc = 20 on both sides: ours 124.8428 vs R 124.8430."""
        r = r_probe["fit"]
        py = gamlss("y ~ x", family=FAMILY,
                    data={"y": np.asarray(FIT_Y), "x": np.asarray(FIT_X)},
                    method="RS", max_iter=20, allow_compat_alias=True)
        np.testing.assert_allclose(
            float(py.deviance), float(r["deviance"]), rtol=1e-3, atol=1e-3,
            err_msg=f"deviance {py.deviance} vs R {r['deviance']}",
        )

    def test_coefficients_track_r(self, r_probe):
        r = r_probe["fit"]
        py = gamlss("y ~ x", family=FAMILY,
                    data={"y": np.asarray(FIT_Y), "x": np.asarray(FIT_X)},
                    method="RS", max_iter=20, allow_compat_alias=True)
        mu_coef = np.ravel(np.asarray(py.coefficients["mu"], dtype=float))
        np.testing.assert_allclose(mu_coef, np.asarray(r["mu_coef"], dtype=float),
                                   rtol=5e-3, atol=5e-3, err_msg="mu coefficients")
        for pname, slot in (("sigma", "sigma_coef"), ("nu", "nu_coef"),
                            ("tau", "tau_coef")):
            ours = float(np.ravel(np.asarray(py.coefficients[pname], dtype=float))[0])
            np.testing.assert_allclose(ours, float(r[slot]), rtol=1e-2, atol=1e-2,
                                       err_msg=f"{pname} coefficient")
