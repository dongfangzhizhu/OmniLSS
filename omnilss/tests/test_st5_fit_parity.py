"""ST5 fit parity: from an infinite deviance to R's answer (D2.62).

What the scanner found
----------------------
ST5's rsdev gate reported "python deviance non-finite (inf)" on the
full-family stream.  The family had no entries in _fitting_init.py at all, so
all four parameters took the generic defaults: mu on target=y, sigma as the
residual scale against that same target (RMSE(y, y) = 0 -> the eps floor),
nu = 1.0, tau = 1.0.  With sigma = 2.2e-16 the standardised distance
u = sqrt(tau)(y-mu)/sqrt(2 sigma^2 + tau (y-mu)^2) saturates at +-1, the
log(1 - u) term goes to -Inf and so does the deviance.  Same signature as
SI (D2.53), exGAUS (D2.51), NBII (D2.50) and JSUO (D2.60).

Live R's ST5() starts at mu = (y+mean(y))/2, sigma = sd(y)/4, nu = 0.03,
tau = 3 (links mu=identity, sigma=log, nu=identity, tau=log -- the same as
ours), and feeds the IRLS d2ldX2 = -dldX^2 with the floor at -1e-15 for
mu/sigma but **-1e-4** for nu/tau.  Our AD path would use the true second
derivative, so the fix transcribes R's closed-form scores and the -dld^2
Hessians verbatim (the SEP1/JSUO pattern).

Fit parity after the fix (full-family-stream sample, n_cyc = 20): ours
141.933876 against R's 141.930008 (rel 2.7e-05).  Coefficients:
mu=[4.01417, -0.00832] vs R [4.01450, -0.00832], sigma=-0.791505 vs
-0.791359, nu=-0.635548 vs -0.635012, tau=-1.371414 vs -1.372557.  Both
sides stop unconverged (R: "Algorithm RS has not yet converged").
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

FAMILY = "ST5"
RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream ST5 fit sample (n = 48).
FIT_Y = [
    3.390893877405779, 3.567015420603673, 2.7155889784851825, 2.462311023516911, 3.9492403057769865, 1.8554705580460902,
    0.47847178015298025, 2.70611556834077, 0.7200277983189667, 1.4374027853083688, 2.9989701514063123, 1.442119223896924,
    0.400496766869768, 3.912047416950202, 2.9566014934019536, 2.193308061413649, 3.841667728791241, 1.1811855451676045,
    0.5388997484713152, 1.286659421507584, 3.3798962844732654, 3.576860762043481, 3.295600152863509, 3.403999151438564,
    2.064055122276556, 0.672191043255831, 0.5279966195191057, 0.8451688155816302, 3.4866094152287945, 3.568888441529188,
    0.6995331792518332, 3.394340165975508, 3.0377478428865814, 2.3984954057485837, 3.3529113203428187, 2.9585937681371166,
    2.7998939674744303, 0.5832272992591421, 1.779589237182992, 2.683070841059233, 3.253543872288013, 3.2233360039370016,
    2.3346440169545075, 3.28715466130173, 2.2725082926515836, 3.1563963672088, 3.18133216536166, 1.3826917106285754,
]
FIT_X = [
    0.7762107812604842, 0.49274164938550846, 1.426210400857574, -0.24327277281125761, -0.3034060723890493, -0.06556388292358731,
    0.599462346039347, -0.044445519888711885, -0.7331915742184381, 0.8871772684906046, 1.1242051520826701, 0.47586160906691194,
    -1.6248558331416605, -0.6469875076421917, -1.144528340220415, 0.19435325994846947, 1.2238363791227869, -0.2395096602212527,
    1.2216233875867104, 0.02772826069260237, -0.5338089666998277, -0.9502289150878125, 0.869401717101496, -0.596396052858898,
    0.16740197929756637, 1.2663875876424435, 1.5751054557742603, 1.1803377862265831, -0.22940194655667936, 0.2810504405404205,
    -0.9299036420514771, -1.8979310426948206, 0.4346119576677496, -0.6681578956495017, 0.6077884240277557, 0.16944638848857416,
    0.8259733945029054, -0.5990031478215309, -1.11246484364857, 0.9870479287575091, 0.8318557876613175, -1.2412723880927665,
    -0.5185695780341645, -0.771919071064924, -3.010152600651614, -0.8948384134274475, 1.7499822452344873, -1.0788012225009067,
]


PROBE_GRID = [
    (0.5, 2.0, 1.0, 0.5, 3.0),
    (2.5, 1.0, 0.8, -1.0, 2.0),
    (-1.0, 0.0, 1.0, 2.0, 4.0),
    (3.0, 2.0, 0.5, 0.03, 3.0),
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
            f" dldm=ST5()$dldm({args}), dldd=ST5()$dldd({args}),"
            f" dldv=ST5()$dldv({args}), dldt=ST5()$dldt({args}),"
            f" d2ldm2=ST5()$d2ldm2({args}), d2ldd2=ST5()$d2ldd2({args}),"
            f" d2ldv2=ST5()$d2ldv2({args}), d2ldt2=ST5()$d2ldt2({args}))"
        )
    probe_block = "list(" + nl + ("," + nl).join(rows) + nl + ")"
    return (
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})" + nl +
        "out <- list()" + nl +
        "y <- " + _r_vector(FIT_Y) + nl +
        "x <- " + _r_vector(FIT_X) + nl +
        "out$initials <- list(" + nl +
        "  mu = eval(ST5()$mu.initial)," + nl +
        "  sigma = eval(ST5()$sigma.initial)," + nl +
        "  nu = eval(ST5()$nu.initial)," + nl +
        "  tau = eval(ST5()$tau.initial)" + nl +
        ")" + nl +
        "out$scores <- " + probe_block + nl +
        "m <- gamlss(y ~ x, family = ST5(), sigma.formula = ~1," + nl +
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


class TestFamilyIsFittable:
    def test_deviance_is_finite(self):
        """Pre-fix the deviance was +inf on every sample."""
        py = gamlss("y ~ x", family=FAMILY,
                    data={"y": np.asarray(FIT_Y), "x": np.asarray(FIT_X)},
                    method="RS", max_iter=5, allow_compat_alias=True)
        assert np.isfinite(float(py.deviance)), "deviance must be finite"

    def test_sigma_start_is_not_on_the_floor(self):
        from omnilss._fitting_init import (
            _initial_mu_target, _initial_parameter_value, _initial_sigma,
        )

        fam = _family(FAMILY)
        y = np.asarray(FIT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        sig0 = float(np.ravel(_initial_sigma(fam, y, mu0, w))[0])
        np.testing.assert_allclose(sig0, np.std(y, ddof=1) / 4.0, rtol=1e-12,
                                   err_msg="sigma.initial must be sd(y)/4")
        assert float(np.ravel(_initial_parameter_value(fam, "nu", y, mu0, w))[0]) == 0.03
        assert float(np.ravel(_initial_parameter_value(fam, "tau", y, mu0, w))[0]) == 3.0


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestDerivativesAgainstLiveR:
    def test_initials_match_r_evaluation(self, r_probe):
        from omnilss._fitting_init import _initial_mu_target, _initial_sigma

        fam = _family(FAMILY)
        y = np.asarray(FIT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        np.testing.assert_allclose(mu0, np.asarray(r_probe["initials"]["mu"]),
                                   rtol=1e-12, err_msg="mu.initial")
        np.testing.assert_allclose(
            float(np.ravel(_initial_sigma(fam, y, mu0, w))[0]),
            r_probe["initials"]["sigma"][0], rtol=1e-12, err_msg="sigma.initial")

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
                    err_msg=f"{pname} score vs R's {slot}",
                )

    def test_hessians_match_r_slots(self, r_probe):
        """R feeds -dld^2 with the floor at -1e-4 for nu/tau (not -1e-15)."""
        fam = _family(FAMILY)
        for row in r_probe["scores"]:
            args = (row["y"], row["mu"], row["sigma"], row["nu"], row["tau"])
            for pname, slot in (("mu", "d2ldm2"), ("sigma", "d2ldd2"),
                                ("nu", "d2ldv2"), ("tau", "d2ldt2")):
                ours = float(np.ravel(np.asarray(
                    fam.hessian_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(
                    ours, float(row[slot]), rtol=1e-12, atol=1e-12,
                    err_msg=f"{pname} hessian vs R's {slot}",
                )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParity:
    def test_rs_fit_matches_r_at_scanner_budget(self, r_probe):
        """n_cyc = 20 on both sides: ours 141.9339 vs R 141.9300."""
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
                                   rtol=1e-2, atol=1e-3, err_msg="mu coefficients")
        for pname, slot in (("sigma", "sigma_coef"), ("nu", "nu_coef"),
                            ("tau", "tau_coef")):
            ours = float(np.ravel(np.asarray(py.coefficients[pname], dtype=float))[0])
            np.testing.assert_allclose(ours, float(r[slot]), rtol=1e-2, atol=1e-3,
                                       err_msg=f"{pname} coefficient")
