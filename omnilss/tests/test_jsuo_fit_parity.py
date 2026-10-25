"""JSUO fit parity: initial values plus R's IRLS hessian form (D2.60).

What the scanner found
----------------------
rsdev for JSUO failed at deviance 1613.89 against live R's 146.78
(rel 1.0e+01, ours *above* R -- the defect direction), with the fitted mu
intercept at 1.9e7 and sigma at e^15: the fit had diverged completely.

Two stacked defects produced it:

1. **Missing initial values** (the D2.51 silent-fallback signature): the
   registry key is JSUO but the family object's .name is "JSUo", and
   _fitting_init.py had no entry under either spelling, so all four
   parameters took the generic defaults -- mu target=y, sigma evaluated
   against that very target (RMSE(y, y) = 0 -> the eps floor), nu=1.0,
   tau=1.0.

2. **The wrong Hessian form for the IRLS**: even with R's start values
   (mu=(y+mean(y))/2, sigma=0.1, nu=0, tau=0.5) the 20-cycle deviance
   stayed 0.4% above R's, because build_ad_family feeds the IRLS the
   *true* second derivative of the log-density while live R's JSUo()
   feeds the observed-information approximation d2ldX2 = -dldX^2 for
   every parameter.  The scores already matched R's dldm through the
   deriv gate (they are the true derivatives on both sides); only the
   Hessian form differed, and the IRLS step is built from it.  With R's
   initials alone the first RS cycle landed at mu=[3.41, -0.51] against
   R's [1.84, +0.36] -- slope sign flipped.

The fix: transcribe R's JSUo() score and d2ld*2 slots as
score_overrides / hessian_overrides (the SHASHo2 pattern from D2.40).

Fit parity after the fix: at the scanner's 20-cycle budget ours sits at
146.8435 against R's 146.7781 (rel 4.5e-04, inside the 1e-3 gate).  Both
sides are still descending (converged=False on both), and the
coefficients track R's to ~1%.
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
from omnilss._fitting_init import (
    _initial_mu_target,
    _initial_parameter_value,
    _initial_sigma,
)
from omnilss.distributions import resolve_family

FAMILY = "JSUO"
RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream JSUO fit sample (n = 48, continuous).
FIT_Y = np.array([
    1.8313234998612062, 3.9747694725619542, 1.2958661962985636, 1.3772528398974846, 0.42875975082598616, 3.953256220423128,
    3.9942732160513312, 2.274989226782398, 2.4104853551826646, 2.7054229465199686, 0.4596139754316576, 3.1499731812474847,
    1.2659076820312867, 3.621595009129091, 0.7315459989898643, 2.9150400872603397, 3.986831064297813, 0.8859463292366472,
    1.5870043377591585, 0.708646432283816, 0.9902663891084114, 1.4008438730268873, 1.8601222760092595, 1.9681962916901257,
    1.7505587372530025, 3.7162630436451227, 2.5799887944444437, 2.2741350825865823, 2.7131190636480573, 0.619050486307088,
    1.5767278658431405, 3.3983977969384567, 3.960189419247, 0.9083651492640441, 1.059705942228612, 1.0640171667760032,
    3.148474074308859, 3.486272808336951, 2.4564732322243934, 1.821273783246238, 1.848215069423664, 1.626266912432161,
    2.1080110122210414, 2.3221625789516347, 3.359120513407232, 1.7691998537792504, 3.0127109987050242, 3.86767777176582,
])
FIT_X = np.array([
    0.9080710710513159, -1.8294247365937295, 0.24572134543288204, -0.11017597263269191, 0.39525561905498663, -1.3334521226370961,
    0.8265921752732406, -1.15642095645555, 1.2910211288664386, 1.3253042359279121, -0.19615826742477477, 0.5427873389414486,
    -0.7318996898532109, 0.45835882775335096, -1.53711487724996, 0.5700960059410185, -0.946423216710621, -0.651900927005222,
    2.508543600439078, -0.10724362743146884, -0.8859549514285099, -1.1743845013365504, 1.7882412322430385, 0.351207599086858,
    -0.13377670504163944, 0.9799071912944822, 0.0685984555316092, -0.1013663837641088, 0.9369741009831983, 0.7063943587171314,
    0.6828818787957898, 0.0938244334915827, -1.0449387493580065, -0.9575125941338887, 0.9764846802561892, 0.05181235617514029,
    -0.8968968506735425, 1.5962375932978123, 1.762155802051713, -0.2553370548478036, -1.1124742457218395, 0.6170142842252949,
    0.8309301962019436, -0.17778296847501687, -1.0500315735054413, -0.3164723798176059, -0.5530410272916296, -0.1908689926203131,
])

# Probe grid for the score comparison: (y, mu, sigma, nu, tau) tuples chosen
# to span the support, including a heavy left tail where asinh(z) is large.
PROBE_GRID = [
    (0.5, 2.0, 1.0, 0.0, 0.5),
    (1.0, 2.0, 1.0, -0.5, 1.0),
    (2.5, 2.0, 0.5, 0.5, 0.5),
    (4.0, 2.0, 0.5, 0.0, 2.0),
    (0.1, 2.0, 1.5, -0.8, 0.3),
    (5.0, 2.0, 1.0, 0.3, 1.5),
    (3.0, 1.5, 0.8, -0.2, 0.8),
]


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
    """R program: JSUo's initial expressions evaluated on the fit sample,
    its dldm/dldd/dldv/dldt slots evaluated on the probe grid, and the
    reference fit at n.cyc = 20."""
    nl = chr(10)
    probe_rows = []
    for y, mu, sigma, nu, tau in PROBE_GRID:
        args = (f"y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r}")
        fam = f"JSUo()$"
        probe_rows.append(
            f"  list(y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r},"
            f" dldm={fam}dldm({args}), dldd={fam}dldd({args}),"
            f" dldv={fam}dldv({args}), dldt={fam}dldt({args}))"
        )
    probe_block = "list(" + nl + ("," + nl).join(probe_rows) + nl + ")"
    return (
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})" + nl +
        "out <- list()" + nl +
        "y <- " + _r_vector(FIT_Y) + nl +
        "x <- " + _r_vector(FIT_X) + nl +
        "fam <- JSUo()" + nl +
        "out$initial <- list(" + nl +
        "  mu = eval(fam$mu.initial)," + nl +
        "  sigma = eval(fam$sigma.initial)," + nl +
        "  nu = eval(fam$nu.initial)," + nl +
        "  tau = eval(fam$tau.initial)" + nl +
        ")" + nl +
        "out$scores <- " + probe_block + nl +
        "m <- gamlss(y ~ x, family = JSUo(), sigma.formula = ~1," + nl +
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
    stdout = _run_r(_r_probe_script().splitlines())
    return json.loads(stdout)


# ---------------------------------------------------------------------------
# Initial values (live-R expressions)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestInitials:
    def test_initial_values_match_r_evaluation(self, r_probe):
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        mu0 = _initial_mu_target(fam, FIT_Y, w)
        sig0 = float(np.ravel(_initial_sigma(fam, FIT_Y, mu0, w))[0])
        nu0 = _initial_parameter_value(fam, "nu", FIT_Y, mu0, w)
        tau0 = _initial_parameter_value(fam, "tau", FIT_Y, mu0, w)
        r_init = r_probe["initial"]
        # mu link is identity, so the link-scale target is R's own vector.
        np.testing.assert_allclose(mu0, np.asarray(r_init["mu"]), rtol=1e-12,
                                   err_msg="mu.initial vs live R")
        np.testing.assert_allclose(sig0, r_init["sigma"][0], rtol=1e-12,
                                   err_msg="sigma.initial vs live R")
        np.testing.assert_allclose(nu0, r_init["nu"][0], rtol=1e-12,
                                   err_msg="nu.initial vs live R")
        np.testing.assert_allclose(tau0, r_init["tau"][0], rtol=1e-12,
                                   err_msg="tau.initial vs live R")

    def test_sigma_start_is_not_on_the_floor(self):
        """The pre-fix failure signature: sigma started at 2.2e-16."""
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        mu0 = _initial_mu_target(fam, FIT_Y, w)
        sig0 = float(np.ravel(_initial_sigma(fam, FIT_Y, mu0, w))[0])
        assert sig0 > 1e-3, f"sigma start {sig0} sits at the generic floor"

    def test_family_name_casing(self):
        """The registry key is JSUO; the branches must key on .name = JSUo."""
        assert resolve_family(FAMILY).name == "JSUo"


# ---------------------------------------------------------------------------
# Score overrides vs live R's dldm/dldd/dldv/dldt
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestScoresAgainstLiveR:
    def test_scores_match_r_slots(self, r_probe):
        fam = resolve_family(FAMILY)
        for row in r_probe["scores"]:
            args = (row["y"], row["mu"], row["sigma"], row["nu"], row["tau"])
            for pname, slot in (("mu", "dldm"), ("sigma", "dldd"),
                                ("nu", "dldv"), ("tau", "dldt")):
                ours = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(
                    ours, float(row[slot]), rtol=1e-10, atol=1e-12,
                    err_msg=(f"{pname} score at y={row['y']}, mu={row['mu']}, "
                             f"sigma={row['sigma']}, nu={row['nu']}, "
                             f"tau={row['tau']} vs R's {slot}"),
                )

    def test_hessians_are_r_squared_score_form(self):
        """R's d2ldX2 = -dldX^2 (observed information), not the AD true
        second derivative -- the D2.60 core."""
        fam = resolve_family(FAMILY)
        for y, mu, sigma, nu, tau in PROBE_GRID:
            args = (y, mu, sigma, nu, tau)
            for pname in fam.parameters:
                d = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                h = float(np.ravel(np.asarray(
                    fam.hessian_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(h, -d * d, rtol=1e-12,
                                           err_msg=f"{pname} hessian form")

    def test_scores_are_consistent_with_the_density(self):
        """Self-check independent of R: the overridden scores must still be
        the derivatives of the family's own log-density (what the deriv
        gate checks on the full stream)."""
        fam = resolve_family(FAMILY)
        h = 1e-6
        for y, mu, sigma, nu, tau in PROBE_GRID:
            for k, pname in enumerate(fam.parameters):
                base = [y, mu, sigma, nu, tau]
                plus = list(base)
                minus = list(base)
                plus[k + 1] += h
                minus[k + 1] -= h
                lp = float(np.ravel(np.asarray(
                    fam.d(*plus, log=True), dtype=float))[0])
                lm = float(np.ravel(np.asarray(
                    fam.d(*minus, log=True), dtype=float))[0])
                fd = (lp - lm) / (2.0 * h)
                analytic = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*base), dtype=float))[0])
                np.testing.assert_allclose(
                    analytic, fd, rtol=2e-5, atol=1e-7,
                    err_msg=f"{pname} score vs central difference of the density",
                )


# ---------------------------------------------------------------------------
# End-to-end fit parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParity:
    def test_rs_fit_matches_r_at_scanner_budget(self, r_probe):
        """n_cyc = 20 on both sides: ours 146.8435 vs R 146.7781.

        rel 4.5e-04, inside the 1e-3 gate.  The pre-fix fit reported
        1613.89 (ours *above* R by 10x -- the defect direction).
        """
        r = r_probe["fit"]
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=20, allow_compat_alias=True)
        np.testing.assert_allclose(
            float(py.deviance), float(r["deviance"]), rtol=1e-3, atol=1e-3,
            err_msg=f"deviance {py.deviance} vs R {r['deviance']}",
        )

    def test_fit_no_longer_diverges(self):
        """The pre-fix signature: mu intercept 1.9e7, sigma e^15.  After the
        fix the coefficients sit in the same region as R's."""
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=20, allow_compat_alias=True)
        mu_coef = np.ravel(np.asarray(py.coefficients["mu"], dtype=float))
        sigma_coef = float(np.ravel(np.asarray(py.coefficients["sigma"], dtype=float))[0])
        assert abs(mu_coef[0]) < 10.0, f"mu intercept {mu_coef[0]} diverged"
        assert abs(sigma_coef) < 5.0, f"sigma coef {sigma_coef} diverged"
