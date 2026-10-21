"""exGAUS fit parity against live R (D2.51).

What this replaces
------------------
The family shipped with **no** initial-value entries at all, so the RS fit
started from three generic defaults, none of which is R's:

* mu: the generic ``target = y`` fallback (R: ``(y + mean(y))/2``);
* sigma: the generic residual-scale default, evaluated against that very
  ``target = y`` -- i.e. ``RMSE(y, y) = 0`` floored to eps (R:
  ``sd(y)/2``);
* nu: the generic ``1.0`` (R: ``max(2*(mean(y) - median(y)), 0.1)``).

The sigma start is the *hard* failure, not a slow start: at sigma = 1e-06
the log-link IRLS working weights collapse and the fit parks on a plateau
far from the optimum.  On the parity scanner's full-family-stream sample
the fit reported deviance **9856.41** against R's **146.905** (rel 6.6e+01)
and never converged -- the same start-value signature that hit the
S-hash family (D2.40) and, for the log-link mu start, the negative
binomial type-II family (D2.50).

R's expressions (installed gamlss.dist 6.1.1, verified by ``deparse`` of
the live family object -- the vendored source matches here)::

    mu.initial    = expression(mu <- (y + mean(y))/2)
    sigma.initial = expression(sigma <- rep(sd(y)/2, length(y)))
    nu.initial    = expression(nu <- rep(max(2 * (mean(y) - median(y)), 0.1),
                                         length(y)))

with mu.link = "identity", sigma.link = nu.link = "log" (live R).

The embedded sample below is the parity scanner's own EXGAUS fit sample
(full-family-stream rng(20261006), ``y ~ x``, ``sigma ~ 1``, n_cyc = 20)
transcribed at digits = 17, so the numbers here bind to the same stream
the scanner gates run on.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import omnilss  # noqa: F401  (package import for path setup)
from omnilss import gamlss
from omnilss._fitting_init import (
    _initial_mu_target,
    _initial_parameter_value,
    _initial_sigma,
)
from omnilss.distributions import resolve_family

FAMILY = "EXGAUS"
RSCRIPT = shutil.which("Rscript")

#: The scanner's full-family-stream EXGAUS sample (n = 48, digits = 17).
FIT_Y = np.array([
    1.3283172315689997, 1.0756909628600884, 3.544906295419793, 2.0569035458585936, 0.6552656358965366, 3.974820634678102,
    0.7804828813223862, 1.286004004409918, 0.7174565560244354, 2.9779643158821782, 0.3624957878798302, 0.9406239422184457,
    0.5358476167580444, 3.2879650605855195, 3.0301023982018838, 1.7925082050749395, 2.962078616789364, 1.4694167419276452,
    2.980809906194814, 2.307679807573539, 2.956205540440016, 3.074210791862404, 0.3593318566936865, 3.994410867123584,
    2.6486966196006922, 3.2661669311658224, 0.8615817150468918, 0.9452486114108563, 3.824224145889671, 2.9132651869092,
    3.962515139814189, 1.2068601525311327, 1.2601136140049225, 1.7383546867154342, 2.4121621698579965, 2.234377810872625,
    1.2648351523482915, 2.0444031176121333, 0.8455221221173324, 0.9608324559155861, 3.0440147023368946, 3.9933904299086507,
    1.996633843550221, 0.7138769315729979, 2.5710590251424996, 2.9324793476856734, 2.177278030442188, 0.7224774241481573,
])
FIT_X = np.array([
    0.3537128240829259, 0.2181679757541826, 0.7004036613417968, -0.12557270722557123, 0.5705996217508994, -1.016146929366217,
    -0.06338030246428493, -1.7645666461519969, 1.3932051832481633, 1.3852197968741964, 0.5652618703687209, 0.25891669096494174,
    0.30507528706791276, 1.7255051177984708, 1.054389671366512, 0.8833748343180082, 0.9011356090909587, -1.1240528826823295,
    -0.5010621812097855, 0.6625260803008833, 2.4071435462121022, 1.0072897818827287, 1.3221399134840515, 0.46037582626376033,
    -0.8970294054206734, 0.6885070832503836, 0.18733439972207178, 1.3698109136067946, -0.41559630260180536, -0.6182545609831146,
    0.8199845182287568, 0.15780968131016948, -0.5156314203022443, 0.5166253148456186, 1.9289332188072768, -0.43806005839584045,
    -0.2914946600545683, 0.4634244200481744, -0.33679527270503157, -1.106746214580469, -1.8060043560005967, 0.5206095283715058,
    0.23483537420704476, 0.5508711812426341, 0.28459160026219904, -2.0584149544367185, 0.86912205188938, 1.1171088388109625,
])

#: Live-R fit on this sample (tools/r_parity_batch.R, digits = 17), the
#: reference the RS path must reproduce.  Recorded 2026-10-22 against
#: gamlss.dist 6.1.1.
R_DEV = 146.90526093167441


def _r_vector(values) -> str:
    """Render a numeric grid as R ``c(...)`` code with plain float literals.

    numpy 2.x renders ``np.float64(0.01)`` inside f-strings, which R cannot
    parse; ``repr(float(v))`` round-trips to the same double (D2.48 lesson).
    """
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def _run_r(lines, timeout=900) -> str:
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


def _r_fit_probe_script() -> str:
    """R program: exGAUS() initial-value expressions + one gamlss() fit."""
    return f"""
suppressPackageStartupMessages({{library(gamlss); library(gamlss.dist); library(jsonlite)}})
out <- list()

fam <- exGAUS()
y <- {_r_vector(FIT_Y)}
x <- {_r_vector(FIT_X)}
d <- data.frame(y = y, x = x)

# R's own initial-value expressions, evaluated on this sample (deparse of the
# live family object confirmed the expressions themselves above).
out$initial <- list(
  mu = eval(fam$mu.initial),
  sigma = eval(fam$sigma.initial),
  nu = eval(fam$nu.initial)
)

m <- gamlss(y ~ x, family = exGAUS(), sigma.formula = ~1,
            control = gamlss.control(n.cyc = 20, trace = FALSE))
out$fit <- list(
  deviance = m$G.deviance,
  mu_coef = as.numeric(coef(m, parameter = "mu")),
  sigma_coef = as.numeric(coef(m, parameter = "sigma")),
  nu_coef = as.numeric(coef(m, parameter = "nu")),
  converged = m$converged
)
cat(toJSON(out, auto_unbox = TRUE, digits = 17, na = "null"))
"""


@pytest.fixture(scope="module")
def r_probe():
    import json

    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    stdout = _run_r(_r_fit_probe_script().splitlines())
    return json.loads(stdout)


# ---------------------------------------------------------------------------
# Initial values vs R's own expressions (pure Python + live-R dump)
# ---------------------------------------------------------------------------


class TestInitialValuesMatchR:
    """The three starts must be R's expressions, not generic fallbacks."""

    def test_mu_target_is_y_plus_mean_over_two(self):
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        target = _initial_mu_target(fam, FIT_Y, w)
        np.testing.assert_allclose(
            target, (FIT_Y + np.mean(FIT_Y)) / 2.0, rtol=1e-12,
            err_msg="mu.initial must be R's (y + mean(y))/2",
        )

    def test_sigma_initial_is_half_sd(self):
        fam = resolve_family(FAMILY)
        mu = (FIT_Y + np.mean(FIT_Y)) / 2.0
        w = np.ones_like(FIT_Y)
        sigma = _initial_sigma(fam, FIT_Y, mu, w)
        # The defect signature: the generic residual-scale default evaluated
        # RMSE(y, mu) with mu = target = y and collapsed to the eps floor.
        assert sigma > 1e-3, (
            f"sigma start {sigma} sits at the generic floor -- the D2.51 "
            "failure mode"
        )
        # R's sd() is the n-1 estimator (D2.38).
        np.testing.assert_allclose(
            sigma, np.std(FIT_Y, ddof=1) / 2.0, rtol=1e-12,
            err_msg="sigma.initial must be R's sd(y)/2",
        )

    def test_nu_initial_is_mean_median_difference_floored(self):
        fam = resolve_family(FAMILY)
        mu = (FIT_Y + np.mean(FIT_Y)) / 2.0
        w = np.ones_like(FIT_Y)
        nu = _initial_parameter_value(fam, "nu", FIT_Y, mu, w)
        expected = max(2.0 * (np.mean(FIT_Y) - np.median(FIT_Y)), 0.1)
        np.testing.assert_allclose(
            nu, expected, rtol=1e-12,
            err_msg="nu.initial must be R's max(2*(mean-median), 0.1)",
        )

    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    def test_starts_match_live_r_expressions(self, r_probe):
        """Same three expressions evaluated by R on the same sample."""
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        mu0 = _initial_mu_target(fam, FIT_Y, w)
        sig0 = _initial_sigma(fam, FIT_Y, mu0, w)
        nu0 = _initial_parameter_value(fam, "nu", FIT_Y, mu0, w)
        r_init = r_probe["initial"]
        np.testing.assert_allclose(mu0, r_init["mu"], rtol=1e-12,
                                   err_msg="mu.initial vs live R")
        np.testing.assert_allclose(sig0, r_init["sigma"][0], rtol=1e-12,
                                   err_msg="sigma.initial vs live R")
        np.testing.assert_allclose(nu0, r_init["nu"][0], rtol=1e-12,
                                   err_msg="nu.initial vs live R")


# ---------------------------------------------------------------------------
# End-to-end RS fit parity on the scanner sample
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestRsFitMatchesLiveR:
    """The scanner-sample fit must land on R's deviance again.

    Before D2.51 this fit reported 9856.410769 against R's 146.905
    (rel 6.6e+01, ``converged=False``).  With R's start values the
    trajectory locks onto live R's: measured 146.904782 vs 146.905261
    (rel 3.3e-06).  The coefficient spreads follow the flat-surface
    disposition (D2.3): near the optimum the ex-Gaussian log-likelihood
    is flat in (sigma, nu), so the two fitters stop at slightly
    different points of the same optimum.
    """

    def test_rs_fit_deviance_and_coefficients(self, r_probe):
        r = r_probe["fit"]
        # Both sides must see the identical sample.
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=20)
        assert py.converged is True
        np.testing.assert_allclose(
            float(py.deviance), float(r["deviance"]),
            rtol=1e-3, atol=1e-3,
            err_msg=f"deviance {py.deviance} vs R {r['deviance']}",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["mu"], dtype=float)),
            np.ravel(np.asarray(r["mu_coef"], dtype=float)),
            rtol=1e-2, atol=1e-3, err_msg="mu coefficients vs R",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["sigma"], dtype=float)),
            np.ravel(np.asarray(r["sigma_coef"], dtype=float)),
            rtol=5e-2, atol=5e-3, err_msg="sigma coefficient vs R",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["nu"], dtype=float)),
            np.ravel(np.asarray(r["nu_coef"], dtype=float)),
            rtol=5e-2, atol=5e-3, err_msg="nu coefficient vs R",
        )

    def test_deviance_is_not_the_pre_fix_value(self):
        """Pin the regression itself: the old failure point is gone.

        The 9856.41 plateau was a *different* point of the parameter
        space, not a noisy version of R's optimum -- asserting the fit
        does not land there guards against a silent fallback regression
        even if R is unavailable.
        """
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=20)
        assert abs(float(py.deviance) - R_DEV) / R_DEV < 1e-3, (
            f"deviance {py.deviance} is not R's optimum {R_DEV}"
        )
