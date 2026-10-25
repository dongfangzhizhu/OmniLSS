"""BCPEUNTR fit parity: the initial values the BCPE branch never gave it (D2.63).

What the scanner found
----------------------
rsdev for BCPEUNTR failed at 121.457 against R's 118.767 (rel 2.3e-02, ours
*above* R -- the defect direction).  The family is a hand-built one (not
build_ad_family): it already carries closed-form scores and Hessians, so the
defect was not derivatives.  It was the initial values: _fitting_init.py's
BCPE branches key on family.name == "BCPE", but this family's .name is
"BCPEUNTR" (all caps), so every parameter fell through to the generic
default -- mu on target=y, sigma as the residual scale against that target
(RMSE(y, y) = 0 -> the eps floor), tau = 1.0 instead of R's 2.

Live R's BCPEuntr() is *not* BCPE's: mu.initial is y + 1e-04 (BCPE uses
(y+mean(y))/2), sigma.initial is 0.1, nu.initial is 1, tau.initial is 2.
Verified with deparse of the live family object.

Fit parity after the fix (full-family-stream sample, n_cyc = 20): ours
118.752221 against R's 118.766593 (rel 1.2e-04).  Coefficients:
mu=[1.94247, -0.06284] vs R [1.94258, -0.06284], sigma=-0.658455 vs
-0.658377, nu=0.901316 vs 0.901440, tau=12.8503 vs 12.2605.  R reports
converged = TRUE (with the truncation-factor warnings above 0.99), ours
reports False -- the deviance is what the gate compares and it agrees.
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

FAMILY = "BCPEUNTR"
RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream BCPEUNTR fit sample (n = 48).
FIT_Y = [
    0.4195923815731094, 3.6615390296257266, 2.42960681808813, 1.623213537266344, 3.4331167156356854, 1.8279446018915688,
    3.7615399542587813, 1.0303121228897922, 3.8331096055190748, 1.309914079536465, 2.5315359655167016, 3.668244770501249,
    3.2467735046155526, 2.595393934534633, 3.604870328347809, 2.762347930286468, 1.9548212286508344, 2.7053588989559234,
    0.6346876834121309, 2.7807850180113705, 1.960279319917135, 1.201013484717459, 3.4507038840292212, 0.4940706117226807,
    2.117908147689809, 0.9709099790239581, 0.8965928950056583, 3.1487364199818755, 0.8223889110055758, 3.230761328668784,
    0.42450678521223356, 3.2972742279707306, 1.1998301449155953, 1.6931876332103057, 1.0313279737122767, 1.4252015429852138,
    1.592265953774952, 0.400836659493885, 1.3186128208489676, 2.983774028372621, 0.7266226134259538, 1.0514688790182913,
    0.32535743028595376, 1.9538621620398042, 0.7705162889583226, 3.027357656957945, 2.4535743506736685, 1.4379854573898347,
]
FIT_X = [
    0.84873623880972, 0.6154261486014421, -0.3850136867367299, -1.3420552736880516, -1.6255192050712548, 0.09019804203019914,
    -0.11602515127405445, -0.8719087479730528, -0.707680757231597, -0.0492935587005509, -2.567003910862146, -0.12937090499697565,
    -0.23069912090462427, 0.057177367461938376, 0.22313853211053214, 1.3764816569846425, -1.9327253973212273, 1.4270765373093846,
    0.3436887984515346, 0.6029491393674058, -0.33961996625464824, 0.16692675127281303, -1.96409381875721, -1.6532624635892041,
    0.315573440367972, 0.10724257273000574, 0.9146077523504572, -0.7429459489931328, -0.4551583534318412, -0.6051554811694366,
    0.06102710056716454, -2.1162626755898413, -0.11792804448735833, 1.4522207247351742, 0.6313223143801384, 0.5772908700985941,
    0.11374468093928361, 1.606923191478266, 0.08694163470691793, -0.8423909662339345, -1.538895892749623, -1.8309701269671683,
    -1.412909445734573, -0.2575176776461377, -0.18477811508498906, -0.06908401197391516, 0.12026892934327113, 0.8011151782999152,
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
    return (
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})" + nl +
        "out <- list()" + nl +
        "y <- " + _r_vector(FIT_Y) + nl +
        "x <- " + _r_vector(FIT_X) + nl +
        "fam <- BCPEuntr()" + nl +
        "out$initials <- list(" + nl +
        "  mu = eval(fam$mu.initial)," + nl +
        "  sigma = eval(fam$sigma.initial)," + nl +
        "  nu = eval(fam$nu.initial)," + nl +
        "  tau = eval(fam$tau.initial)" + nl +
        ")" + nl +
        "m <- gamlss(y ~ x, family = fam, sigma.formula = ~1," + nl +
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


class TestInitials:
    def test_sigma_start_is_not_on_the_floor(self):
        fam = _family(FAMILY)
        y = np.asarray(FIT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        sig0 = float(np.ravel(_initial_sigma(fam, y, mu0, w))[0])
        np.testing.assert_allclose(sig0, 0.1, rtol=1e-12,
                                   err_msg="sigma.initial must be 0.1")
        assert sig0 > 1e-12, f"sigma start {sig0} sits at the generic floor"

    def test_mu_target_is_y_plus_eps(self):
        """R: mu <- y + 1e-04 -- NOT BCPE's (y+mean(y))/2."""
        fam = _family(FAMILY)
        y = np.asarray(FIT_Y, dtype=float)
        w = np.ones_like(y)
        np.testing.assert_allclose(np.asarray(_initial_mu_target(fam, y, w)),
                                   y + 1e-04, rtol=1e-12)

    def test_tau_and_nu_initials(self):
        fam = _family(FAMILY)
        y = np.asarray(FIT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        assert float(np.ravel(_initial_parameter_value(fam, "nu", y, mu0, w))[0]) == 1.0
        assert float(np.ravel(_initial_parameter_value(fam, "tau", y, mu0, w))[0]) == 2.0

    def test_family_name_is_uppercase(self):
        """The registry key is BCPEUNTR and so is .name; the BCPE branches
        must not be relied on to match."""
        assert resolve_family(FAMILY).name == "BCPEUNTR"


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestInitialsAgainstLiveR:
    def test_initials_match_r_evaluation(self, r_probe):
        fam = _family(FAMILY)
        y = np.asarray(FIT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        np.testing.assert_allclose(mu0, np.asarray(r_probe["initials"]["mu"]),
                                   rtol=1e-12, err_msg="mu.initial")
        np.testing.assert_allclose(
            float(np.ravel(_initial_sigma(fam, y, mu0, w))[0]),
            r_probe["initials"]["sigma"][0], rtol=1e-12, err_msg="sigma.initial")
        np.testing.assert_allclose(
            float(np.ravel(_initial_parameter_value(fam, "nu", y, mu0, w))[0]),
            r_probe["initials"]["nu"][0], rtol=1e-12, err_msg="nu.initial")
        np.testing.assert_allclose(
            float(np.ravel(_initial_parameter_value(fam, "tau", y, mu0, w))[0]),
            r_probe["initials"]["tau"][0], rtol=1e-12, err_msg="tau.initial")


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParity:
    def test_rs_fit_matches_r_at_scanner_budget(self, r_probe):
        """n_cyc = 20 on both sides: ours 118.7522 vs R 118.7666."""
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
                                   rtol=1e-3, atol=1e-3, err_msg="mu coefficients")
        for pname, slot in (("sigma", "sigma_coef"), ("nu", "nu_coef")):
            ours = float(np.ravel(np.asarray(py.coefficients[pname], dtype=float))[0])
            np.testing.assert_allclose(ours, float(r[slot]), rtol=1e-3, atol=1e-3,
                                       err_msg=f"{pname} coefficient")
