"""ZABNB / ZIBNB: R's reported deviance is numerically invalid (D2.64).

What the scanner found
----------------------
Both families failed rsdev with our deviance *above* R's (ZABNB 211.719 vs
202.905, ZIBNB 221.177 vs 199.444).  The natural reading -- our fit stopped at
a worse point -- turns out to be wrong, and the evidence is a sigma sweep
through R's own density at R's own fitted values:

    sigma      1e-4      1e-6      1e-8      1e-10     1e-12     1e-15     eps
    ZABNB    214.8326  214.8255  214.8254  214.8254  214.8279  213.3046  204.9114
    ZIBNB    221.3789  221.3749  221.3748  221.3748  221.3743  221.0975  219.8483

The likelihood is flat from 1e-6 to 1e-10 and then collapses as sigma
approaches the double-precision floor -- the lbeta differences in dBNB lose
their significant digits (m = 1/sigma + 1 and n = mu*nu/sigma are both ~1e15,
and the log-density is their difference).

R's optimisation drives sigma to exactly 2.220446e-16 (exp(-36.04)) on both
families, its outer loop never converges, and its per-cycle deviance jumps by
~10, so the G.deviance it reports (202.905 / 199.444) is not the likelihood at
any sigma -- it is a stale mid-cycle number from a non-converged iteration.
Our fits stop at 211.7189 / 221.1768, below R's numerically valid values.

Two consequences for our side, both implemented here:
  * the ZABNB/ZIBNB IRLS now receives R's hurdle-corrected scores and
    d2ldX2 = -dld^2 weights (ZABNB's tau slot is -1/(tau(1-tau)), not
    -dld^2), so the weights match R's wherever the trajectory is meaningful;
  * the scanner carries a _RSDEV_R_REFERENCE_INVALID entry that compares
    against R's density at sigma = 1e-10 instead of the invalid report, and
    only when our deviance is below it.

Our own density returns NaN at sigma = eps (same cancellation), which is why
the fit does not follow R onto the floor.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss
from omnilss.distributions import resolve_family

RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream fit samples (n = 48, counts).
SAMPLES = {
"ZABNB": dict(
    y=[
        7.0, 5.0, 4.0, 1.0, 6.0, 4.0,
        7.0, 3.0, 0.0, 0.0, 7.0, 2.0,
        5.0, 4.0, 0.0, 5.0, 7.0, 0.0,
        6.0, 4.0, 2.0, 2.0, 7.0, 6.0,
        0.0, 6.0, 0.0, 1.0, 0.0, 2.0,
        5.0, 7.0, 2.0, 2.0, 6.0, 1.0,
        3.0, 0.0, 6.0, 6.0, 6.0, 3.0,
        7.0, 2.0, 1.0, 4.0, 4.0, 0.0,
    ],
    x=[
        0.7667184877329531, 2.581768468560908, -1.746823371403672, 1.0255723816088926, -0.4539229996252858, 0.5847625416394496,
        1.0420377559555052, -0.08815014711390415, -0.7511474318153764, -0.6704252714083329, 0.09291794207447387, -1.17669744566929,
        -0.15662289355562936, 0.44970373751009524, -0.5200109248437361, -1.8453534030151428, 0.14091173177002142, 1.121919681619947,
        -1.25698908032195, 1.4102396207437706, -0.2693179467347592, -0.338159973805197, -0.5912971296011, 0.6171188843785173,
        -1.1079531679963415, -1.8335819962155, -1.2823665287696453, -1.6168354433567578, -1.157844735308932, -1.2046725800178386,
        0.008639541472347507, 0.7667257596349425, -1.557278605776024, -0.6438450422432429, 0.10065405267487369, 0.3745568970839295,
        2.5185969781035245, -0.009923235649138077, 0.09027687358590902, -0.2536885075754081, -1.2484979377424357, -0.5390400796005534,
        1.3270895001919445, 1.0559452031913819, 0.9074471997247658, -0.8209636590110332, -0.5888225012542043, -1.131937070129153,
    ],
),
"ZIBNB": dict(
    y=[
        7.0, 0.0, 1.0, 7.0, 1.0, 1.0,
        4.0, 7.0, 5.0, 0.0, 1.0, 6.0,
        7.0, 4.0, 5.0, 2.0, 6.0, 3.0,
        7.0, 1.0, 5.0, 4.0, 5.0, 5.0,
        0.0, 0.0, 7.0, 0.0, 6.0, 0.0,
        2.0, 1.0, 0.0, 2.0, 7.0, 1.0,
        6.0, 7.0, 0.0, 7.0, 7.0, 6.0,
        2.0, 2.0, 3.0, 7.0, 4.0, 3.0,
    ],
    x=[
        0.6844246566464786, 0.23854270393846191, 1.882368611127282, -0.13872440413920054, 1.2052997356555963, -1.6268219261683425,
        0.36183690739509544, 1.3566469014228366, 0.8464536525804497, -0.8790836007008335, -0.3214121508185425, 0.8131098634914746,
        1.146283399702772, -0.41727388124395437, 0.49429025208101623, 0.6145446547471621, -0.19031759100573273, 0.6870583881405584,
        0.6218758255711628, 1.9879247842930745, -1.1495204146729594, 1.287140166150893, 0.4572055297391318, -0.45957248007674595,
        0.11831280651541208, -0.05826523062013919, -1.0398037218288299, -0.5716169173578416, 0.0420734232758041, 1.0609967595424799,
        -0.4210222875719756, -0.384140262071295, 1.3342826383437731, 0.39363376836398434, 0.4098309280857815, 0.8065178286954869,
        1.4024152706270976, 0.9481191663006241, -1.1122243252436552, 0.6243963343796116, -0.406289694232506, -0.9031341825511184,
        0.1891526512379927, -0.06363601220304842, -1.4172967357868516, -1.1025838461464372, -0.12743105269569807, 0.8581838490916867,
    ],
),
}

#: R's reported G.deviance on these samples, and the value R's own density
#: returns at a numerically safe sigma (the 1e-6..1e-10 plateau).
R_REPORTED = {"ZABNB": 202.9046037917726, "ZIBNB": 199.4437321287374}
R_VALID_AT = {"ZABNB": 214.832625, "ZIBNB": 221.374813}
DEV_OURS = {"ZABNB": 211.718924, "ZIBNB": 221.176844}

_FAMILY_CACHE: dict[str, object] = {}


def _family(name: str):
    if name not in _FAMILY_CACHE:
        _FAMILY_CACHE[name] = resolve_family(name)
    return _FAMILY_CACHE[name]


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


def _r_sweep(name: str) -> dict:
    """Fit in R, then sweep sigma through R's own density at R's fitted values.

    Fitting inside the test (rather than pinning the reported coefficients)
    matters: at sigma near the floor the likelihood is so sensitive that
    rounding a coefficient changes the deviance by whole units.
    """
    import json

    s = SAMPLES[name]
    lines = [
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})",
        "y <- c(" + ", ".join(repr(v) for v in s["y"]) + ")",
        "x <- c(" + ", ".join(repr(v) for v in s["x"]) + ")",
        "m <- suppressWarnings(gamlss(y ~ x, family = " + name + "(), data = data.frame(y = y, x = x),",
        "      sigma.formula = ~1, control = gamlss.control(n.cyc = 20, trace = FALSE)))",
        'fmu <- as.numeric(fitted(m, what = "mu")); fs <- as.numeric(fitted(m, what = "sigma"))',
        'fnu <- as.numeric(fitted(m, what = "nu")); ft <- as.numeric(fitted(m, what = "tau"))',
        "sig <- c(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-15, 2.220446e-16)",
        "dev <- sapply(sig, function(s0) -2 * sum(d" + name + "(y, mu = fmu, sigma = s0, nu = fnu, tau = ft, log = TRUE)))",
        "cat(toJSON(list(reported = m$G.deviance, converged = m$converged,",
        "                  sigma_fitted = fs[1], sigmas = sig, devs = as.numeric(dev)),",
        "            auto_unbox = TRUE, digits = 17))",
    ]
    return json.loads(_run_r(lines))


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
@pytest.mark.parametrize("name", ["ZABNB", "ZIBNB"])
class TestRReferenceIsNumericallyInvalid:
    def test_r_sigma_hits_the_double_precision_floor(self, name):
        probe = _r_sweep(name)
        assert probe["converged"] is False, "R must report a non-converged fit"
        assert probe["sigma_fitted"] == pytest.approx(2.220446e-16, rel=1e-9), (
            "R's fit must end exactly on the double epsilon -- that is the "
            "precondition the registration rests on"
        )

    def test_r_density_is_stable_then_collapses(self, name):
        probe = _r_sweep(name)
        devs = dict(zip([float(v) for v in probe["sigmas"]],
                        [float(v) for v in probe["devs"]]))
        np.testing.assert_allclose(devs[1e-6], devs[1e-10], rtol=1e-5,
                                   err_msg="R's likelihood should be flat here")
        assert abs(devs[2.220446e-16] - devs[1e-10]) / devs[1e-10] > 5e-3, (
            "R's density must lose >= 0.5% at the eps floor"
        )
        np.testing.assert_allclose(devs[1e-10], R_VALID_AT[name], rtol=5e-4,
                                   err_msg="recorded valid-at value drifted")

    def test_reported_deviance_is_not_the_deviance_at_the_floor(self, name):
        probe = _r_sweep(name)
        devs = [float(v) for v in probe["devs"]]
        assert abs(float(probe["reported"]) - devs[-1]) / max(devs[-1], 1.0) > 1e-3
        np.testing.assert_allclose(float(probe["reported"]), R_REPORTED[name],
                                   rtol=5e-4, err_msg="recorded R report drifted")

    def test_our_fit_is_below_r_valid_value(self, name):
        s = SAMPLES[name]
        py = gamlss("y ~ x", family=name,
                    data={"y": np.asarray(s["y"]), "x": np.asarray(s["x"])},
                    method="RS", max_iter=20, allow_compat_alias=True)
        dev = float(py.deviance)
        np.testing.assert_allclose(dev, DEV_OURS[name], rtol=1e-3,
                                   err_msg="our recorded deviance drifted")
        assert dev < R_VALID_AT[name], (
            f"ours {dev} must be below R's numerically valid {R_VALID_AT[name]}"
        )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
@pytest.mark.parametrize("name", ["ZABNB", "ZIBNB"])
class TestHurdleScores:
    def test_scores_match_r_slots(self, name):
        """R's ZABNB/ZIBNB dld* are hurdle-corrected, not the plain BNB ones."""
        import json

        s = SAMPLES[name]
        y0 = float(s["y"][0])
        probe_pts = [(y0, 1.5, 0.5, 0.3, 0.25), (0.0, 1.5, 0.5, 0.3, 0.25),
                     (3.0, 2.0, 0.8, 0.5, 0.4)]
        nl = chr(10)
        rows = []
        for y, mu, sigma, nu, tau in probe_pts:
            args = f"y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r}"
            rows.append(
                f"  list(y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r},"
                f" dldm={name}()$dldm({args}), dldd={name}()$dldd({args}),"
                f" dldv={name}()$dldv({args}), dldt={name}()$dldt({args}),"
                f" d2ldt2={name}()$d2ldt2({args}))"
            )
        lines = [
            "suppressPackageStartupMessages({library(gamlss.dist); library(jsonlite)})",
            "cat(toJSON(list(" + nl + ("," + nl).join(rows) + nl + "), auto_unbox = TRUE, digits = 17))",
        ]
        rows_r = json.loads(_run_r(lines))
        fam = _family(name)
        for row in rows_r:
            args = (row["y"], row["mu"], row["sigma"], row["nu"], row["tau"])
            for pname, slot in (("mu", "dldm"), ("sigma", "dldd"),
                                ("nu", "dldv"), ("tau", "dldt")):
                ours = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(
                    ours, float(row[slot]), rtol=1e-9, atol=1e-11,
                    err_msg=f"{name} {pname} score vs R's {slot}",
                )
            ours_h = float(np.ravel(np.asarray(
                fam.hessian_functions["tau"](*args), dtype=float))[0])
            np.testing.assert_allclose(
                ours_h, float(row["d2ldt2"]), rtol=1e-9, atol=1e-11,
                err_msg=f"{name} tau hessian vs R's d2ldt2",
            )
