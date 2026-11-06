"""R gamlss pp() / gamlss.pp power-polynomial smoother, plus scs().

pp()/gamlss.pp() are compared with live R for one, two and three powers.  R's
npoly = 3 branch evaluates the second power with coef[5] instead of coef[6], so
its fitted values disagree with its own coefficients; the port reproduces that
bug and the test pins it down explicitly.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.powerpoly import PPFit, PPSpec, gamlss_pp, pp, scs
from omnilss.smoother_fits import gamlss_cs
from omnilss.smoother_specs import CSSpline, cs

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
set.seed(7)
n <- 60
x <- seq(0.2, 2.0, length.out = n)
y1 <- 1 + 2 * x^0.5 + rnorm(n, 0, 0.1)
y3 <- 1 + 0.5 * x^0.5 + 1.0 * x + 0.2 * x^2 + rnorm(n, 0, 0.02)
w <- rep(1, n)
run <- function(xx, yy, start) {
  s <- pp(xx, start = as.list(start))
  f <- gamlss.pp(s, yy, w)
  list(values = as.numeric(attr(s, "values")), npoly = as.integer(attr(s, "npoly")),
       coef = as.numeric(f$coefSmo), fv = as.numeric(f$fitted.values),
       nl = as.numeric(f$nl.df), var = as.numeric(f$var), resid = as.numeric(f$residuals))
}
out <- list(x = x, y1 = y1, y3 = y3,
            p1 = run(x, y1, c(0.5)),
            p2 = run(x, y1, c(0.5, 1.0)),
            p3 = run(x, y3, c(0.5, 1.0, 2.0)))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _x() -> np.ndarray:
    return np.linspace(0.2, 2.0, 60)


def _y() -> np.ndarray:
    rs = np.random.RandomState(7)
    x = _x()
    return 1.0 + 2.0 * np.sqrt(x) + rs.normal(0.0, 0.1, x.shape[0])


def test_pp_shifts_and_scales_like_r() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    with pytest.warns(UserWarning, match="has been shifted by"):
        spec = pp(x, start=[0.5])
    # fp.scale(): min gap (1) minus min(x) (-2) = 3; the range 5 keeps scale 1.
    assert spec.shift == pytest.approx(3.0)
    assert spec.scale == pytest.approx(1.0)
    np.testing.assert_allclose(spec.values, (x + 3.0) / 1.0)
    positive = pp(_x(), start=[0.5])
    assert positive.shift == 0.0
    # range 1.8 -> 10^(sign(log10 1.8) * trunc(|log10 1.8|)) = 10^0 = 1
    assert positive.scale == pytest.approx(1.0)


def test_pp_decade_scale() -> None:
    x = np.linspace(1.0, 25.0, 10)
    spec = pp(x, start=[0.5])
    assert spec.scale == pytest.approx(10.0)
    np.testing.assert_allclose(spec.values, x / 10.0)


def test_pp_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="no starting values"):
        pp(_x())
    with pytest.raises(ValueError, match="can be 1,2 or 3"):
        pp(_x(), start=[0.5, 1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="bivariate"):
        pp(np.zeros((5, 2)), start=[0.5])


def test_pp_explicit_shift_and_scale_are_kept() -> None:
    spec = pp(_x(), start=[0.5], shift=2.0, scale=5.0)
    assert spec.shift == 2.0 and spec.scale == 5.0
    np.testing.assert_allclose(spec.values, (_x() + 2.0) / 5.0)


def test_gamlss_pp_single_power_recovers_the_curve() -> None:
    x, y = _x(), _y()
    fit = gamlss_pp(pp(x, start=[0.5]), y, np.ones(x.shape[0]))
    assert isinstance(fit, PPFit)
    assert fit.nl_df == 2.0
    assert fit.lambda_ != fit.lambda_  # R stores NA
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)
    np.testing.assert_allclose(fit.var, fit.fitted_values, rtol=0)
    assert fit.coef[0] == pytest.approx(0.5, abs=0.15)


def test_gamlss_pp_two_powers_have_four_parameters() -> None:
    x, y = _x(), _y()
    fit = gamlss_pp(pp(x, start=[0.5, 1.0]), y, np.ones(x.shape[0]))
    assert fit.nl_df == 4.0
    assert fit.coef.shape == (5,)


def test_gamlss_pp_three_powers_reproduce_the_r_typo() -> None:
    x = _x()
    rs = np.random.RandomState(11)
    y = 1.0 + 0.5 * np.sqrt(x) + x + 0.2 * x ** 2 + rs.normal(0.0, 0.02, x.shape[0])
    fit = gamlss_pp(pp(x, start=[0.5, 1.0, 2.0]), y, np.ones(x.shape[0]))
    assert fit.nl_df == 6.0
    assert fit.coef.shape == (7,)
    values = pp(x, start=[0.5, 1.0, 2.0]).values
    powers, beta = fit.coef[:3], fit.coef[3:]
    correct = beta[0] + beta[1] * values ** powers[0] + beta[2] * values ** powers[1] + beta[3] * values ** powers[2]
    r_expression = beta[0] + beta[1] * values ** powers[0] + beta[1] * values ** powers[1] + beta[3] * values ** powers[2]
    np.testing.assert_allclose(fit.fitted_values, r_expression, rtol=1e-12)
    assert np.max(np.abs(fit.fitted_values - correct)) > 1e-3


def test_gamlss_pp_rejects_other_specs() -> None:
    x, y = _x(), _y()
    with pytest.raises(TypeError, match="returned by pp"):
        gamlss_pp(cs(x), y, np.ones(x.shape[0]))


def test_scs_builds_a_cubic_spline_spec() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    spec = scs(x, df=3)
    assert isinstance(spec, CSSpline)
    assert spec.name == "scs"
    assert spec.df == 3.0
    assert spec.na_positions == (2,)
    assert spec.control["penalty"] == 1.4


def test_scs_matches_the_cs_fit() -> None:
    x, y = _x(), _y()
    w = np.ones(x.shape[0])
    ours = gamlss_cs(scs(x, df=3), y, w)
    reference = gamlss_cs(cs(x, df=3), y, w)
    np.testing.assert_allclose(ours.fitted_values, reference.fitted_values, rtol=1e-12)
    assert ours.lambda_ == pytest.approx(reference.lambda_, rel=1e-12)


def test_scs_rejects_matrices() -> None:
    with pytest.raises(ValueError, match="bivariate"):
        scs(np.zeros((4, 2)), df=3)


@pytest.mark.slow
def test_pp_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "powerpoly.R").exists():
        pytest.skip("vendored powerpoly.R not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pp.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300, cwd=str(_REPO)
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    x = np.asarray(ref["x"], dtype=float)
    cases = [
        ("p1", np.asarray(ref["y1"], dtype=float), [0.5]),
        ("p2", np.asarray(ref["y1"], dtype=float), [0.5, 1.0]),
        ("p3", np.asarray(ref["y3"], dtype=float), [0.5, 1.0, 2.0]),
    ]
    for key, y, start in cases:
        spec = pp(x, start=start)
        fit = gamlss_pp(spec, y, np.ones(x.shape[0]))
        block = ref[key]
        np.testing.assert_allclose(spec.values, np.asarray(block["values"], dtype=float), rtol=0, atol=0)
        assert int(np.asarray(block["npoly"]).ravel()[0]) == spec.npoly
        # nls stops at its own convergence tolerance, so the powers agree to ~5e-5.
        np.testing.assert_allclose(
            fit.coef, np.asarray(block["coef"], dtype=float), rtol=0, atol=1e-4
        )
        np.testing.assert_allclose(
            fit.fitted_values, np.asarray(block["fv"], dtype=float), rtol=0, atol=1e-5
        )
        assert fit.nl_df == float(np.asarray(block["nl"]).ravel()[0])
