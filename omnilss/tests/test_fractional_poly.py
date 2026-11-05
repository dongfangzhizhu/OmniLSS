"""R gamlss fractional-polynomial smoother: bfp / fp / gamlss.fp.

All three are compared element-wise with live R (R/gamlss/R/fracpoly.R); the
basis, spec attributes, selected powers, fitted/residual/variance and
coefficients agree to machine precision.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.fractional_poly import FPFit, FPSpec, bfp, fp, gamlss_fp

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
x <- d$x; y <- d$y; w <- rep(1, length(x))
s <- fp(x, npoly = 2)
B <- bfp(x, powers = c(0, 1))
f <- gamlss.fp(s, y, w)
cat("JSONSTART"); cat(toJSON(list(
  B = as.numeric(B), Bdim = as.integer(dim(B)),
  values = as.numeric(attr(s, "values")),
  shift = attr(s, "shift"), scale = attr(s, "scale"), df = attr(s, "df"),
  fv = as.numeric(f$fitted.values), resid = as.numeric(f$residuals),
  var = as.numeric(f$var), nldf = f$nl.df, powers = as.numeric(f$lambda),
  coef = as.numeric(coef(f$coefSmo))
), auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 0, n: int = 80):
    np.random.seed(seed)
    x = np.linspace(0.1, 2.0, n)
    y = 1.0 + 2.0 * np.log(x) + np.random.normal(0.0, 0.2, n)
    return x, y, np.ones(n)


def test_bfp_basis_with_log_and_power() -> None:
    x = np.array([0.5, 1.0, 2.0])
    basis = bfp(x, (0.0, 1.0), shift=0.0, scale=1.0)
    np.testing.assert_allclose(basis[:, 0], np.log(x), rtol=1e-12)
    np.testing.assert_allclose(basis[:, 1], x, rtol=1e-12)


def test_bfp_repeated_power_uses_log_times_previous() -> None:
    x = np.array([0.5, 1.0, 2.0])
    basis = bfp(x, (1.0, 1.0), shift=0.0, scale=1.0)
    np.testing.assert_allclose(basis[:, 1], np.log(x) * x, rtol=1e-12)


def test_bfp_computes_shift_and_scale_when_missing() -> None:
    x = np.linspace(0.1, 2.0, 20)
    # fp.scale: positive x -> shift 0; range ~1.9 -> scale 1.
    basis = bfp(x, (1.0,), shift=None, scale=None)
    np.testing.assert_allclose(basis[:, 0], x, rtol=1e-12)


def test_fp_spec() -> None:
    x, _, _ = _data()
    spec = fp(x, npoly=2)
    assert isinstance(spec, FPSpec)
    assert spec.npoly == 2
    assert spec.df == 4.0
    assert spec.smooth_class == "smooth"
    assert spec.na_positions == ()
    np.testing.assert_allclose(spec.values, x, rtol=1e-12)


def test_fp_df_scales_with_npoly() -> None:
    x, _, _ = _data()
    assert fp(x, npoly=1).df == 2.0
    assert fp(x, npoly=3).df == 6.0


def test_fp_shift_for_nonpositive_x() -> None:
    x = np.array([0.0, 1.0, 2.0, 4.0])
    with pytest.warns(UserWarning, match="shifted by"):
        spec = fp(x, npoly=1)
    assert spec.shift > 0.0
    assert np.all(spec.values > 0.0)


def test_fp_validates_npoly_and_dimensions() -> None:
    x, _, _ = _data()
    with pytest.raises(ValueError, match="can be 1,2 or 3"):
        fp(x, npoly=4)
    with pytest.raises(ValueError, match="bivariate"):
        fp(np.zeros((5, 2)))


def test_gamlss_fp_fit_contract() -> None:
    x, y, w = _data()
    fit = gamlss_fp(fp(x, npoly=2), y, w)
    assert isinstance(fit, FPFit)
    assert fit.fitted_values.shape == y.shape
    assert fit.powers.shape == (2,)
    assert fit.nl_df == 4.0
    assert fit.deviance > 0.0
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)
    assert np.all(np.isfinite(fit.var))


def test_gamlss_fp_searches_all_npoly_values() -> None:
    x, y, w = _data()
    for npoly in (1, 2, 3):
        fit = gamlss_fp(fp(x, npoly=npoly), y, w)
        assert fit.powers.shape == (npoly,)
        assert fit.nl_df == 2 * npoly
        assert all(power in (-2, -1, -0.5, 0, 0.5, 1, 2, 3) for power in fit.powers)


def test_gamlss_fp_prediction_matches_the_fit_at_observed_x() -> None:
    x, y, w = _data()
    spec = fp(x, npoly=2)
    fit = gamlss_fp(spec, y, w)
    predicted = gamlss_fp(spec, y, w, xeval=x)
    np.testing.assert_allclose(predicted, fit.fitted_values, rtol=1e-10, atol=1e-12)


def test_gamlss_fp_rejects_other_specs_and_bad_lengths() -> None:
    x, y, w = _data()
    from omnilss.smoother_specs import pb

    with pytest.raises(TypeError, match="returned by fp"):
        gamlss_fp(pb(x), y, w)
    with pytest.raises(ValueError, match="different lengths"):
        gamlss_fp(fp(x, npoly=2), np.zeros(5), np.ones(5))


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "fracpoly.R").exists():
        pytest.skip("vendored fracpoly.R not present")
    x, y, w = _data()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps({"x": x.tolist(), "y": y.tolist()}), encoding="utf-8")
        script_path = Path(tmp) / "s.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    def scalar(key: str) -> float:
        return float(np.asarray(ref[key]).ravel()[0])

    basis = bfp(x, (0.0, 1.0))
    np.testing.assert_allclose(
        basis,
        np.asarray(ref["B"], float).reshape(tuple(int(v) for v in ref["Bdim"]), order="F"),
        rtol=1e-14,
    )
    spec = fp(x, npoly=2)
    np.testing.assert_allclose(spec.values, np.asarray(ref["values"], float), rtol=1e-14)
    assert spec.shift == pytest.approx(scalar("shift"))
    assert spec.scale == pytest.approx(scalar("scale"))
    assert spec.df == pytest.approx(scalar("df"))

    fit = gamlss_fp(spec, y, w)
    np.testing.assert_allclose(fit.powers, np.asarray(ref["powers"], float), rtol=0, atol=0)
    np.testing.assert_allclose(fit.fitted_values, np.asarray(ref["fv"], float), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fit.residuals, np.asarray(ref["resid"], float), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fit.var, np.asarray(ref["var"], float), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fit.coefficients, np.asarray(ref["coef"], float), rtol=1e-12, atol=1e-14)
    assert fit.nl_df == pytest.approx(scalar("nldf"))
