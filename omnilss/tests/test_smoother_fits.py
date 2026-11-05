"""R gamlss.pb / gamlss.ps backfitting steps.

Both are compared with live R on the same data.  gamlss.pb is checked for a
fixed lambda, a df target, and the ML / GAIC / GCV lambda searches; gamlss.ps
is checked for the df-driven golden-section search and for a fixed lambda.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from omnilss.smoother_fits import SmootherFit, gamlss_pb, gamlss_ps
from omnilss.smoother_specs import pb, ps

_REPO = Path(__file__).resolve().parents[2]

_PB_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
x <- d$x; y <- d$y; w <- rep(1, length(x)); xe <- d$xeval
run <- function(lam = NULL, df = NULL, method = "ML", k = 2) {
  s <- pb(x)
  attr(s, "lambda") <- lam
  attr(s, "df") <- df
  ctl <- attr(s, "control"); ctl$method <- method; ctl$k <- k
  attr(s, "control") <- ctl
  e <- new.env(); assign("start.Lambda.abcd", 10, envir = e)
  attr(s, "gamlss.env") <- e; attr(s, "NameForLambda") <- "start.Lambda.abcd"
  f <- gamlss.pb(s, y, w)
  list(lambda = f$lambda, edf = f$coefSmo$edf, nldf = f$nl.df,
       beta = as.numeric(f$coefSmo$coef), fv = as.numeric(f$fitted.values),
       var = as.numeric(f$var), pred = as.numeric(f$coefSmo$fun(xe)))
}
out <- list(fixed = run(lam = 3), bydf = run(df = 5), ml = run(method = "ML"),
            gaic = run(method = "GAIC", k = 2), gcv = run(method = "GCV", k = 2))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""

_PS_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
x <- d$x; y <- d$y; w <- rep(1, length(x))
run <- function(df = 3, lam = NULL) {
  s <- ps(x, df = df, lambda = lam)
  f <- gamlss.ps(s, y, w)
  list(lambda = f$lambda, nldf = f$nl.df, coef = as.numeric(f$coefSmo$coef),
       fv = as.numeric(f$fitted.values), var = as.numeric(f$var))
}
out <- list(df3 = run(df = 3), fixed = run(df = 3, lam = 50))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 7, n: int = 60):
    rs = np.random.RandomState(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rs.normal(0.0, 0.2, n)
    return x, y, np.ones(n)


def _run(script: str, payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "s.R"
        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def test_fixed_lambda_fit_shape_and_relations() -> None:
    x, y, w = _data()
    fit = gamlss_pb(pb(x, lambda_=3.0), y, w)
    assert isinstance(fit, SmootherFit)
    assert fit.fitted_values.shape == (60,)
    # pb() drops to 10 intervals below 99 observations, so n=60 gives 13 columns.
    assert fit.coef.shape == (13,)
    assert fit.nl_df == pytest.approx(fit.edf - 2.0, rel=1e-12)
    assert fit.lambda_ == 3.0
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)


def test_prediction_uses_a_natural_spline() -> None:
    x, y, w = _data()
    fit = gamlss_pb(pb(x, lambda_=3.0), y, w)
    grid = np.linspace(0.0, 1.0, 11)
    predicted = fit.predict(grid)
    # The spline interpolates the fitted values at the observed x.
    np.testing.assert_allclose(fit.predict(x), fit.fitted_values, rtol=1e-10, atol=1e-12)
    assert predicted.shape == (11,)
    np.testing.assert_allclose(predicted[0], fit.fitted_values[0], rtol=1e-10)


def test_xeval_returns_the_prediction() -> None:
    x, y, w = _data()
    spec = pb(x, lambda_=3.0)
    grid = np.linspace(0.0, 1.0, 6)
    np.testing.assert_allclose(
        gamlss_pb(spec, y, w, xeval=grid), gamlss_pb(spec, y, w).predict(grid), rtol=1e-12
    )


def test_df_target_reaches_the_stored_df() -> None:
    x, y, w = _data()
    spec = replace(pb(x), df=5.0)
    fit = gamlss_pb(spec, y, w)
    assert fit.edf == pytest.approx(5.0, abs=1e-4)
    # pb(df=3) stores 3 + 2, so it targets the same edf.
    assert gamlss_pb(pb(x, df=3), y, w).edf == pytest.approx(5.0, abs=1e-4)


def test_lambda_selection_methods_differ_in_lambda() -> None:
    x, y, w = _data()
    base = pb(x)
    lambdas = {}
    for method in ("ML", "GAIC", "GCV"):
        spec = replace(base, control={**base.control, "method": method})
        lambdas[method] = gamlss_pb(spec, y, w).lambda_
    assert len({round(value, 6) for value in lambdas.values()}) == 3
    assert all(1e-7 <= value <= 1e7 for value in lambdas.values())


def test_max_df_caps_the_effective_df() -> None:
    x, y, w = _data()
    spec = replace(pb(x), control={**pb(x).control, "method": "ML"}, max_df=5.0)
    fit = gamlss_pb(spec, y, w)
    assert fit.edf <= 5.0 + 1e-6


def test_pb_requires_matching_lengths() -> None:
    x, _y, _w = _data()
    with pytest.raises(ValueError, match="different lengths"):
        gamlss_pb(pb(x), np.zeros(5), np.ones(5))


def test_ps_df_driven_lambda_and_nl_df() -> None:
    x, y, w = _data(seed=11)
    fit = gamlss_ps(ps(x, df=3), y, w)
    # ps() stores df + 2, so a df = 3 target gives edf = 5 and nl.df = 3.
    assert fit.nl_df == pytest.approx(3.0, abs=1e-3)
    assert fit.edf == pytest.approx(fit.nl_df + 2.0, rel=1e-12)
    assert fit.lambda_ > 0.0


def test_ps_fixed_lambda() -> None:
    x, y, w = _data(seed=11)
    fit = gamlss_ps(ps(x, df=3, lambda_=50.0), y, w)
    assert fit.lambda_ == 50.0
    assert fit.nl_df > 0.0


def test_ps_rejects_other_specs() -> None:
    x, y, w = _data()
    with pytest.raises(TypeError, match="returned by ps"):
        gamlss_ps(pb(x), y, w)


@pytest.mark.slow
def test_pb_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "pb.R").exists():
        pytest.skip("vendored pb.R not present")
    x, y, w = _data()
    grid = np.linspace(0.05, 0.95, 7)
    ref = _run(_PB_SCRIPT, {"x": x.tolist(), "y": y.tolist(), "xeval": grid.tolist()})

    fixed = gamlss_pb(pb(x, lambda_=3.0), y, w)
    np.testing.assert_allclose(fixed.coef, np.asarray(ref["fixed"]["beta"], float), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        fixed.fitted_values, np.asarray(ref["fixed"]["fv"], float), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(fixed.var, np.asarray(ref["fixed"]["var"], float), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(fixed.predict(grid), np.asarray(ref["fixed"]["pred"], float), rtol=1e-10)

    by_df = gamlss_pb(replace(pb(x), df=5.0), y, w)
    # R stops the root search at uniroot's 1e-4 tolerance; we solve it tighter,
    # so both land on the target df within R's own accuracy.
    assert by_df.edf == pytest.approx(float(np.asarray(ref["bydf"]["edf"]).ravel()[0]), abs=1e-4)

    base = pb(x)
    for method, key in (("ML", "ml"), ("GAIC", "gaic"), ("GCV", "gcv")):
        spec = replace(base, control={**base.control, "method": method})
        fit = gamlss_pb(spec, y, w)
        assert fit.lambda_ == pytest.approx(
            float(np.asarray(ref[key]["lambda"]).ravel()[0]), rel=1e-5
        )
        assert fit.edf == pytest.approx(float(np.asarray(ref[key]["edf"]).ravel()[0]), abs=1e-5)


@pytest.mark.slow
def test_ps_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "PS.R").exists():
        pytest.skip("vendored PS.R not present")
    x, y, w = _data(seed=11)
    ref = _run(_PS_SCRIPT, {"x": x.tolist(), "y": y.tolist()})

    df_fit = gamlss_ps(ps(x, df=3), y, w)
    assert df_fit.lambda_ == pytest.approx(
        float(np.asarray(ref["df3"]["lambda"]).ravel()[0]), rel=1e-10
    )
    np.testing.assert_allclose(df_fit.coef, np.asarray(ref["df3"]["coef"], float), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        df_fit.fitted_values, np.asarray(ref["df3"]["fv"], float), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(df_fit.var, np.asarray(ref["df3"]["var"], float), rtol=1e-10, atol=1e-13)

    fixed = gamlss_ps(ps(x, df=3, lambda_=50.0), y, w)
    np.testing.assert_allclose(fixed.coef, np.asarray(ref["fixed"]["coef"], float), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        fixed.fitted_values, np.asarray(ref["fixed"]["fv"], float), rtol=1e-11, atol=1e-12
    )
