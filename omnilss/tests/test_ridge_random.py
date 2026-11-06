"""R gamlss.ri / gamlss.random backfitting steps.

ri() is checked for a fixed lambda, a df target, an Lp other than 2, and the
ML / GAIC lambda searches; random() for a fixed lambda, the ML fixed point and
a df target.  Every case is compared with live R on the same data.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnilss.ridge_random import (
    RIFit,
    RISpec,
    RandomFit,
    RandomSpec,
    gamlss_random,
    gamlss_ri,
    random_spec,
    ri,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
X <- matrix(d$X, nrow = d$nr, ncol = d$nc); y <- d$y; w <- rep(1, length(y))
Xs <- scale(X); p <- ncol(Xs)
mk <- function(lam = NULL, df = NULL, method = "ML", Lp = 2, order = 0, start = 10, k = 2) {
  s <- rep(0, nrow(Xs))
  attr(s, "X") <- Xs
  attr(s, "D") <- if (order == 0) diag(p) else diff(diag(p), diff = order)
  attr(s, "lambda") <- lam; attr(s, "df") <- df
  attr(s, "method") <- method; attr(s, "Lp") <- Lp; attr(s, "kappa") <- 1e-5
  attr(s, "iter") <- 100; attr(s, "c.crit") <- 1e-6; attr(s, "order") <- order
  attr(s, "start") <- start; attr(s, "k") <- k
  e <- new.env(); assign("start.Lambda.abcd", start, envir = e)
  attr(s, "gamlss.env") <- e; attr(s, "NameForLambda") <- "start.Lambda.abcd"
  attr(s, "namesX") <- as.character(1:p)
  s
}
pack <- function(f) list(lambda = f$lambda, edf = f$coefSmo$edf, nldf = f$nl.df,
                         coef = as.numeric(f$coefSmo$coef), fv = as.numeric(f$fitted.values),
                         var = as.numeric(f$var))
gfit <- function(...) pack(gamlss.ri(mk(...), y, w))
rr <- function(df = NULL, lam = NULL) {
  xf <- factor(d$g, levels = seq_len(max(d$g)))
  s <- rep(0, length(xf))
  attr(s, "contrasts") <- C(xf, rep(0, nlevels(xf)), 1)
  attr(s, "df") <- df; attr(s, "lambda") <- lam; attr(s, "start") <- 10
  f <- gamlss.random(s, y, w)
  list(lambda = f$lambda, edf = f$coefSmo$edf, nldf = f$nl.df,
       coef = as.numeric(f$coefSmo$coef), fv = as.numeric(f$y), var = as.numeric(f$var))
}
out <- list(fixed = gfit(lam = 3), bydf = gfit(df = 1.5), lp1 = gfit(lam = 3, Lp = 1),
            ml = gfit(), gaic = gfit(method = "GAIC"),
            rnd_ml = rr(), rnd_fixed = rr(lam = 5), rnd_df = rr(df = 1.5))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 0, n: int = 60):
    rs = np.random.RandomState(seed)
    x1 = rs.normal(size=n)
    x2 = rs.normal(size=n)
    X = np.column_stack([x1, x2])
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rs.normal(0.0, 0.3, n)
    codes = rs.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2])
    factor = pd.Series(pd.Categorical(codes, categories=[1, 2, 3]))
    return X, y, np.ones(n), factor


def _reference() -> dict:
    X, y, _w, factor = _data()
    payload = {
        "X": X.flatten(order="F").tolist(),
        "nr": X.shape[0],
        "nc": X.shape[1],
        "y": y.tolist(),
        "g": (factor.cat.codes + 1).tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "rr.R"
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
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _scalar(block: dict, key: str) -> float:
    values = np.asarray(block[key], dtype=float).ravel()
    return float(values[0]) if values.size else float("nan")


def test_ri_scales_the_design() -> None:
    X, _y, _w, _f = _data()
    spec = ri(X, lambda_=3.0)
    assert isinstance(spec, RISpec)
    np.testing.assert_allclose(spec.X.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(spec.X.std(axis=0, ddof=1), 1.0, rtol=1e-12)
    assert spec.D.shape == (2, 2)
    np.testing.assert_allclose(spec.D, np.eye(2))


def test_ri_requires_a_df_or_a_lambda() -> None:
    X, _y, _w, _f = _data()
    with pytest.raises(ValueError, match="Specify at least one"):
        ri(X)


def test_ri_guards_and_warnings() -> None:
    X, _y, _w, _f = _data()
    with pytest.warns(UserWarning, match="lambda was negative"):
        assert ri(X, lambda_=-2.0).lambda_ == 0.0
    with pytest.warns(UserWarning, match="the df are set to p"):
        assert ri(X, df=10.0).df == 2.0
    with pytest.warns(UserWarning, match="the df are set to 0"):
        assert ri(X, df=-1.0).df == 0.0


def test_ri_order_builds_a_difference_penalty() -> None:
    X, _y, _w, _f = _data()
    spec = ri(X, df=1.0, order=1)
    assert spec.D.shape == (1, 2)
    np.testing.assert_allclose(spec.D, np.array([[-1.0, 1.0]]))


def test_gamlss_ri_fixed_lambda_relations() -> None:
    X, y, w, _f = _data()
    fit = gamlss_ri(ri(X, lambda_=3.0), y, w)
    assert isinstance(fit, RIFit)
    assert fit.lambda_ == 3.0
    assert fit.nl_df == pytest.approx(fit.edf - 1.0, rel=1e-12)
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)
    assert fit.edf < 2.0


def test_gamlss_ri_df_target() -> None:
    X, y, w, _f = _data()
    fit = gamlss_ri(ri(X, df=1.5), y, w)
    assert fit.edf == pytest.approx(1.5, abs=1e-6)


def test_gamlss_ri_lp_changes_the_fit() -> None:
    X, y, w, _f = _data()
    ridge = gamlss_ri(ri(X, lambda_=3.0), y, w)
    lasso_like = gamlss_ri(ri(X, lambda_=3.0, Lp=1.0), y, w)
    assert not np.allclose(ridge.coef, lasso_like.coef)
    assert lasso_like.edf != pytest.approx(ridge.edf)


def test_gamlss_ri_ml_and_gaic_select_lambda() -> None:
    X, y, w, _f = _data()
    base = replace(ri(X, lambda_=1.0), lambda_=None)
    ml = gamlss_ri(base, y, w)
    gaic = gamlss_ri(replace(base, method="GAIC"), y, w)
    assert ml.lambda_ > 0.0 and gaic.lambda_ > 0.0
    assert ml.sigb2 is not None and ml.sige2 is not None
    assert gaic.sigb2 is None and gaic.sige2 is None


def test_gamlss_ri_predicts_with_the_appended_rows() -> None:
    X, y, w, _f = _data()
    extra = np.array([[1.0, -1.0], [0.5, 0.5]])
    base = ri(X, lambda_=3.0)
    # R appends prediction rows after ri() has standardised the fitting data, so
    # the same centre/scale has to be applied to them.
    appended = (extra - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    spec = replace(base, X=np.vstack([base.X, appended]))
    predicted = gamlss_ri(spec, y, w, xeval=np.zeros(2))
    fit = gamlss_ri(base, y, w)
    np.testing.assert_allclose(predicted, appended @ fit.coef, rtol=1e-12)


def test_gamlss_ri_rejects_other_specs() -> None:
    X, y, w, _f = _data()
    with pytest.raises(TypeError, match="returned by ri"):
        gamlss_ri(random_spec(_f), y, w)


def test_random_spec_encodes_the_factor() -> None:
    X, _y, _w, factor = _data()
    spec = random_spec(factor)
    assert isinstance(spec, RandomSpec)
    assert list(spec.codes[:3]) == list(factor.cat.codes[:3] + 1)
    assert spec.levels == [1, 2, 3]
    assert spec.X.tolist() == [0.0] * len(factor)


def test_random_spec_requires_a_factor_and_clamps_df() -> None:
    X, _y, _w, factor = _data()
    with pytest.raises(ValueError, match="expects a factor"):
        random_spec(pd.Series([1.0, 2.0, 3.0]))
    assert random_spec(factor, df=10.0).df == 3.0
    assert random_spec(factor, df=-1.0).df == 0.001


def test_gamlss_random_fixed_lambda_uses_the_level_formula() -> None:
    _X, y, w, factor = _data()
    codes = factor.cat.codes.to_numpy() + 1
    fit = gamlss_random(random_spec(factor, lambda_=5.0), y, w)
    assert isinstance(fit, RandomFit)
    expected = np.array(
        [np.sum((w * y)[codes == level]) / (np.sum(w[codes == level]) + 5.0) for level in (1, 2, 3)]
    )
    np.testing.assert_allclose(fit.coef, expected, rtol=1e-12)
    np.testing.assert_allclose(fit.fitted_values, expected[codes - 1], rtol=1e-12)
    assert fit.nl_df == pytest.approx(fit.edf, rel=1e-12)


def test_gamlss_random_ml_and_df_paths() -> None:
    _X, y, w, factor = _data()
    ml = gamlss_random(random_spec(factor), y, w)
    by_df = gamlss_random(random_spec(factor, df=1.5), y, w)
    assert ml.lambda_ > 0.0 and ml.sigb2 > 0.0
    assert by_df.edf == pytest.approx(1.5, abs=1e-3)


def test_gamlss_random_prediction_is_not_portable() -> None:
    _X, y, w, factor = _data()
    with pytest.raises(NotImplementedError, match="prediction branch"):
        gamlss_random(random_spec(factor, lambda_=5.0), y, w, xeval=np.array([1.0]))


@pytest.mark.slow
def test_ri_and_random_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "ri.R").exists():
        pytest.skip("vendored gamlss ri.R not present")
    ref = _reference()
    X, y, w, factor = _data()

    fixed = gamlss_ri(ri(X, lambda_=3.0), y, w)
    np.testing.assert_allclose(fixed.coef, np.asarray(ref["fixed"]["coef"], float), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        fixed.fitted_values, np.asarray(ref["fixed"]["fv"], float), rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(fixed.var, np.asarray(ref["fixed"]["var"], float), rtol=1e-12)
    assert fixed.edf == pytest.approx(_scalar(ref["fixed"], "edf"), rel=1e-12)

    by_df = gamlss_ri(ri(X, df=1.5), y, w)
    # Both solve the same uniroot system; the residual gap is root tolerance.
    assert by_df.lambda_ == pytest.approx(_scalar(ref["bydf"], "lambda"), rel=1e-9)
    assert by_df.edf == pytest.approx(_scalar(ref["bydf"], "edf"), abs=1e-9)

    lasso = gamlss_ri(ri(X, lambda_=3.0, Lp=1.0), y, w)
    np.testing.assert_allclose(lasso.coef, np.asarray(ref["lp1"]["coef"], float), rtol=1e-12, atol=1e-13)

    base = replace(ri(X, lambda_=1.0), lambda_=None)
    ml = gamlss_ri(base, y, w)
    assert ml.lambda_ == pytest.approx(_scalar(ref["ml"], "lambda"), rel=1e-10)
    np.testing.assert_allclose(ml.coef, np.asarray(ref["ml"]["coef"], float), rtol=1e-10, atol=1e-12)

    gaic = gamlss_ri(replace(base, method="GAIC"), y, w)
    assert gaic.lambda_ == pytest.approx(_scalar(ref["gaic"], "lambda"), rel=1e-4)

    rfit = gamlss_random(random_spec(factor), y, w)
    assert rfit.lambda_ == pytest.approx(_scalar(ref["rnd_ml"], "lambda"), rel=1e-12)
    np.testing.assert_allclose(rfit.coef, np.asarray(ref["rnd_ml"]["coef"], float), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(rfit.var, np.asarray(ref["rnd_ml"]["var"], float), rtol=1e-12)

    rfixed = gamlss_random(random_spec(factor, lambda_=5.0), y, w)
    np.testing.assert_allclose(rfixed.fitted_values, np.asarray(ref["rnd_fixed"]["fv"], float), rtol=1e-12)

    rdf = gamlss_random(random_spec(factor, df=1.5), y, w)
    assert rdf.lambda_ == pytest.approx(_scalar(ref["rnd_df"], "lambda"), rel=1e-9)
