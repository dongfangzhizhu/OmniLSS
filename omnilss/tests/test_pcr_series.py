"""R gamlss.foreach principal-component regression (fitPCR, pc, pcr).

The whole PC path from fitPCR() is compared with the vendored sources for a
free path, a fixed-df path and the supervised variant; pc()/pcr() are compared
for their df, GAIC, t-value and SPCR branches.  The R files need foreach, which
is not installed here, so the reference script defines a sequential stand-in.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.pcr_series import (
    PCFit,
    PCRFit,
    fit_pcr,
    fitPCR,
    gamlss_pc,
    gamlss_pcr,
    get_sig_terms,
    pc,
    pcr,
    weighted_mean,
    weighted_sd,
    which_yX_Corr,
)

_REPO = Path(__file__).resolve().parents[2]

_FOREACH = """foreach <- function(...) {
  args <- list(...); nm <- names(args)
  iter <- nm[nzchar(nm) & !startsWith(nm, ".")][1]
  structure(list(name = iter, values = args[[iter]],
                 combine = if (".combine" %in% nm) args[[".combine"]] else NULL),
            class = "foreachStub")
}
`%dopar%` <- function(obj, expr) {
  e <- substitute(expr); env <- parent.frame()
  out <- lapply(obj$values, function(v) { assign(obj$name, v, envir = env); eval(e, env) })
  if (is.null(obj$combine)) out else do.call(obj$combine, out)
}
"""

_FIT_SCRIPT = _FOREACH + """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
suppressWarnings(source("R/gamlss.foreach/R/PCR-GAMLSS.R"))
suppressWarnings(source("R/gamlss.foreach/R/fitPCR.R"))
suppressWarnings(source("R/gamlss.foreach/R/fitPCR-GAMLSS.R"))
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
X <- as.matrix(d$X); colnames(X) <- paste0("x", 1:ncol(X)); y <- d$y
pack <- function(o) list(coef = as.numeric(o$coefficients), gamma = as.numeric(o$gamma),
  segamma = as.numeric(o$se.gamma), center = as.numeric(o$center), scale = as.numeric(o$scale),
  fv = as.numeric(o$fitted.values), gaic = as.numeric(o$gaic), pc = as.numeric(o$pc),
  sigma = as.numeric(o$sigma), resid = as.numeric(o$residuals), xvar = as.numeric(o$Xvar),
  loadings = as.numeric(o$loadings), M = as.numeric(o$M), meany = as.numeric(o$mean.y))
cat("JSONSTART"); cat(toJSON(list(
  free = pack(fitPCR(x = X, y = y, M = 4, plot = FALSE)),
  fixed = pack(fitPCR(x = X, y = y, M = 4, df = 2, plot = FALSE)),
  sup = pack(fitPCR(x = X, y = y, M = 3, supervised = TRUE, r = 0.3, plot = FALSE))),
  auto_unbox = FALSE, digits = 17))
"""

_SMOOTH_SCRIPT = _FOREACH + """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
suppressWarnings(source("R/gamlss.foreach/R/PCR-GAMLSS.R"))
suppressWarnings(source("R/gamlss.foreach/R/fitPCR.R"))
suppressWarnings(source("R/gamlss.foreach/R/fitPCR-GAMLSS.R"))
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
X <- as.matrix(d$X); colnames(X) <- paste0("x", 1:ncol(X)); y <- d$y; w <- rep(1, length(y))
data <- data.frame(y = y)
packPC <- function(s) {
  f <- gamlss.pc(s, y, w)
  list(fv = as.numeric(f$fitted.values), resid = as.numeric(f$residuals),
       var = as.numeric(f$var), nldf = as.numeric(f$nl.df),
       coef = as.numeric(f$coefSmo$coef), beta = as.numeric(f$coefSmo$beta),
       edf = as.numeric(f$coefSmo$edf), aic = as.numeric(f$coefSmo$AIC))
}
packPCR <- function(s) {
  f <- gamlss.pcr(s, y, w)
  list(fv = as.numeric(f$fitted.values), resid = as.numeric(f$residuals),
       nldf = as.numeric(f$nl.df), pc = as.numeric(f$coefSmo$pc),
       beta = as.numeric(f$coefSmo$coefficients))
}
out <- list(
  pc_df = packPC(pc(x = X, df = 3)),
  pc_gaic = packPC(pc(x = X, method = "GAIC")),
  pc_t = packPC(pc(x = X, method = "t-values")),
  pcr_df = packPCR(pcr(x = X, df = 3)),
  pcr_gaic = packPCR(pcr(x = X, method = "GAIC")),
  pcr_t = packPCR(pcr(x = X, method = "t-values")),
  pcr_spcr = packPCR(pcr(x = X, method = "SPCR")))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 0, n: int = 60):
    rs = np.random.RandomState(seed)
    X = rs.normal(size=(n, 5))
    y = 1.0 + X[:, 0] + 0.5 * X[:, 1] + rs.normal(0.0, 0.3, n)
    return X, y


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
            timeout=600,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def test_weighted_moments_match_the_r_formulas() -> None:
    values = np.array([1.0, 2.0, 4.0, 5.0])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    assert weighted_mean(values, w) == pytest.approx(np.sum(w * values) / np.sum(w))
    expected = np.sqrt(
        (np.sum(w * values**2) * np.sum(w) - np.sum(w * values) ** 2)
        / (np.sum(w) ** 2 - np.sum(w**2))
    )
    assert weighted_sd(values, w) == pytest.approx(expected)
    # With unit weights this is the n - 1 sample sd.
    assert weighted_sd(values, np.ones(4)) == pytest.approx(np.std(values, ddof=1))


def test_fit_pcr_path_invariants() -> None:
    X, y = _data()
    fit = fit_pcr(X, y, M=4)
    assert isinstance(fit, PCRFit)
    assert fit.coefficients.shape == (5, 4)
    assert fit.fitted_values.shape == (60, 4)
    assert fit.gaic.shape == (4,)
    assert fit.pc == int(np.argmin(fit.gaic)) + 1
    scaled = (X - fit.center) / fit.scale
    np.testing.assert_allclose(fit.fitted_values[:, 0], scaled @ fit.coefficients[:, 0] + fit.mean_y, rtol=1e-12)
    np.testing.assert_allclose(fit.residuals, y[:, None] - fit.fitted_values, rtol=1e-12)
    assert fit.M == 4 and fit.k == 2.0


def test_fit_pcr_fixed_df_short_circuits_the_search() -> None:
    X, y = _data()
    fit = fit_pcr(X, y, M=4, df=2)
    assert fit.pc == 2 and fit.M == 2
    assert fit.gaic.size == 0
    np.testing.assert_allclose(fit.fitted(), fit.fitted_values[:, 1], rtol=0)
    np.testing.assert_allclose(fit.residuals_at(2), fit.residuals[:, 1], rtol=0)


def test_fit_pcr_clamps_m_to_the_matrix_rank() -> None:
    X, y = _data()
    fit = fit_pcr(X, y, M=50)
    assert fit.M == 5


def test_fit_pcr_accessors() -> None:
    X, y = _data()
    fit = fit_pcr(X, y, M=4)
    np.testing.assert_allclose(fit.coef(param="gamma"), fit.gamma, rtol=0)
    np.testing.assert_allclose(fit.coef(pc=2), fit.coefficients[:, 1], rtol=0)
    covariance = fit.vcov(pc=2)
    assert covariance.shape == (5, 5)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), fit.vcov(pc=2, type="se"), rtol=1e-12)
    np.testing.assert_allclose(fit.predict(pc=2), fit.fitted(pc=2), rtol=0)
    np.testing.assert_allclose(fit.predict(X, pc=2), fit.fitted(pc=2), rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="not the same"):
        fit.predict(np.zeros((3, 4)))


def test_get_sig_terms_zeroes_the_small_t_values() -> None:
    X, y = _data()
    fit = fit_pcr(X, y, M=4, df=4)
    result = get_sig_terms(fit, X)
    t_values = np.abs(fit.gamma / fit.se_gamma)
    np.testing.assert_allclose(result["gamma"] != 0, t_values > 1.96)
    scaled = (X - fit.center) / fit.scale
    np.testing.assert_allclose(
        result["fitted_values"], scaled @ result["coefficients"] + fit.mean_y, rtol=1e-12
    )


def test_pc_spec_shapes_and_validation() -> None:
    X, _y = _data()
    spec = pc(x=X, df=3)
    assert spec.max_no == 3 and spec.df == 3.0
    assert spec.components["x"].shape == (60, 3)
    assert spec.components["rotation"].shape == (5, 3)
    assert spec.k == pytest.approx(np.log(60))
    with pytest.raises(ValueError, match="less than the number of variables"):
        pc(x=X, df=9)
    with pytest.raises(ValueError, match="has to be set in pc"):
        pc()
    with pytest.raises(ValueError, match="use only one of the arguments"):
        pc(x=X, x_vars=["a"])


def test_pc_reads_a_precomputed_svd() -> None:
    from omnilss.pcr import getSVD

    X, _y = _data()
    spec = pc(x_svd=getSVD(X), df=3)
    assert spec.components["x"].shape[1] == 3
    with pytest.raises(ValueError, match="created by GetSVD"):
        pc(x_svd={"d": np.zeros(2)})


def test_gamlss_pc_df_branch() -> None:
    X, y = _data()
    fit = gamlss_pc(pc(x=X, df=3), y, np.ones(60))
    assert isinstance(fit, PCFit)
    assert fit.edf == 3.0 and fit.nl_df == 3.0 and fit.lambda_ == 0.0
    assert fit.fitted_values.shape == (60,)
    assert np.all(fit.var > 0.0)
    scores = pc(x=X, df=3).components["x"]
    np.testing.assert_allclose(fit.fitted_values, scores @ fit.coef, rtol=1e-12)


def test_gamlss_pc_selection_branches() -> None:
    X, y = _data()
    w = np.ones(60)
    gaic = gamlss_pc(pc(x=X, method="GAIC"), y, w)
    assert 1.0 <= gaic.edf <= 5.0
    significant = gamlss_pc(pc(x=X, method="t-values"), y, w)
    assert 0.0 <= significant.edf <= 5.0
    assert significant.fitted_values.shape == (60,)


def test_gamlss_pc_and_pcr_reject_foreign_specs() -> None:
    X, y = _data()
    with pytest.raises(TypeError, match="returned by pc"):
        gamlss_pc(pcr(x=X), y, np.ones(60))
    with pytest.raises(TypeError, match="returned by pcr"):
        gamlss_pcr(pc(x=X, df=2), y, np.ones(60))


def test_pcr_spec_validation() -> None:
    X, _y = _data()
    spec = pcr(x=X, df=3)
    assert spec.M == 3 and spec.df == 3.0 and spec.method == "GAIC"
    assert spec.state == {"lambda": 0}
    with pytest.raises(ValueError, match="less than the number of variables"):
        pcr(x=X, df=9)
    with pytest.raises(ValueError, match="use only one of the arguments"):
        pcr(x=X, x_vars=["a"])
    with pytest.raises(ValueError, match="has to be set in pcr"):
        pcr()
    with pytest.raises(ValueError, match="should be one of"):
        pcr(x=X, method="GCV")


def test_gamlss_pcr_methods() -> None:
    X, y = _data()
    w = np.ones(60)
    fixed = gamlss_pcr(pcr(x=X, df=3), y, w)
    assert fixed.edf == 3.0 and fixed.nl_df == 3.0 and fixed.lambda_ == 3.0
    np.testing.assert_allclose(fixed.var, fixed.fitted_values, rtol=0)
    gaic = gamlss_pcr(pcr(x=X, method="GAIC"), y, w)
    assert 1.0 <= gaic.edf <= 5.0
    tv = gamlss_pcr(pcr(x=X, method="t-values"), y, w)
    assert tv.gamma is not None and tv.edf == float(np.sum(tv.gamma != 0))
    spcr = gamlss_pcr(pcr(x=X, method="SPCR"), y, w)
    assert spcr.coef_smo.coefficients.shape[0] <= 5


def test_which_yx_corr_selects_and_returns_the_empty_case() -> None:
    X, y = _data()
    names = [f"x{index + 1}" for index in range(5)]
    frame = which_yX_Corr(y, X, r=0.2, print=False, names=names)
    assert list(frame.columns) == ["x1", "x2", "x4"]
    assert frame.shape == (60, 3)
    empty = which_yX_Corr(y, X, r=0.99, print=False, names=names)
    assert empty.shape == (60, 0)
    assert list(empty.columns) == []


@pytest.mark.slow
def test_fitpcr_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.foreach" / "R" / "fitPCR.R").exists():
        pytest.skip("vendored gamlss.foreach not present")
    X, y = _data()
    ref = _run(_FIT_SCRIPT, {"X": X.tolist(), "y": y.tolist()})
    names = [f"x{index + 1}" for index in range(5)]
    cases = {
        "free": fit_pcr(X, y, M=4, names=names),
        "fixed": fit_pcr(X, y, M=4, df=2, names=names),
        "sup": fit_pcr(X, y, M=3, supervised=True, r=0.3, names=names),
    }
    for key, fit in cases.items():
        block = ref[key]
        assert fit.pc == int(np.asarray(block["pc"]).ravel()[0])
        assert fit.M == int(np.asarray(block["M"]).ravel()[0])
        assert fit.mean_y == pytest.approx(float(np.asarray(block["meany"]).ravel()[0]), rel=1e-12)
        np.testing.assert_allclose(fit.center, np.asarray(block["center"], float), rtol=1e-12)
        np.testing.assert_allclose(fit.scale, np.asarray(block["scale"], float), rtol=1e-12)
        np.testing.assert_allclose(fit.gamma, np.asarray(block["gamma"], float), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(fit.se_gamma, np.asarray(block["segamma"], float), rtol=1e-12)
        np.testing.assert_allclose(
            fit.coefficients, np.asarray(block["coef"], float).reshape(fit.coefficients.shape, order="F"),
            rtol=1e-12, atol=1e-13,
        )
        np.testing.assert_allclose(
            fit.fitted_values, np.asarray(block["fv"], float).reshape(fit.fitted_values.shape, order="F"),
            rtol=1e-12, atol=1e-13,
        )
        np.testing.assert_allclose(
            fit.loadings, np.asarray(block["loadings"], float).reshape(fit.loadings.shape, order="F"),
            rtol=1e-12, atol=1e-13,
        )
        np.testing.assert_allclose(fit.sigma, np.asarray(block["sigma"], float), rtol=1e-12)
        np.testing.assert_allclose(fit.xvar, np.asarray(block["xvar"], float), rtol=1e-11)
        gaic = np.asarray(block["gaic"], float)
        if gaic.size == fit.gaic.size:
            np.testing.assert_allclose(fit.gaic, gaic, rtol=1e-12)


@pytest.mark.slow
def test_pc_and_pcr_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.foreach" / "R" / "PCR-GAMLSS.R").exists():
        pytest.skip("vendored gamlss.foreach not present")
    X, y = _data()
    w = np.ones(60)
    ref = _run(_SMOOTH_SCRIPT, {"X": X.tolist(), "y": y.tolist()})

    pc_cases = {
        "pc_df": gamlss_pc(pc(x=X, df=3), y, w),
        "pc_gaic": gamlss_pc(pc(x=X, method="GAIC"), y, w),
        "pc_t": gamlss_pc(pc(x=X, method="t-values"), y, w),
    }
    for key, fit in pc_cases.items():
        block = ref[key]
        assert fit.edf == float(np.asarray(block["edf"]).ravel()[0])
        np.testing.assert_allclose(fit.fitted_values, np.asarray(block["fv"], float), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(fit.residuals, np.asarray(block["resid"], float), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(fit.var, np.asarray(block["var"], float), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(fit.coef, np.asarray(block["coef"], float), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(fit.beta, np.asarray(block["beta"], float), rtol=1e-12, atol=1e-13)
        assert fit.aic == pytest.approx(float(np.asarray(block["aic"]).ravel()[-1]), rel=1e-12)

    pcr_cases = {
        "pcr_df": (gamlss_pcr(pcr(x=X, df=3), y, w), 5),
        "pcr_gaic": (gamlss_pcr(pcr(x=X, method="GAIC"), y, w), 5),
        "pcr_t": (gamlss_pcr(pcr(x=X, method="t-values"), y, w), 5),
        "pcr_spcr": (gamlss_pcr(pcr(x=X, method="SPCR"), y, w), 3),
    }
    for key, (fit, columns) in pcr_cases.items():
        block = ref[key]
        assert fit.edf == float(np.asarray(block["nldf"]).ravel()[0])
        assert fit.coef_smo.pc == int(np.asarray(block["pc"]).ravel()[0])
        np.testing.assert_allclose(fit.fitted_values, np.asarray(block["fv"], float), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(fit.residuals, np.asarray(block["resid"], float), rtol=1e-12, atol=1e-13)
        beta = np.asarray(block["beta"], float)
        matrix = beta.reshape((columns, beta.size // columns), order="F")
        np.testing.assert_allclose(
            fit.coef_smo.coefficients, matrix, rtol=1e-12, atol=1e-13
        )