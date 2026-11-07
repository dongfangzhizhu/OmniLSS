"""R gamlss tail diagnostics: the log-survival curve helpers and loglogSurv1/2/3.

loglogSurv1/2/3 regress log(-log(S(y))) on a transformed tail covariate; R calls
gamlss(newY ~ x), which for a normal linear model is OLS.  Two datasets x three
transforms are compared with live R (coefficients, error sum of squares,
threshold, transformed covariate and response).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.FitTail import (
    LogLogSurvFit,
    logSurv,
    loglogSurv,
    loglogSurv1,
    loglogSurv2,
    loglogSurv3,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
source("R/gamlss/R/FitTail.R")
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
pack <- function(y, prob, type) {
  cdF <- ECDF(y); mY <- quantile(y, probs = prob); Y <- y[y > mY]
  newY <- log(-log(1 - cdF(Y)))
  x <- switch(type, log(log(Y)), log(Y), Y)
  m <- switch(type,
              loglogSurv1(y, prob = prob, print = FALSE),
              loglogSurv2(y, prob = prob, print = FALSE),
              loglogSurv3(y, prob = prob, print = FALSE))
  list(coef = as.numeric(coef(m)), ess = sum((newY - fitted(m))^2),
       x = as.numeric(x), newY = as.numeric(newY), thr = as.numeric(mY),
       n = length(Y))
}
y1 <- d$y1; y2 <- d$y2
out <- list(
  a1 = pack(y1, 0.9, 1), a2 = pack(y1, 0.9, 2), a3 = pack(y1, 0.9, 3),
  b1 = pack(y2, 0.8, 1), b2 = pack(y2, 0.8, 2), b3 = pack(y2, 0.8, 3))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _data(seed: int = 0):
    rs = np.random.RandomState(seed)
    y1 = 1.0 + rs.weibull(1.5, size=200) * 2
    y2 = np.exp(rs.normal(1.0, 0.8, size=150))
    return y1, y2


def test_logSurv_is_nonincreasing() -> None:
    y = np.array([0.5, 1.0, 2.0, 4.0])
    out = logSurv(y)
    assert out.x.shape == y.shape
    assert out.curve.shape == y.shape
    # survival decreases => log survival non-increasing
    assert np.all(np.diff(out.curve) <= 1e-12)


def test_loglogSurv_finite_values() -> None:
    y = np.array([1.0, 1.5, 2.0, 3.0])
    out = loglogSurv(y)
    assert np.all(np.isfinite(out.curve))


def test_type_ii_regresses_on_log_y() -> None:
    y1, _y2 = _data()
    fit = loglogSurv2(y1, prob=0.9, print=False)
    assert isinstance(fit, LogLogSurvFit)
    threshold = np.quantile(y1, 0.9)
    tail = y1[y1 > threshold]
    np.testing.assert_allclose(fit.tail, tail, rtol=0)
    np.testing.assert_allclose(fit.x, np.log(tail), rtol=1e-15)
    design = np.column_stack([np.ones(tail.size), np.log(tail)])
    expected, *_ = np.linalg.lstsq(design, fit.y, rcond=None)
    np.testing.assert_allclose(fit.coefficients, expected, rtol=1e-13)
    np.testing.assert_allclose(fit.ess, np.sum((fit.y - design @ expected) ** 2), rtol=1e-13)
    assert fit.sigma == pytest.approx(np.sqrt(fit.ess / tail.size), rel=1e-13)


def test_the_three_types_differ_only_in_the_covariate() -> None:
    y1, _y2 = _data()
    first = loglogSurv1(y1, prob=0.9, print=False)
    second = loglogSurv2(y1, prob=0.9, print=False)
    third = loglogSurv3(y1, prob=0.9, print=False)
    np.testing.assert_allclose(first.tail, second.tail, rtol=0)
    np.testing.assert_allclose(first.y, second.y, rtol=0)
    np.testing.assert_allclose(first.x, np.log(np.log(first.tail)), rtol=1e-14)
    np.testing.assert_allclose(second.x, np.log(second.tail), rtol=1e-15)
    np.testing.assert_allclose(third.x, third.tail, rtol=0)
    assert (first.type, second.type, third.type) == (1, 2, 3)


def test_the_empirical_survival_uses_unique_values_and_n_plus_one() -> None:
    y = np.array([1.2, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0])
    fit = loglogSurv3(y, prob=0.5, print=False)
    threshold = np.quantile(y, 0.5)
    tail = y[y > threshold]
    count = y.size
    expected = []
    for value in tail:
        # stepfun(ysort, c(0, ecdf)) is left-continuous in R, so the value at a
        # support point is the cumulative count up to and including it, over n + 1.
        below = np.sum(y <= value)
        expected.append(np.log(-np.log(1.0 - below / (count + 1.0))))
    np.testing.assert_allclose(fit.y, np.asarray(expected), rtol=1e-14)


def test_threshold_below_one_is_rejected() -> None:
    y = np.linspace(0.1, 0.9, 50)
    with pytest.raises(ValueError, match="must be greater than 1"):
        loglogSurv1(y, print=False)


def test_prob_is_validated_and_an_empty_tail_is_reported() -> None:
    y1, _y2 = _data()
    with pytest.raises(ValueError, match="prob must be between 0 and 1"):
        loglogSurv2(y1, prob=1.5, print=False)
    with pytest.raises(ValueError, match="no observations above"):
        loglogSurv2(y1, prob=1.0, print=False)


def test_print_reports_the_coefficients_and_ess(capsys) -> None:
    y1, _y2 = _data()
    fit = loglogSurv2(y1, prob=0.9, print=True)
    captured = capsys.readouterr().out
    assert "coefficients [" in captured
    assert "error sum of squares" in captured
    assert f"{fit.coefficients[0]:.8f}" in captured


def test_plot_is_not_ported_but_announces_itself() -> None:
    y1, _y2 = _data()
    with pytest.warns(UserWarning, match="plot is not ported"):
        loglogSurv1(y1, prob=0.9, print=False, plot=True)


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "FitTail.R").exists():
        pytest.skip("vendored FitTail.R not present")
    y1, y2 = _data()
    payload = {"y1": y1.tolist(), "y2": y2.tolist()}
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "tail.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    cases = {
        "a1": (y1, 0.9, loglogSurv1),
        "a2": (y1, 0.9, loglogSurv2),
        "a3": (y1, 0.9, loglogSurv3),
        "b1": (y2, 0.8, loglogSurv1),
        "b2": (y2, 0.8, loglogSurv2),
        "b3": (y2, 0.8, loglogSurv3),
    }
    for key, (data, prob, function) in cases.items():
        block = ref[key]
        fit = function(data, prob=prob, print=False)
        np.testing.assert_allclose(
            fit.coefficients, np.asarray(block["coef"], float), rtol=1e-12, atol=1e-14, err_msg=key
        )
        assert fit.ess == pytest.approx(float(np.asarray(block["ess"], float).ravel()[0]), rel=1e-12)
        assert fit.threshold == pytest.approx(
            float(np.asarray(block["thr"], float).ravel()[0]), rel=0, abs=0
        )
        np.testing.assert_allclose(fit.x, np.asarray(block["x"], float), rtol=0, atol=0, err_msg=key)
        np.testing.assert_allclose(fit.y, np.asarray(block["newY"], float), rtol=0, atol=0, err_msg=key)
        assert fit.tail.size == int(np.asarray(block["n"]).ravel()[0])