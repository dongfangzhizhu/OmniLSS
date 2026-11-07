"""R gamlss.tr::fitTail / fitTailAll -- live-R parity and ML stationarity.

R source: ``R/gamlss.tr/R/FitTrunTail.R``.

fitTail selects a tail of y, builds a truncated family whose cut point is the
last observation *outside* the tail, and fits it with gamlssML (link-scale
numerical maximisation).  fitTailAll repeats that fit for a decreasing number
of tail observations and returns a Hill-type time series.

Live R is the oracle here for:

* the tail selection / cut point (including R's nudge when the cut point ties
  with the first value inside the tail),
* the truncation side being opposite to the tail side,
* the fitted mu/sigma/deviance of fitTail,
* the fitTailAll matrix, column names and tsp(start, end, frequency),
* the deviance R's own d*/p* kernels assign to our coefficients.

Note: live R 4.6.0's gamlssML can fail to converge on truncated families for
*some* data (it runs mu -> 0 where the truncation normalisation collapses).
Both sides are fed the identical dataset here, so R converges and the numbers
are a real golden; the R-side failure is recorded separately in the work log.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.gamlss_tr import (
    TailAllFit,
    TailFit,
    _match_tail_side,
    _resolve,
    _tail_selection,
    fitTail,
    fitTailAll,
    trun,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
srcs <- c('trun.d-30-03-24.R','trun.p-30-03-24.R','trun.q-27-12-12.R',
          'trun.r-28-12-12.R','gen.trun-30-04-24.R','trun.R','FitTrunTail.R')
for (f in srcs) source(file.path('R/gamlss.tr/R', f))
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
y1 <- as.numeric(d$y1); y2 <- as.numeric(d$y2)

tailInfo <- function(y, family, percentage, howmany, type) {
  ly <- length(y)
  hm <- if (is.null(howmany)) floor(ly*(percentage/100)) else howmany
  ys <- y[order(y)]
  Y <- if (type=="right") tail(ys, hm) else head(ys, hm)
  k <- ly - hm
  mY <- ys[k]; mYp1 <- ys[k+1]
  mYadj <- mY
  if (mY == mYp1) mYadj <- mY - sum(diff(ys[(k-5):(k+5)]))/100
  FAM <- as.gamlss.family(family)
  list(howmany=as.numeric(hm), mY=as.numeric(mY), mYp1=as.numeric(mYp1),
       mYadj=as.numeric(mYadj), Y=as.numeric(Y),
       typetrun=if (type=="right") "left" else "right",
       famName=FAM$family[1], npar=as.numeric(FAM$nopar),
       pnames=names(FAM$parameters))
}

truncDev <- function(y, par, fam, mu, sigma, typetrun) {
  dfun <- get(paste0("d",fam)); pfun <- get(paste0("p",fam))
  norm <- if (typetrun=="left") log(1 - pfun(par, mu=mu, sigma=sigma))
          else log(pfun(par, mu=mu, sigma=sigma))
  -2*sum(dfun(y, mu=mu, sigma=sigma, log=TRUE) - norm)
}

rfit <- suppressWarnings(fitTail(y1, family="WEI3", howmany=30, type="right"))
rfmu <- as.numeric(fitted(rfit,"mu")[1]); rfsg <- as.numeric(fitted(rfit,"sigma")[1])
rfdev <- as.numeric(deviance(rfit))

hmAll <- 12; startAll <- 5
T <- suppressWarnings(fitTailAll(y1, family="WEI3", howmany=hmAll,
                                 type="right", plot=FALSE, print=FALSE, trace=0))

out <- list(
  A = tailInfo(y1, "WEI3", 10, 30, "right"),
  B = tailInfo(y1, "NO", 10, 20, "left"),
  C = tailInfo(y2, "WEI3", 10, 31, "right"),
  D = tailInfo(y1, "WEI3", 10, NULL, "right"),
  dev_given = truncDev(as.numeric(d$Y), as.numeric(d$threshold), "WEI3",
                       as.numeric(d$mu), as.numeric(d$sigma), "left"),
  r_fit = list(mu=rfmu, sigma=rfsg, dev=rfdev),
  all = list(rest=as.numeric(hmAll-(2+startAll)),
             rownames=as.character(rownames(T)),
             colnames=as.character(colnames(T)),
             nrow=as.numeric(nrow(T)), ncol=as.numeric(ncol(T)),
             tsp=as.numeric(tsp(T)), matrix=as.matrix(T))
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""


def _data():
    rs = np.random.RandomState(42)
    y1 = 1.0 + rs.weibull(1.5, size=200) * 2.0
    # every value is repeated three times, so a cut point can tie with the
    # first value inside the tail (R's mY nudge branch).
    y2 = np.repeat(np.round(np.linspace(1.0, 10.0, 40), 6), 3)
    return y1, y2


def _scalar(value):
    return float(np.asarray(value, dtype=float).ravel()[0])


def _vector(value):
    return np.asarray(value, dtype=float).ravel()


def _strings(value):
    return [str(v) for v in np.asarray(value).ravel()]


@pytest.fixture(scope="module")
def r_reference():
    """Run the live-R oracle once and share its output across the module."""
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.tr" / "R" / "FitTrunTail.R").exists():
        pytest.skip("vendored gamlss.tr source not present")

    y1, y2 = _data()
    fit = fitTail(y1, family="WEI3", howmany=30, type="right")
    payload = {
        "y1": y1.tolist(),
        "y2": y2.tolist(),
        "mu": fit.parameters["mu"],
        "sigma": fit.parameters["sigma"],
        "threshold": fit.threshold,
        "Y": fit.y.tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "fitTail.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-400:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _structure(y, family, percentage, howmany, side):
    """The deterministic part of fitTail, exposed for structure parity."""
    base = _resolve(family)
    selected, threshold, count = _tail_selection(y, percentage, howmany, side)
    return {
        "howmany": count,
        "mY": threshold,
        "Y": selected,
        "typetrun": "left" if side == "right" else "right",
        "famName": base.name,
        "npar": len(base.parameters),
        "pnames": list(base.parameters),
    }


# ---------------------------------------------------------------------------
# Live-R structure parity
# ---------------------------------------------------------------------------


def test_tail_selection_matches_live_r(r_reference):
    y1, y2 = _data()
    cases = {
        "A": (y1, "WEI3", 10, 30, "right"),
        "B": (y1, "NO", 10, 20, "left"),
        "C": (y2, "WEI3", 10, 31, "right"),
        "D": (y1, "WEI3", 10, None, "right"),
    }
    for key, (data, family, percentage, howmany, side) in cases.items():
        block = r_reference[key]
        ours = _structure(data, family, percentage, howmany, side)
        assert ours["howmany"] == _scalar(block["howmany"]), key
        # R's mYadj is the value actually handed to trun(); mY is the raw one.
        assert ours["mY"] == pytest.approx(_scalar(block["mYadj"]), rel=0, abs=0), key
        np.testing.assert_allclose(
            ours["Y"], _vector(block["Y"]), rtol=0, atol=0, err_msg=key
        )
        assert ours["typetrun"] == _strings(block["typetrun"])[0], key
        assert ours["famName"] == _strings(block["famName"])[0], key
        assert ours["npar"] == _scalar(block["npar"]), key
        assert ours["pnames"] == _strings(block["pnames"]), key


def test_the_tied_cut_point_is_nudged_exactly_like_r(r_reference):
    """Case C: mY == mYp1, so R subtracts 1/100 of the local span."""
    block = r_reference["C"]
    assert _scalar(block["mY"]) == _scalar(block["mYp1"])
    assert _scalar(block["mYadj"]) < _scalar(block["mY"])
    _y1, y2 = _data()
    ours = _structure(y2, "WEI3", 10, 31, "right")
    assert ours["mY"] == pytest.approx(_scalar(block["mYadj"]), rel=0, abs=0)


# ---------------------------------------------------------------------------
# Live-R numerical parity
# ---------------------------------------------------------------------------


def test_our_fitted_coefficients_match_live_r_fitTail(r_reference):
    y1, _ = _data()
    fit = fitTail(y1, family="WEI3", howmany=30, type="right")
    r = r_reference["r_fit"]
    assert fit.parameters["mu"] == pytest.approx(_scalar(r["mu"]), rel=1e-5)
    assert fit.parameters["sigma"] == pytest.approx(_scalar(r["sigma"]), rel=1e-5)
    assert fit.deviance == pytest.approx(_scalar(r["dev"]), rel=1e-8)


def test_live_r_density_evaluates_our_coefficients_to_the_same_deviance(r_reference):
    """Two-step parity check: R's own dWEI3/pWEI3 score our solution."""
    y1, _ = _data()
    fit = fitTail(y1, family="WEI3", howmany=30, type="right")
    assert fit.deviance == pytest.approx(
        _scalar(r_reference["dev_given"]), rel=1e-10, abs=1e-10
    )


def test_fit_tail_all_matrix_matches_live_r(r_reference):
    y1, _ = _data()
    ours = fitTailAll(y1, family="WEI3", howmany=12, type="right", trace=0)
    block = r_reference["all"]
    ref = np.asarray(block["matrix"], dtype=float)
    assert ours.matrix.shape == ref.shape
    np.testing.assert_allclose(ours.matrix, ref, rtol=1e-4, atol=1e-4)
    assert ours.colnames == tuple(_strings(block["colnames"]))
    assert ours.npar == _scalar(block["ncol"])
    assert (ours.start, ours.end, ours.frequency) == tuple(
        int(v) for v in _vector(block["tsp"])
    )


# ---------------------------------------------------------------------------
# Pure-Python contracts (no R needed)
# ---------------------------------------------------------------------------


def test_fit_tail_returns_the_r_shaped_object():
    y1, _ = _data()
    fit = fitTail(y1, family="WEI3", howmany=30, type="right")
    assert isinstance(fit, TailFit)
    assert fit.family_name == "WEI3"
    assert fit.parameter_names == ("mu", "sigma")
    assert fit.howmany == 30
    assert fit.n == 30
    assert fit.type == "right"
    assert fit.truncation_type == "left"
    assert fit.y.shape == (30,)
    assert np.all(np.diff(fit.y) >= 0)
    assert fit.aic == pytest.approx(fit.deviance + 2 * len(fit.parameter_names))
    for name in fit.parameter_names:
        np.testing.assert_allclose(
            fit.fitted(name), np.full(30, fit.parameters[name]), rtol=0, atol=0
        )


def test_right_and_left_tails_use_opposite_truncation_sides():
    y1, _ = _data()
    right = fitTail(y1, family="WEI3", howmany=30, type="right")
    left = fitTail(y1, family="WEI3", howmany=30, type="left")
    assert right.truncation_type == "left"
    assert left.truncation_type == "right"
    assert float(right.y.min()) > float(left.y.max())


def test_percentage_default_and_explicit_howmany():
    y1, _ = _data()
    by_percentage = fitTail(y1, family="WEI3", type="right")
    assert by_percentage.howmany == 20  # floor(200 * 10 / 100)
    by_howmany = fitTail(y1, family="WEI3", howmany=30, type="right")
    assert by_howmany.howmany == 30


def test_partial_type_matching_like_match_arg():
    assert _match_tail_side("r") == "right"
    assert _match_tail_side("l") == "left"
    y1, _ = _data()
    assert fitTail(y1, family="WEI3", howmany=30, type="r").type == "right"
    with pytest.raises(ValueError, match="type must be one of"):
        _match_tail_side("middle")


def test_too_few_observations_is_rejected():
    y1, _ = _data()
    with pytest.raises(ValueError, match="too few observations selected"):
        fitTail(y1, family="WEI3", howmany=5, type="right")
    with pytest.raises(ValueError, match="too few observations selected"):
        fitTail(y1, family="WEI3", percentage=1, type="right")
    with pytest.raises(ValueError, match="too few observations selected"):
        fitTailAll(y1, family="WEI3", howmany=7, type="right")


def test_weights_length_is_checked():
    y1, _ = _data()
    with pytest.raises(ValueError, match="same length"):
        fitTail(y1, family="WEI3", howmany=30, type="right", weights=np.ones(7))


def test_the_solution_is_a_stationary_point_of_the_truncated_likelihood():
    """converged=True is only a step-size collapse; check the gradient."""
    from omnilss.ad import build_ad_family

    y1, _ = _data()
    fit = fitTail(y1, family="WEI3", howmany=30, type="right")
    ad = build_ad_family(
        source_family=trun(par=fit.threshold, family="WEI3", type=fit.truncation_type)
    )
    kwargs = {
        "y": fit.y,
        "mu": np.full(fit.n, fit.parameters["mu"]),
        "sigma": np.full(fit.n, fit.parameters["sigma"]),
    }
    for name in ("mu", "sigma"):
        score = np.asarray(ad.score_functions[name](**kwargs), dtype=float)
        # d deviance / d parameter = -2 * score; the link-scale gradient adds
        # d parameter / d eta, which is positive for both log links.
        assert abs(float(-2.0 * np.sum(score))) < 1e-4, name


def test_different_start_values_reach_the_same_maximum():
    y1, _ = _data()
    baseline = fitTail(y1, family="WEI3", howmany=30, type="right")
    for mu, sigma in ((3.0, 2.0), (6.0, 3.0), (10.0, 1.0), (5.0, 5.0)):
        other = fitTail(
            y1,
            family="WEI3",
            howmany=30,
            type="right",
            mu_start=mu,
            sigma_start=sigma,
        )
        assert other.parameters["mu"] == pytest.approx(
            baseline.parameters["mu"], rel=1e-5
        )
        assert other.parameters["sigma"] == pytest.approx(
            baseline.parameters["sigma"], rel=1e-5
        )
        assert other.deviance == pytest.approx(baseline.deviance, rel=1e-8)


def test_fit_tail_all_shape_rownames_and_tsp():
    y1, _ = _data()
    result = fitTailAll(y1, family="WEI3", howmany=12, type="right")
    assert isinstance(result, TailAllFit)
    assert result.matrix.shape == (5, 2)  # howmany - (npar + start)
    assert result.rownames == ("12", "11", "10", "9", "8")
    assert result.colnames == ("mu", "sigma")
    assert (result.start, result.end, result.frequency) == (8, 12, 1)
    assert np.all(np.isfinite(result.matrix))


def test_trace_prints_progress_without_shadowing_print(capsys):
    y1, _ = _data()
    fitTailAll(y1, family="WEI3", howmany=12, type="right", trace=1)
    out = capsys.readouterr().out
    assert ".." in out
    assert "1" in out
