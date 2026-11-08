"""R gamlss.ggplots single-variable diagnostic helpers -- live-R parity.

R sources: ``R/gamlss.ggplots/R/{symmetry_plot.R, y_ecdf.R, y_hist.R,
y_acf.R, resid_index.R}``.  ggplot2 is not installed, so the oracle below
sources the vendored R files after stripping the ``ggplot2::`` prefix and
installing stubs for the plotting calls; the *real R function bodies* run and
``ggplot()`` records the data frame they would have plotted.

Pinned R quirks: median-split with ``floor/ceiling((n+1)/2)`` in
``y_symmetry``; ``median +/- value * mad`` outlier flags in ``y_dots`` /
``resid_index`` (NAs dropped by observation position); R's biased ACF with
``lag.max = floor(10*log10(n))``, Durbin-Levinson PACF and the fixed 95%
line; ``format(digits=4)`` rounding in ``y_epdf``; ``rank/(n+1)`` in
``y_epit``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import omnilss.ggplots_diag as g

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""# ggplot2 is not installed: strip the prefix and stub the plotting calls so
# the REAL vendored function bodies execute; ggplot() records its data frame.
library(jsonlite)
store <- new.env(); store$data <- NULL
record <- function(data=NULL, mapping=NULL, ...) { store$data <- data; structure(list(), class="ggstub") }
Ops.ggstub <- function(e1, e2) e1
stub <- function(...) NULL
for (f in c("aes","geom_point","geom_abline","ggtitle","geom_linerange","geom_text",
            "geom_vline","geom_hline","geom_histogram","geom_density","theme",
            "xlab","ylab","xlim","theme_minimal","annotate","element_blank",
            "element_line","geom_errorbar","geom_ribbon","stat_ecdf","after_stat",
            "geom_line","geom_smooth","scale_x_continuous","geom_bar","coord_flip")) {
  assign(f, stub, envir=globalenv())
}
assign("ggplot", record, envir=globalenv())
get_residuals <- function(obj) obj$residuals
is.gamlss <- function(obj) TRUE
load_src <- function(path) {
  txt <- gsub('ggplot2::', '', readLines(path), fixed=TRUE)
  eval(parse(text=paste(txt, collapse='\n')), envir=globalenv()); invisible(NULL)
}
load_src('R/gamlss.ggplots/R/symmetry_plot.R')
load_src('R/gamlss.ggplots/R/y_ecdf.R')
load_src('R/gamlss.ggplots/R/y_hist.R')
load_src('R/gamlss.ggplots/R/y_acf.R')
load_src('R/gamlss.ggplots/R/resid_index.R')

y  <- c(1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0)
y2 <- c(1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 50.0, -40.0)
res <- c(0.1, -0.3, 0.5, 2.7, -3.1, 0.2, 0.4, -0.6, 1.1, -0.2)
yna <- c(1.0, NA, 2.0, 3.0, 4.0, NA, 5.0, 6.0, 7.0, 8.0)
grab <- function(expr) { store$data <- NULL; force(expr); store$data }
out <- list(
  epit_y = as.numeric(y_epit(y)),
  epit_ties = as.numeric(y_epit(y2)),
  epit_na = as.numeric(y_epit(yna)),
  epdf_y = y_epdf(y, plot=FALSE),
  epdf_ties = y_epdf(y2, plot=FALSE),
  sym_y = grab(y_symmetry(y)),
  sym_ties = grab(y_symmetry(y2)),
  residsym = grab(resid_symmetry(list(residuals=res))),
  acf_y = grab(y_acf(y)),
  pacf_y = grab(y_pacf(y)),
  acf_long = grab(y_acf(1:30)),
  dots_y = as.numeric(y_dots(y, plot=FALSE)),
  dots_out = as.numeric(y_dots(y2, plot=FALSE)),
  dots_v1 = as.numeric(y_dots(y2, value=1, plot=FALSE)),
  residx_y = grab(resid_index(resid=res, plot=FALSE)),
  residx_out = grab(resid_index(resid=res, value=1, plot=FALSE))
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_Y = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0])
_Y2 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 50.0, -40.0])
_RES = np.array([0.1, -0.3, 0.5, 2.7, -3.1, 0.2, 0.4, -0.6, 1.1, -0.2])


def _vector(v):
    return np.asarray(v, dtype=float).ravel()


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.ggplots" / "R" / "symmetry_plot.R").exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "ggplots.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-400:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def test_y_epit_matches_live_r(r_reference):
    for key, values in (("epit_y", _Y), ("epit_ties", _Y2)):
        np.testing.assert_allclose(g.y_epit(values), _vector(r_reference[key]),
                                   rtol=0, atol=0, err_msg=key)


def test_y_epdf_matches_live_r(r_reference):
    for key, values in (("epdf_y", _Y), ("epdf_ties", _Y2)):
        table = g.y_epdf(values, plot=False)
        ref = r_reference[key]
        np.testing.assert_allclose(table.x, [q["x"] for q in ref], rtol=0, atol=0)
        np.testing.assert_allclose(table.freq, [q["freq"] for q in ref], rtol=0, atol=0)


def test_symmetry_data_matches_live_r(r_reference):
    for key, values in (("sym_y", _Y), ("sym_ties", _Y2)):
        data = g.y_symmetry(values)
        ref = r_reference[key]
        np.testing.assert_allclose(data.lower_half,
                                   [q["lower_half"] for q in ref], rtol=0, atol=0)
        np.testing.assert_allclose(data.upper_half,
                                   [q["upper_half"] for q in ref], rtol=0, atol=0)


def test_resid_symmetry_matches_live_r(r_reference):
    data = g.resid_symmetry(type("M", (), {"residuals": _RES})())
    ref = r_reference["residsym"]
    np.testing.assert_allclose(data.lower_half, [q["lower_half"] for q in ref],
                               rtol=0, atol=0)
    np.testing.assert_allclose(data.upper_half, [q["upper_half"] for q in ref],
                               rtol=0, atol=0)


def test_acf_and_pacf_match_live_r(r_reference):
    for key, fun in (("acf_y", g.y_acf), ("pacf_y", g.y_pacf)):
        data = fun(_Y)
        ref = r_reference[key]
        np.testing.assert_allclose(data.lag, [q["lag"] for q in ref], rtol=0, atol=0)
        np.testing.assert_allclose(data.acf, [q["acf"] for q in ref],
                                   rtol=1e-10, atol=1e-12)
    long = r_reference["acf_long"]
    np.testing.assert_allclose(g.y_acf(np.arange(1, 31)).lag,
                               [q["lag"] for q in long], rtol=0, atol=0)
    np.testing.assert_allclose(g.y_acf(np.arange(1, 31)).acf,
                               [q["acf"] for q in long], rtol=1e-10, atol=1e-12)
    assert g.y_acf(_Y).ciline == pytest.approx(
        -0.6197950323045616, rel=1e-12
    )


def test_y_dots_matches_live_r(r_reference):
    np.testing.assert_allclose(g.y_dots(_Y, plot=False),
                               _vector(r_reference["dots_y"]), rtol=0, atol=0)
    np.testing.assert_allclose(g.y_dots(_Y2, plot=False),
                               _vector(r_reference["dots_out"]), rtol=0, atol=0)
    np.testing.assert_allclose(g.y_dots(_Y2, value=1, plot=False),
                               _vector(r_reference["dots_v1"]), rtol=0, atol=0)


def test_resid_index_matches_live_r(r_reference):
    for key, value in (("residx_y", 2.0), ("residx_out", 1.0)):
        data = g.resid_index(resid=_RES, value=value, plot=False)
        ref = r_reference[key]
        np.testing.assert_allclose(data.obs, [q["obs"] for q in ref], rtol=0, atol=0)
        np.testing.assert_allclose(data.sdres, [q["sdres"] for q in ref],
                                   rtol=0, atol=0)
        assert list(data.color) == [q["color"] for q in ref]
        assert data.threshold == value
