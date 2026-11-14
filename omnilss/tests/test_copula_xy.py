"""gamlss.ggplots ``xy_*`` (underscore family) plot-data ports -- live-R parity.

R source: ``R/gamlss.ggplots/R/Copula_ralated.R``.  The R functions return
ggplot2 objects; a stub-ggplot2 probe (``ggplot`` / ``geom_contour`` /
``geom_point`` recorders) captures the numerical data the layers consume.

Live-R verified numeric kernels (all <= 1e-12 unless noted):

* **PIT** ``u=rank(x)/(n+1)``, ``v=rank(y)/(n+1)`` with R ties="average";
* **50% HDR** ``kde2d(x,y,n=100)`` grid + ``z50`` (col-major flatten, stable
  descending sort, cumulative mass >= 0.5) -- also checked on random data;
* **xy_median** marker (``cw``/``pc`` are the deterministic component-wise
  median; ``sp`` differs by ~3e-4 due to R optim vs scipy Nelder-Mead).

R quirks pinned: ``xy_pit_scatter`` plots ``(v,u)`` and builds HDR/median from
``(v,u)``; ``xy_scatter``/``xy_pit_scatter``/``xy_pit_density`` accept a
2-column matrix (``y=NULL``) while ``xy_hist``/``xy_pit_hist``/``xy_density``/
``xy_density_plain`` have no ``y`` default so their matrix branch is dead R
code (our ports accept it as the intended behaviour, not cross-checked in R).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.copula_xy import (
    DensityData,
    HistogramData,
    PITScores,
    ScatterData,
    hdr50_level,
    pit_scores,
    xy_density,
    xy_density_plain,
    xy_hist,
    xy_pit_density,
    xy_pit_hist,
    xy_pit_scatter,
    xy_scatter,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""
suppressPackageStartupMessages(library(jsonlite))
txt <- readLines('R/gamlss.ggplots/R/Copula_ralated.R')
txt <- gsub('ggplot2::','',txt,fixed=TRUE)
txt <- gsub('gamlss.foreach::','',txt,fixed=TRUE)
store <- new.env(); store$df <- NULL; store$contour <- NULL; store$pts <- NULL
rec_ggplot <- function(data=NULL, ...) { store$df <- as.data.frame(data); structure(list(),class='ggstub') }
rec_contour <- function(data=NULL, ..., breaks=NULL) {
  store$contour <- list(z=as.numeric(as.data.frame(data)$z), breaks=as.numeric(breaks)); structure(list(),class='ggstub') }
rec_point <- function(..., x=NULL, y=NULL) {
  if(!is.null(x) && !is.null(y)) store$pts <- c(as.numeric(x),as.numeric(y))
  structure(list(),class='ggstub') }
Ops.ggstub <- function(e1,e2) e1
stub <- function(...) structure(list(),class='ggstub')
for (f in c('aes','after_stat','stat_density_2d','geom_bin2d','geom_tile','geom_contour_filled',
            'scale_fill_gradient','scale_fill_viridis_c','labs','theme_minimal')) assign(f, stub, envir=globalenv())
ggplot <- rec_ggplot; geom_contour <- rec_contour; geom_point <- rec_point
getSVD <- function(df) list(u=matrix(0,nrow(df),2), d=c(1,1))
eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())
grab <- function(expr) { store$df<-NULL; store$contour<-NULL; store$pts<-NULL; force(expr)
  list(df=if(is.null(store$df)) NULL else as.list(as.data.frame(store$df)),
       z=if(is.null(store$contour)) NULL else store$contour$z,
       z50=if(is.null(store$contour)) NULL else store$contour$breaks,
       pts=if(length(store$pts)) as.numeric(store$pts) else NULL) }
set.seed(7)
x  <- c(1.2,2.5,3.1,4.8,5.0,7.3,8.1,9.6,10.2,12.0)
y  <- c(3.0,1.5,4.2,2.1,6.8,5.5,7.2,9.0,3.3,8.4)
xt <- c(1,1,2,2,3,3,4,4); yt <- c(4,2,5,1,3,3,6,2)
set.seed(19); xr <- rnorm(24); yr <- rnorm(24)
M <- cbind(x,y)
out <- list(
  scatter      = grab(xy_scatter(x,y,method='cw')),
  scatter_pc   = grab(xy_scatter(x,y,method='pc')),
  scatter_sp   = grab(xy_scatter(x,y,method='sp')),
  scatter_noHDR= grab(xy_scatter(x,y,method='cw',HDR=FALSE)),
  pit_scatter  = grab(xy_pit_scatter(x,y,method='cw')),
  pit_scatter_ties = grab(xy_pit_scatter(xt,yt,method='cw')),
  scatter_rand = grab(xy_scatter(xr,yr,method='cw')),
  pit_scatter_rand = grab(xy_pit_scatter(xr,yr,method='cw')),
  hist         = grab(xy_hist(x,y,bins=8)),
  pit_hist     = grab(xy_pit_hist(x,y,bins=8)),
  pit_hist_ties= grab(xy_pit_hist(xt,yt,bins=5)),
  scatter_mat = grab(xy_scatter(M,method='cw')),
  pit_scatter_mat = grab(xy_pit_scatter(M,method='cw')),
  density      = grab(xy_density(x,y)),
  density_plain= grab(xy_density_plain(x,y)),
  pit_density  = grab(xy_pit_density(x,y)),
  pit_density_ties = grab(xy_pit_density(xt,yt)),
  x=as.numeric(x), y=as.numeric(y), xt=as.numeric(xt), yt=as.numeric(yt),
  xr=as.numeric(xr), yr=as.numeric(yr)
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    src = _REPO / "R" / "gamlss.ggplots" / "R" / "Copula_ralated.R"
    if not src.exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "xy.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    # D2.42: a broken R probe must FAIL the gate, never silently skip it.
    assert "JSONSTART" in proc.stdout, f"R probe failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _arr(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).ravel()


def _assert_pit(ours: PITScores, ref: dict, key: str) -> None:
    np.testing.assert_allclose(ours.u, _arr(ref["df"]["u"]), rtol=0, atol=1e-15,
                               err_msg=f"{key}:u")
    np.testing.assert_allclose(ours.v, _arr(ref["df"]["v"]), rtol=0, atol=1e-15,
                               err_msg=f"{key}:v")


def _assert_scatter(ours: ScatterData, ref: dict, key: str,
                    sp_median: bool = False) -> None:
    np.testing.assert_allclose(ours.x, _arr(ref["df"]["x"]) if "x" in ref["df"] else _arr(ref["df"]["u"]),
                               rtol=0, atol=1e-15, err_msg=f"{key}:x")
    # For pit_scatter the plotted x is v, y is u -- handled in its own test.
    med_atol = 1e-3 if sp_median else 1e-12
    np.testing.assert_allclose(ours.median, _arr(ref["pts"]), rtol=0,
                               atol=med_atol, err_msg=f"{key}:median")
    if ref["z"]:
        assert ours.hdr is not None and ours.hdr_level is not None
        np.testing.assert_allclose(ours.hdr.z.ravel(order="F"), _arr(ref["z"]),
                                   rtol=0, atol=1e-11, err_msg=f"{key}:hdr_z")
        assert ours.hdr_level == pytest.approx(_arr(ref["z50"])[0], abs=1e-11)
    else:
        assert ours.hdr is None and ours.hdr_level is None


def test_pit_kernel_matches_live_r(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    xt, yt = _arr(r_reference["xt"]), _arr(r_reference["yt"])
    _assert_pit(pit_scores(x, y), r_reference["pit_hist"], "pit_hist")
    _assert_pit(pit_scores(xt, yt), r_reference["pit_hist_ties"], "pit_hist_ties")
    _assert_pit(pit_scores(x, y), r_reference["pit_density"], "pit_density")
    _assert_pit(pit_scores(xt, yt), r_reference["pit_density_ties"], "pit_density_ties")
    # PIT ranks are in (0,1) with the (n+1) denominator
    p = pit_scores(x, y)
    assert np.all(p.u > 0) and np.all(p.u < 1)
    assert np.all(p.v > 0) and np.all(p.v < 1)


def test_pit_ties_use_average_ranks(r_reference):
    xt, yt = _arr(r_reference["xt"]), _arr(r_reference["yt"])
    p = pit_scores(xt, yt)
    # xt = [1,1,2,2,3,3,4,4] -> average ranks [1.5,1.5,3.5,3.5,5.5,5.5,7.5,7.5]/9
    np.testing.assert_allclose(p.u, (np.array([1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5]) / 9),
                               rtol=0, atol=1e-15)
    _assert_pit(p, r_reference["pit_hist_ties"], "ties")


def test_scatter_data_matches_live_r(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    s = xy_scatter(x, y, method="cw")
    assert isinstance(s, ScatterData)
    np.testing.assert_allclose(s.x, x, rtol=0, atol=0)
    np.testing.assert_allclose(s.y, y, rtol=0, atol=0)
    _assert_scatter(s, r_reference["scatter"], "scatter")
    # pc is the same deterministic median in R (cw->pc bug)
    spc = xy_scatter(x, y, method="pc")
    np.testing.assert_allclose(spc.median, _arr(r_reference["scatter_pc"]["pts"]), rtol=0, atol=1e-12)
    # sp geometric median: optimizer-dependent, loose tolerance
    ssp = xy_scatter(x, y, method="sp")
    np.testing.assert_allclose(ssp.median, _arr(r_reference["scatter_sp"]["pts"]), rtol=0, atol=1e-3)


def test_scatter_without_hdr(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    s = xy_scatter(x, y, method="cw", HDR=False)
    assert s.hdr is None and s.hdr_level is None
    np.testing.assert_allclose(s.median, _arr(r_reference["scatter_noHDR"]["pts"]), rtol=0, atol=1e-12)


def test_scatter_random_and_matrix_input(r_reference):
    xr, yr = _arr(r_reference["xr"]), _arr(r_reference["yr"])
    s = xy_scatter(xr, yr, method="cw")
    _assert_scatter(s, r_reference["scatter_rand"], "scatter_rand")
    # 2-column matrix (y=None) is reachable for xy_scatter
    M = np.column_stack([_arr(r_reference["x"]), _arr(r_reference["y"])])
    sm = xy_scatter(M, method="cw")
    np.testing.assert_allclose(sm.x, M[:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(sm.median, _arr(r_reference["scatter_mat"]["pts"]), rtol=0, atol=1e-12)


def test_hdr50_level_matches_live_r(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    grid, z50 = hdr50_level(x, y, n=100)
    np.testing.assert_allclose(grid.z.ravel(order="F"), _arr(r_reference["scatter"]["z"]),
                               rtol=0, atol=1e-11)
    assert z50 == pytest.approx(_arr(r_reference["scatter"]["z50"])[0], abs=1e-11)
    assert grid.z.shape == (100, 100)


def test_pit_scatter_uses_swapped_vu(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    p = pit_scores(x, y)
    s = xy_pit_scatter(x, y, method="cw")
    # plotted coordinates are (x=v, y=u), NOT the df(u,v) columns
    np.testing.assert_allclose(s.x, p.v, rtol=0, atol=0)
    np.testing.assert_allclose(s.y, p.u, rtol=0, atol=0)
    np.testing.assert_allclose(s.median, _arr(r_reference["pit_scatter"]["pts"]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.hdr.z.ravel(order="F"), _arr(r_reference["pit_scatter"]["z"]),
                               rtol=0, atol=1e-11)
    assert s.hdr_level == pytest.approx(_arr(r_reference["pit_scatter"]["z50"])[0], abs=1e-11)


def test_pit_scatter_ties_random_and_matrix(r_reference):
    xt, yt = _arr(r_reference["xt"]), _arr(r_reference["yt"])
    st = xy_pit_scatter(xt, yt, method="cw")
    np.testing.assert_allclose(st.hdr.z.ravel(order="F"), _arr(r_reference["pit_scatter_ties"]["z"]),
                               rtol=0, atol=1e-11)
    assert st.hdr_level == pytest.approx(_arr(r_reference["pit_scatter_ties"]["z50"])[0], abs=1e-11)
    xr, yr = _arr(r_reference["xr"]), _arr(r_reference["yr"])
    sr = xy_pit_scatter(xr, yr, method="cw")
    np.testing.assert_allclose(sr.hdr.z.ravel(order="F"), _arr(r_reference["pit_scatter_rand"]["z"]),
                               rtol=0, atol=1e-11)
    assert sr.hdr_level == pytest.approx(_arr(r_reference["pit_scatter_rand"]["z50"])[0], abs=1e-11)
    M = np.column_stack([_arr(r_reference["x"]), _arr(r_reference["y"])])
    sm = xy_pit_scatter(M, method="cw")
    assert sm.hdr_level == pytest.approx(_arr(r_reference["pit_scatter_mat"]["z50"])[0], abs=1e-11)


def test_hist_data_matches_live_r(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    xt, yt = _arr(r_reference["xt"]), _arr(r_reference["yt"])
    h = xy_hist(x, y, bins=8)
    assert isinstance(h, HistogramData) and h.bins == 8
    np.testing.assert_allclose(h.x, _arr(r_reference["hist"]["df"]["x"]), rtol=0, atol=0)
    np.testing.assert_allclose(h.y, _arr(r_reference["hist"]["df"]["y"]), rtol=0, atol=0)
    ph = xy_pit_hist(x, y, bins=8)
    np.testing.assert_allclose(ph.x, _arr(r_reference["pit_hist"]["df"]["u"]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(ph.y, _arr(r_reference["pit_hist"]["df"]["v"]), rtol=0, atol=1e-15)
    pht = xy_pit_hist(xt, yt, bins=5)
    np.testing.assert_allclose(pht.x, _arr(r_reference["pit_hist_ties"]["df"]["u"]), rtol=0, atol=1e-15)
    assert pht.bins == 5


def test_density_data_matches_live_r(r_reference):
    x, y = _arr(r_reference["x"]), _arr(r_reference["y"])
    xt, yt = _arr(r_reference["xt"]), _arr(r_reference["yt"])
    for ours, key in [(xy_density(x, y), "density"),
                      (xy_density_plain(x, y), "density_plain")]:
        assert isinstance(ours, DensityData)
        np.testing.assert_allclose(ours.x, _arr(r_reference[key]["df"]["x"]), rtol=0, atol=0)
        np.testing.assert_allclose(ours.y, _arr(r_reference[key]["df"]["y"]), rtol=0, atol=0)
    pd = xy_pit_density(x, y)
    np.testing.assert_allclose(pd.x, _arr(r_reference["pit_density"]["df"]["u"]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(pd.y, _arr(r_reference["pit_density"]["df"]["v"]), rtol=0, atol=1e-15)
    pdt = xy_pit_density(xt, yt)
    np.testing.assert_allclose(pdt.x, _arr(r_reference["pit_density_ties"]["df"]["u"]), rtol=0, atol=1e-15)


def test_two_column_matrix_intended_behaviour():
    """Ports accept y=None + 2-col array even where R's matrix branch is dead."""
    M = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 6.0]])
    for fn in (xy_hist, xy_pit_hist, xy_density, xy_density_plain, xy_pit_density):
        out = fn(M)
        np.testing.assert_allclose(out.x, M[:, 0] if fn in (xy_hist, xy_density, xy_density_plain)
                                   else pit_scores(M[:, 0], M[:, 1]).u, rtol=0, atol=1e-15)
