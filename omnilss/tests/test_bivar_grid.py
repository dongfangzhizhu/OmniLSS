"""R gamlss.ggplots bivariate ECDF / bivar.fun grid data -- live-R parity.

R source: ``R/gamlss.ggplots/R/Copula_ralated_not_ggplot2.R`` (base-R surface
plots) plus ``ECDF_2D`` from ``Copula_ralated.R``.  ggplot2 is not installed, so
the oracle sources ``Copula_ralated.R`` after stripping ``ggplot2::`` and
stubbing ggplot calls (only ``ECDF_2D`` / ``median_*`` are needed), then stubs
the base graphics recorders (``persp`` / ``image`` / ``contour``) to capture the
exact ``x`` / ``y`` / ``z`` grid the REAL R function body would have drawn.

Pinned R numeric semantics:
* ``xy.ECDF.*``: 40x40 grid ``z = outer(xg, yg, Vectorize(ECDF_2D))`` with
  ``ECDF_2D = mean(x<=vx & y<=vy)``; all three variants compute the identical
  grid, ``xy.ECDF.persp`` hard-codes its axis labels (dead args).
* ``bivar.fun.*``: ``n`` x ``n`` grid from ``expand.grid`` (x varies fastest)
  and column-major ``matrix(...)`` -> ``Z[i, j] = f(gx[i], gy[j])``; ``n=100``
  default; ``as.numeric`` flattens column-major (``ravel(order="F")``).

13 groups verified to exact equality (0 absolute error), including ties,
n=1 / n=100 defaults and asymmetric xlim/ylim.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.bivar_grid import (
    BivarGrid,
    bivar_fun_contour,
    bivar_fun_image,
    bivar_fun_persp,
    xy_ECDF_contour,
    xy_ECDF_image,
    xy_ECDF_persp,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""# ggplot2 not installed: strip prefix + stub ggplot calls so Copula_ralated.R
# can be sourced for ECDF_2D.  Then stub base graphics to record the grid data.
suppressPackageStartupMessages(library(jsonlite))
txt <- gsub('ggplot2::', '', readLines('R/gamlss.ggplots/R/Copula_ralated.R'), fixed=TRUE)
stub <- function(...) NULL
for (f in c("aes","geom_point","geom_abline","ggtitle","geom_linerange","geom_text",
            "geom_vline","geom_hline","geom_histogram","geom_density","theme",
            "xlab","ylab","xlim","theme_minimal","annotate","element_blank",
            "geom_contour_filled","geom_contour","labs","coord_fixed","guides",
            "scale_size","scale_fill_gradient2","geom_tile","ggplot","aes_string")) {
  assign(f, stub, envir=globalenv())
}
getSVD <- function(df) list(u=matrix(0,nrow(df),2), d=c(1,1))
Ops.ggstub <- function(e1, e2) e1   # so `ggplot(...) + geom_*()` stays harmless
eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())

store <- new.env(); store$last <- NULL
rec <- function(x, y, z, ...) {
  store$last <- list(x=as.numeric(x), y=as.numeric(y), z=as.numeric(z)); invisible(NULL)
}
persp <- rec; image <- rec; contour <- rec; points <- function(...) NULL
txt2 <- gsub('MASS::', '', readLines('R/gamlss.ggplots/R/Copula_ralated_not_ggplot2.R'), fixed=TRUE)
eval(parse(text=paste(txt2, collapse='\n')), envir=globalenv())
grab <- function(expr) { store$last <- NULL; force(expr); store$last }

set.seed(7)
x  <- c(1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0)
y  <- c(3.0, 1.5, 4.2, 2.1, 6.8, 5.5, 7.2, 9.0, 3.3, 8.4)
x2 <- c(1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 50.0, -40.0)  # ties
y1 <- c(2.0, 4.0, 1.0, 5.0, 3.0, 6.0, 2.5, 4.5)
x1 <- c(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5)
yt <- c(4, 2, 5, 1, 3, 3, 6, 2)

out <- list(
  ecdf_persp  = grab(xy.ECDF.persp(x, y)),
  ecdf_image  = grab(xy.ECDF.image(x, y)),
  ecdf_contour= grab(xy.ECDF.contour(x, y)),
  bfun_contour= grab(bivar.fun.contour(function(a,b) a^2+b^2, xlim=c(-2,2), ylim=c(-1,1), n=5)),
  bfun_image  = grab(bivar.fun.image(function(a,b) sin(a)+cos(b), xlim=c(0,3), ylim=c(0,2), n=4)),
  bfun_persp  = grab(bivar.fun.persp(function(a,b) a*b, xlim=c(-1,1), ylim=c(-2,2), n=6)),
  ecdf_ties   = grab(xy.ECDF.persp(x2, c(3.0,1.5,4.2,2.1,6.8,5.5,7.2,9.0))),
  ecdf_n8     = grab(xy.ECDF.image(x1, y1)),
  ecdf_cont   = grab(xy.ECDF.contour(c(1,1,2,2,3,3,4,4), yt)),
  bfun_default= grab(bivar.fun.contour(function(a,b) a+b)),
  bfun_asym   = grab(bivar.fun.image(function(a,b) exp(-a^2-b^2), xlim=c(-1,2), ylim=c(0,3), n=7)),
  bfun_n1     = grab(bivar.fun.persp(function(a,b) a-b, xlim=c(0,1), ylim=c(0,1), n=1)),
  bfun_n2     = grab(bivar.fun.persp(function(a,b) a*0+b*0+7, xlim=c(-5,5), ylim=c(-2,2), n=3))
)
# The underscore-named R export xy_ECDF_contour (Copula_ralated.R:521) is a
# SEPARATE symbol from xy.ECDF.contour (not_ggplot2.R).  It builds the same
# 40x40 ECDF grid via expand.grid + mapply(ECDF_2D).  Capture its data.frame so
# we can prove our xy_ECDF_contour covers its numeric kernel too (not a
# coincidental name match).  Re-register ggplot as a recorder of its data arg.
gg_rec <- function(data=NULL, ...) { store$df <- data; structure(list(), class="ggstub") }
assign("ggplot", gg_rec, envir=globalenv())
store$df <- NULL
xy_ECDF_contour(x, y)
gd <- store$df
out$underscore_grid <- list(x=as.numeric(unique(gd$x)), y=as.numeric(unique(gd$y)),
                            z=as.numeric(gd$Fn))
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_X = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0])
_Y = np.array([3.0, 1.5, 4.2, 2.1, 6.8, 5.5, 7.2, 9.0, 3.3, 8.4])
_X2 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 50.0, -40.0])
_X1 = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
_Y1 = np.array([2.0, 4.0, 1.0, 5.0, 3.0, 6.0, 2.5, 4.5])
_XT = np.array([1, 1, 2, 2, 3, 3, 4, 4])
_YT = np.array([4, 2, 5, 1, 3, 3, 6, 2])


def _as_grid(ref: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray(ref["x"], dtype=float),
            np.asarray(ref["y"], dtype=float),
            np.asarray(ref["z"], dtype=float))


def _assert_grid(ours: BivarGrid, ref: dict, key: str) -> None:
    rx, ry, rz = _as_grid(ref)
    np.testing.assert_allclose(ours.x, rx, rtol=0, atol=0, err_msg=key + ":x")
    np.testing.assert_allclose(ours.y, ry, rtol=0, atol=0, err_msg=key + ":y")
    # R as.numeric(z) flattens column-major (row=x, col=y)
    np.testing.assert_allclose(ours.z.ravel(order="F"), rz, rtol=0, atol=0,
                               err_msg=key + ":z")
    assert ours.z.shape == (len(rx), len(ry)), key


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.ggplots" / "R" /
            "Copula_ralated_not_ggplot2.R").exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "bivar.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-400:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def test_ecdf_grids_match_live_r(r_reference):
    cases = {
        "ecdf_persp": xy_ECDF_persp(_X, _Y),
        "ecdf_image": xy_ECDF_image(_X, _Y),
        "ecdf_contour": xy_ECDF_contour(_X, _Y),
        "ecdf_ties": xy_ECDF_persp(_X2, _Y[:8]),
        "ecdf_n8": xy_ECDF_image(_X1, _Y1),
        "ecdf_cont": xy_ECDF_contour(_XT, _YT),
    }
    for key, ours in cases.items():
        _assert_grid(ours, r_reference[key], key)


def test_ecdf_variants_compute_identical_grid(r_reference):
    # the three R variants differ only in the drawing call
    a, b, c = (xy_ECDF_persp(_X, _Y), xy_ECDF_image(_X, _Y),
               xy_ECDF_contour(_X, _Y))
    np.testing.assert_array_equal(a.z, b.z)
    np.testing.assert_array_equal(a.z, c.z)
    np.testing.assert_array_equal(a.x, b.x)


def test_bivar_fun_grids_match_live_r(r_reference):
    cases = {
        "bfun_contour": bivar_fun_contour(lambda a, b: a**2 + b**2,
                                          xlim=(-2, 2), ylim=(-1, 1), n=5),
        "bfun_image": bivar_fun_image(lambda a, b: np.sin(a) + np.cos(b),
                                      xlim=(0, 3), ylim=(0, 2), n=4),
        "bfun_persp": bivar_fun_persp(lambda a, b: a * b,
                                      xlim=(-1, 1), ylim=(-2, 2), n=6),
        "bfun_default": bivar_fun_contour(lambda a, b: a + b),
        "bfun_asym": bivar_fun_image(lambda a, b: np.exp(-a**2 - b**2),
                                     xlim=(-1, 2), ylim=(0, 3), n=7),
        "bfun_n1": bivar_fun_persp(lambda a, b: a - b,
                                   xlim=(0, 1), ylim=(0, 1), n=1),
        "bfun_n2": bivar_fun_persp(lambda a, b: a * 0 + b * 0 + 7,
                                   xlim=(-5, 5), ylim=(-2, 2), n=3),
    }
    for key, ours in cases.items():
        _assert_grid(ours, r_reference[key], key)


def test_bivar_fun_variants_compute_identical_grid():
    f = lambda a, b: a**2 + b**2
    kw = dict(xlim=(-2, 2), ylim=(-1, 1), n=5)
    a, b, c = (bivar_fun_contour(f, **kw), bivar_fun_image(f, **kw),
               bivar_fun_persp(f, **kw))
    np.testing.assert_array_equal(a.z, b.z)
    np.testing.assert_array_equal(a.z, c.z)


def test_grid_shape_and_layout():
    # Z[i, j] = f(gx[i], gy[j]) (row=x, col=y)
    g = bivar_fun_contour(lambda a, b: a * 10 + b, xlim=(0, 2), ylim=(0, 3), n=3)
    assert g.z.shape == (3, 3)
    np.testing.assert_allclose(g.z[:, 0], [0, 10, 20])   # y=0
    np.testing.assert_allclose(g.z[0, :], [0, 1.5, 3])  # x=0
    assert isinstance(g, BivarGrid)


def test_ecdf_grid_is_monotone_and_bounded():
    g = xy_ECDF_persp(_X, _Y)
    assert g.z.shape == (40, 40)
    assert g.z.min() >= 0.0 and g.z.max() <= 1.0
    # non-decreasing along both axes
    assert np.all(np.diff(g.z, axis=0) >= -1e-15)
    assert np.all(np.diff(g.z, axis=1) >= -1e-15)
    assert g.z[-1, -1] == 1.0


def test_underscore_xy_ECDF_contour_is_same_numeric_kernel(r_reference):
    # R has BOTH xy.ECDF.contour (point-named, not_ggplot2.R) and
    # xy_ECDF_contour (underscore, Copula_ralated.R:521) as exports.  The
    # latter's grid must equal our xy_ECDF_contour -- proves real coverage of
    # that symbol, not just an accidental name collision.
    ref = r_reference["underscore_grid"]
    _assert_grid(xy_ECDF_contour(_X, _Y), ref, "underscore_grid")
