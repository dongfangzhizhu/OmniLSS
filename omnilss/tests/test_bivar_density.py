"""R gamlss.ggplots bivariate KDE / ECDF-hist grid data -- live-R parity.

R sources: ``R/gamlss.ggplots/R/Copula_ralated_not_ggplot2.R`` (``xy.density.*``)
and ``Copula_ralated.R`` (``xy_hist_kde2d``, ``xy_ECDF_hist``).  MASS is
installed, so ``MASS::kde2d`` runs for real; base-graphics recorders
(``persp``/``image``/``contour``) capture the grid, and a ``ggplot`` recorder
captures the long ``expand.grid`` data frames of the ggplot2 variants.

Numeric kernels (live-R verified, all <= 1e-15):
* ``MASS::kde2d`` -- ``bandwidth_nrd`` = ``4*1.06*min(sqrt(var(x)),
  IQR/1.34)*n^(-1/5)`` with R sample variance (ddof=1) and type-7 quantiles,
  then ``h/4``; grid ``z = dnorm(ax) %*% t(dnorm(ay)) / (N*hx*hy)``.
* ``xy_ECDF_hist`` -- the same 40x40 ECDF grid as ``xy_ECDF_contour``
  (``expand.grid`` + ``mapply(ECDF_2D)``), a distinct R export.
* ``as.vector`` / ``as.numeric`` flatten an R matrix column-major
  (``ravel(order="F")``); ``expand.grid(x, y)`` varies x fastest.

R quirk recorded (not reproduced as an error): ``xy.density.*`` declares
``function(x, y, ...)`` with no default for ``y``, so the ``is.null(y)``
2-column-matrix branch is unreachable -- passing only a matrix raises
``argument "y" is missing``.  Our ports accept ``y=None`` with a 2-column array
as the intended behaviour.
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
    bandwidth_nrd,
    kde2d,
    xy_ECDF_hist,
    xy_density_contour,
    xy_density_image,
    xy_density_persp,
    xy_hist_kde2d,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""# MASS is installed: keep MASS:: prefix so the real kde2d runs.
suppressPackageStartupMessages(library(jsonlite))
store <- new.env(); store$last <- NULL; store$df <- NULL
# ---- Copula_ralated.R: strip ggplot2::, stub glyphs, record ggplot(data=) ----
txt <- gsub('ggplot2::', '', readLines('R/gamlss.ggplots/R/Copula_ralated.R'), fixed=TRUE)
stub <- function(...) NULL
for (f in c("aes","geom_point","geom_abline","ggtitle","geom_linerange","geom_text",
            "geom_vline","geom_hline","geom_histogram","geom_density","theme",
            "xlab","ylab","xlim","theme_minimal","annotate","element_blank",
            "geom_contour_filled","geom_contour","labs","coord_fixed","guides",
            "scale_size","scale_fill_gradient2","geom_tile","geom_bin2d",
            "scale_fill_gradient","scale_fill_viridis_c","stat_density_2d",
            "aes_string")) assign(f, stub, envir=globalenv())
rec_ggplot <- function(data=NULL, ...) { store$df <- data; structure(list(), class="ggstub") }
Ops.ggstub <- function(e1, e2) e1
ggplot <- rec_ggplot
getSVD <- function(df) list(u=matrix(0,nrow(df),2), d=c(1,1))
eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())
# ---- not_ggplot2.R: record base graphics grid ----
rec <- function(x, y, z, ...) { store$last <- list(x=as.numeric(x), y=as.numeric(y), z=as.numeric(z)); invisible(NULL) }
persp <- rec; image <- rec; contour <- rec; points <- function(...) NULL
txt2 <- readLines('R/gamlss.ggplots/R/Copula_ralated_not_ggplot2.R')
eval(parse(text=paste(txt2, collapse='\n')), envir=globalenv())
grab <- function(expr) { store$last <- NULL; force(expr); store$last }
grabdf <- function(expr) { store$df <- NULL; force(expr); as.data.frame(store$df) }

set.seed(7)
x  <- c(1.2,2.5,3.1,4.8,5.0,7.3,8.1,9.6,10.2,12.0)
y  <- c(3.0,1.5,4.2,2.1,6.8,5.5,7.2,9.0,3.3,8.4)
x2 <- c(1.0,1.0,2.0,2.0,3.0,3.0,50.0,-40.0)
y2 <- c(4,2,5,1,3,3,6,2)
set.seed(19)
xr <- rnorm(30); yr <- rnorm(30)
out <- list(
  dens_contour = grab(xy.density.contour(x,y,bins=8)),
  dens_image   = grab(xy.density.image(x,y,bins=8)),
  dens_persp   = grab(xy.density.persp(x,y,bins=6)),
  dens_ties    = grab(xy.density.contour(x2,y2,bins=5)),
  hist_kde     = grabdf(xy_hist_kde2d(x,y,bins=8)),
  ecdf_hist    = grabdf(xy_ECDF_hist(x,y)),
  dens_n25     = grab(xy.density.contour(xr,yr,bins=25)),
  dens_n3      = grab(xy.density.image(xr,yr,bins=3)),
  dens_default = grab(xy.density.persp(xr,yr)),       # bins=25 default
  hist_kde_n12 = grabdf(xy_hist_kde2d(xr,yr,bins=12)),
  ecdf_ties    = grabdf(xy_ECDF_hist(c(1,1,2,2,3,3), c(3,1,2,2,1,3))),
  bw_x         = as.numeric(MASS::bandwidth.nrd(xr)),
  bw_y         = as.numeric(MASS::bandwidth.nrd(yr)),
  rx = as.numeric(xr), ry = as.numeric(yr)
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_X = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0])
_Y = np.array([3.0, 1.5, 4.2, 2.1, 6.8, 5.5, 7.2, 9.0, 3.3, 8.4])
_X2 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 50.0, -40.0])
_Y2 = np.array([4, 2, 5, 1, 3, 3, 6, 2])


def _assert_grid(ours: BivarGrid, ref: dict, key: str, atol: float = 1e-12) -> None:
    rx, ry, rz = (np.asarray(ref["x"], float), np.asarray(ref["y"], float),
                  np.asarray(ref["z"], float))
    np.testing.assert_allclose(ours.x, rx, rtol=0, atol=atol, err_msg=key + ":x")
    np.testing.assert_allclose(ours.y, ry, rtol=0, atol=atol, err_msg=key + ":y")
    np.testing.assert_allclose(ours.z.ravel(order="F"), rz, rtol=0, atol=atol,
                               err_msg=key + ":z")
    assert ours.z.shape == (len(rx), len(ry)), key


def _long_table(ours: BivarGrid, zcol: str = "z") -> tuple:
    # R expand.grid(x=gx, y=gy): x varies fastest; z = as.vector(matrix)
    ex = np.tile(ours.x, len(ours.y))
    ey = np.repeat(ours.y, len(ours.x))
    return ex, ey, ours.z.ravel(order="F")


def _assert_long(ours: BivarGrid, rows: list, key: str, zcol: str,
                 atol: float = 1e-12) -> None:
    ex, ey, ez = _long_table(ours)
    rx = np.array([r["x"] for r in rows], float)
    ry = np.array([r["y"] for r in rows], float)
    rz = np.array([r[zcol] for r in rows], float)
    np.testing.assert_allclose(ex, rx, rtol=0, atol=atol, err_msg=key + ":x")
    np.testing.assert_allclose(ey, ry, rtol=0, atol=atol, err_msg=key + ":y")
    np.testing.assert_allclose(ez, rz, rtol=0, atol=atol, err_msg=key + ":z")


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.ggplots" / "R" /
            "Copula_ralated_not_ggplot2.R").exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "density.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-500:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def test_bandwidth_nrd_matches_live_r(r_reference):
    rx = np.asarray(r_reference["rx"], float)
    ry = np.asarray(r_reference["ry"], float)
    # jsonlite wraps scalars in a single-element array (auto_unbox=FALSE)
    bw_x = np.asarray(r_reference["bw_x"], float).ravel()[0]
    bw_y = np.asarray(r_reference["bw_y"], float).ravel()[0]
    assert bandwidth_nrd(rx) == pytest.approx(bw_x, abs=1e-12)
    assert bandwidth_nrd(ry) == pytest.approx(bw_y, abs=1e-12)


def test_density_grids_match_live_r(r_reference):
    cases = {
        "dens_contour": xy_density_contour(_X, _Y, bins=8),
        "dens_image": xy_density_image(_X, _Y, bins=8),
        "dens_persp": xy_density_persp(_X, _Y, bins=6),
        "dens_ties": xy_density_contour(_X2, _Y2, bins=5),
    }
    for key, ours in cases.items():
        _assert_grid(ours, r_reference[key], key)


def test_density_random_data_and_bins_match_live_r(r_reference):
    rx = np.asarray(r_reference["rx"], float)
    ry = np.asarray(r_reference["ry"], float)
    cases = {
        "dens_n25": xy_density_contour(rx, ry, bins=25),
        "dens_n3": xy_density_image(rx, ry, bins=3),
        "dens_default": xy_density_persp(rx, ry),  # bins=25 default
    }
    for key, ours in cases.items():
        _assert_grid(ours, r_reference[key], key)


def test_kde2d_long_tables_match_live_r(r_reference):
    _assert_long(xy_hist_kde2d(_X, _Y, bins=8), r_reference["hist_kde"],
                 "hist_kde", "z")
    rx = np.asarray(r_reference["rx"], float)
    ry = np.asarray(r_reference["ry"], float)
    _assert_long(xy_hist_kde2d(rx, ry, bins=12), r_reference["hist_kde_n12"],
                 "hist_kde_n12", "z")


def test_ecdf_hist_matches_live_r(r_reference):
    _assert_long(xy_ECDF_hist(_X, _Y), r_reference["ecdf_hist"],
                 "ecdf_hist", "Fn")
    _assert_long(
        xy_ECDF_hist(np.array([1, 1, 2, 2, 3, 3]),
                     np.array([3, 1, 2, 2, 1, 3])),
        r_reference["ecdf_ties"], "ecdf_ties", "Fn")


def test_ecdf_hist_is_same_grid_as_contour_variant(r_reference):
    # xy_ECDF_hist (underscore R export) and xy.ECDF.contour (point-named)
    # compute the identical 40x40 grid.
    from omnilss.bivar_grid import xy_ECDF_contour
    np.testing.assert_array_equal(xy_ECDF_hist(_X, _Y).z,
                                  xy_ECDF_contour(_X, _Y).z)


def test_density_variants_compute_identical_grid():
    a = xy_density_contour(_X, _Y, bins=8)
    b = xy_density_image(_X, _Y, bins=8)
    np.testing.assert_allclose(a.z, b.z)
    np.testing.assert_allclose(a.x, b.x)
    # persp has a different default bins (25), pass explicit bins to compare
    c = xy_density_persp(_X, _Y, bins=8)
    np.testing.assert_allclose(a.z, c.z)


def test_matrix_input_intent_and_length_validation():
    # R's matrix branch is unreachable (y has no default); our port supports it.
    arr = np.column_stack([_X, _Y])
    g = xy_density_contour(arr, bins=8)
    np.testing.assert_allclose(g.z, xy_density_contour(_X, _Y, bins=8).z)
    with pytest.raises(ValueError, match="same length"):
        kde2d(_X, _Y[:-1])
    with pytest.raises(ValueError, match="2-column"):
        xy_density_contour(np.arange(5.0), bins=8)


def test_kde2d_shape_and_positive_density():
    g = kde2d(_X, _Y, n=10)
    assert isinstance(g, BivarGrid)
    assert g.z.shape == (10, 10)
    assert np.all(g.z >= 0.0)
    assert np.all(np.isfinite(g.z))
    # integrates (roughly) to 1 over the grid area
    area = (g.x[1] - g.x[0]) * (g.y[1] - g.y[0])
    assert g.z.sum() * area > 0.5
