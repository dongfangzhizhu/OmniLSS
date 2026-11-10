"""R gamlss.ggplots::y_ecdf non-parametric confidence bands -- live-R parity.

R source: ``R/gamlss.ggplots/R/y_ecdf.R``.  The oracle sources the vendored R
file after stripping ``ggplot2::`` and stubbing the drawing calls; ``ECDF`` is
bound to the real ``gamlss::ECDF``.  ``geom_ribbon`` / ``stat_ecdf`` recorders
capture the exact data frames the REAL R function body would have plotted.

Pinned R semantics (10 groups, all <= 1.6e-15):
* Owen / Jager-Wellner bands: ``pbeta/qbeta`` with shape 1/3 (scipy
  ``scipy.stats.beta.cdf/ppf``), polynomial ``lambda`` in ``log(nn)`` with
  n<=100 / n>100 branches, ``nn = n+1``, ``p = cumsum(table)/nn``.
* 99%-Owen ``lambda`` is negative for small n -> ``sqrt`` NaN -> all-NaN band
  (R emits "NaNs produced"); reproduced, NOT fixed.
* ECDF scores use ``gamlss::ECDF`` denominator **n+1** (not n).
* ``other_prep_data`` drops NAs by observation position; outlier colour via
  ``median +/- value*IQR`` (type-7 quantiles); bands are computed on the
  NA-dropped vector ``d$y``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.ecdf_bands import EcdfBand, EcdfData, y_ecdf

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""# ggplot2 not installed: strip prefix, stub glyphs, record geom_ribbon and
# stat_ecdf data.  ECDF comes from the real gamlss namespace.
suppressPackageStartupMessages(library(jsonlite))
store <- new.env(); store$rib <- NULL; store$ecdf <- NULL
txt <- gsub('ggplot2::', '', readLines('R/gamlss.ggplots/R/y_ecdf.R'), fixed=TRUE)
stub <- function(...) NULL
for (f in c("aes","xlab","ylab","ggtitle","xlim","theme_minimal","theme",
            "geom_hline","geom_text","geom_point","geom_line","after_stat",
            "geom_linerange","scale_x_continuous","coord_flip","geom_histogram",
            "geom_density","element_blank","element_line","annotate")) {
  assign(f, stub, envir=globalenv())
}
rec_rib <- function(data=NULL, ...) { store$rib <- as.data.frame(data); structure(list(),class="ggstub") }
Ops.ggstub <- function(e1,e2) e1
rec_ecdf <- function(data=NULL, ...) { store$ecdf <- as.data.frame(data); structure(list(),class="ggstub") }
ggplot <- function(...) structure(list(), class="ggstub")
ECDF <- gamlss::ECDF
geom_ribbon <- rec_rib; stat_ecdf <- rec_ecdf
eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())
grab <- function(expr) { store$rib<-NULL; store$ecdf<-NULL; force(expr); list(rib=store$rib, ecdf=store$ecdf) }

set.seed(3)
y  <- c(1.2,2.5,3.1,4.8,5.0,7.3,8.1,9.6,10.2,12.0)
y2 <- c(1,1,2,2,3,3,4,4,5,5,6,6)
yna <- c(1.0,NA,2.5,3.0,NA,7.0,8.0,9.0)
set.seed(31)
yn <- rnorm(150)
yo <- c(rnorm(20), 30, -25, 28, -30)
out <- list(
  owen95 = grab(y_ecdf(y, type="Owen", conf.level="95")),
  jw95   = grab(y_ecdf(y, type="JW",   conf.level="95")),
  owen99 = grab(y_ecdf(y, type="Owen", conf.level="99")),
  jw99   = grab(y_ecdf(y, type="JW",   conf.level="99")),
  ties   = grab(y_ecdf(y2, type="Owen")),
  withna = grab(y_ecdf(yna, type="Owen")),
  n150_owen95 = grab(y_ecdf(yn, type="Owen", conf.level="95")),
  n150_jw99   = grab(y_ecdf(yn, type="JW",   conf.level="99")),
  outl_v1 = grab(y_ecdf(yo, type="Owen", value=1)),
  outl_v4 = grab(y_ecdf(yo, type="JW",   value=4)),
  yn = as.numeric(yn), yo = as.numeric(yo)
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_Y = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0])
_Y2 = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
_YNA = np.array([1.0, np.nan, 2.5, 3.0, np.nan, 7.0, 8.0, 9.0])


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.ggplots" / "R" / "y_ecdf.R").exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "ecdf.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-500:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _assert_band(ours: EcdfBand, rows: list, key: str) -> None:
    rx = np.array([r["x"] for r in rows], float)
    rlo = np.array([r.get("lower", np.nan) for r in rows], float)
    rup = np.array([r.get("upper", np.nan) for r in rows], float)
    np.testing.assert_allclose(ours.x, rx, rtol=0, atol=1e-12, err_msg=key + ":x")
    # NaN pattern must match exactly (99%-Owen produces all-NaN bands)
    assert np.array_equal(np.isnan(ours.lower), np.isnan(rlo)), key + " lower NaN"
    assert np.array_equal(np.isnan(ours.upper), np.isnan(rup)), key + " upper NaN"
    np.testing.assert_allclose(ours.lower[~np.isnan(rlo)], rlo[~np.isnan(rlo)],
                               rtol=0, atol=1e-12, err_msg=key + ":lower")
    np.testing.assert_allclose(ours.upper[~np.isnan(rup)], rup[~np.isnan(rup)],
                               rtol=0, atol=1e-12, err_msg=key + ":upper")


def _assert_ecdf(ours: EcdfData, rows: list, key: str) -> None:
    robs = np.array([r["obs"] for r in rows], float)
    ry = np.array([r["y"] for r in rows], float)
    rscores = np.array([r["scores"] for r in rows], float)
    np.testing.assert_allclose(ours.obs, robs, rtol=0, atol=0, err_msg=key + ":obs")
    np.testing.assert_allclose(ours.y, ry, rtol=0, atol=0, err_msg=key + ":y")
    np.testing.assert_allclose(ours.scores, rscores, rtol=0, atol=1e-12,
                               err_msg=key + ":scores")
    if all("color" in r for r in rows):
        assert list(ours.color) == [r["color"] for r in rows], key + ":color"


def test_all_band_types_match_live_r(r_reference):
    cases = {
        "owen95": (_Y, "Owen", "95"),
        "jw95": (_Y, "JW", "95"),
        "owen99": (_Y, "Owen", "99"),
        "jw99": (_Y, "JW", "99"),
        "ties": (_Y2, "Owen", "95"),
        "withna": (_YNA, "Owen", "95"),
    }
    for key, (y, bt, cl) in cases.items():
        band, data = y_ecdf(y, type=bt, conf_level=cl)
        _assert_band(band, r_reference[key]["rib"], key)
        _assert_ecdf(data, r_reference[key]["ecdf"], key)


def test_n_gt_100_and_outlier_colouring_match_live_r(r_reference):
    yn = np.asarray(r_reference["yn"], float)
    yo = np.asarray(r_reference["yo"], float)
    cases = {
        "n150_owen95": (yn, "Owen", "95", 2),
        "n150_jw99": (yn, "JW", "99", 2),
        "outl_v1": (yo, "Owen", "95", 1),
        "outl_v4": (yo, "JW", "95", 4),
    }
    for key, (y, bt, cl, value) in cases.items():
        band, data = y_ecdf(y, type=bt, conf_level=cl, value=value)
        _assert_band(band, r_reference[key]["rib"], key)
        _assert_ecdf(data, r_reference[key]["ecdf"], key)


def test_owen99_band_is_all_nan_like_r(r_reference):
    # R's 99%-Owen lambda polynomial is negative for small n -> sqrt NaN.
    band, _ = y_ecdf(_Y, type="Owen", conf_level="99")
    assert np.all(np.isnan(band.lower)) and np.all(np.isnan(band.upper))
    rows = r_reference["owen99"]["rib"]
    assert all("lower" not in r and "upper" not in r for r in rows)


def test_ecdf_scores_use_n_plus_1_denominator():
    # gamlss::ECDF(c(1,2,2,3)) evaluated on the 4 observations:
    # #{<=}/(n+1) = c(1/5, 3/5, 3/5, 4/5) = c(0.2, 0.6, 0.6, 0.8)
    # (the two tied 2's both get 0.6)
    _, data = y_ecdf(np.array([1.0, 2.0, 2.0, 3.0]), type="Owen")
    np.testing.assert_allclose(data.scores, [0.2, 0.6, 0.6, 0.8], rtol=0, atol=0)


def test_nan_positions_are_dropped():
    _, data = y_ecdf(_YNA, type="Owen")
    assert data.y.size == 6
    np.testing.assert_allclose(data.obs, [1, 3, 4, 6, 7, 8], rtol=0, atol=0)
    assert np.all(np.isfinite(data.y))


def test_band_is_sorted_monotone_and_bounded():
    band, _ = y_ecdf(_Y, type="Owen", conf_level="95")
    assert isinstance(band, EcdfBand)
    assert np.all(np.diff(band.x) > 0)             # distinct values ascending
    assert np.all(band.lower >= 0) and np.all(band.upper <= 1)
    assert np.all(band.lower <= band.upper)
    assert band.conf_level == "95" and band.band_type == "Owen"


def test_invalid_arguments():
    with pytest.raises(ValueError, match="type"):
        y_ecdf(_Y, type="nope")
    with pytest.raises(ValueError, match="conf_level"):
        y_ecdf(_Y, conf_level="90")
