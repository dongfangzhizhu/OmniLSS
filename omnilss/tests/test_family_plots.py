"""R gamlss.dist::plotBEOI / plotBEZI / plotZAGA / plotZAIG parity.

R sources: ``R/gamlss.dist/R/BEOI.r``, ``BEZI.r``, ``ZAGA.R``, ``ZAIG.R``.

The R functions draw a density grid plus the point mass at the inflated
boundary.  This port returns the plot data instead of drawing.  The live-R
oracle below sources the vendored R files into the global environment and
overrides ``plot``/``points`` with recorders, so the *real R functions* run and
hand back exactly the data they would have plotted (including plotBEOI's
leftover ``print(pr1)`` side effect).

Two R quirks are pinned here:

* ``from`` is a dead argument -- every function hard-codes a 0.001 grid start;
* ``plotBEOI`` uses the point y=1 / ``pr1``, the other three use y=0 / ``pr0``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.family_plots import (
    InflatedDensityPlot,
    plotBEOI,
    plotBEZI,
    plotZAGA,
    plotZAIG,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss.dist))
library(jsonlite)
source('R/gamlss.dist/R/BEOI.r')
source('R/gamlss.dist/R/BEZI.r')
source('R/gamlss.dist/R/ZAGA.R')
source('R/gamlss.dist/R/ZAIG.R')

store <- new.env(); store$plot <- NULL; store$points <- NULL
plot <- function(x, ...) {
  f <- as.formula(x); env <- parent.frame(); a <- list(...)
  store$plot <- list(y=as.numeric(eval(f[[3]], env)),
                     pdf=as.numeric(eval(f[[2]], env)),
                     main=as.character(a$main), ylim=as.numeric(a$ylim),
                     type=as.character(a$type))
  invisible(NULL)
}
points <- function(x, y=NULL, ...) {
  store$points <- c(store$points, list(list(x=as.numeric(x), y=as.numeric(y))))
  invisible(NULL)
}
run <- function(fun, ...) {
  store$plot <- NULL; store$points <- NULL
  txt <- capture.output(fun(...))
  list(plot=store$plot, points=store$points, printed=txt)
}

out <- list(
  BEOI = run(plotBEOI),
  BEZI = run(plotBEZI),
  ZAGA = run(plotZAGA),
  ZAIG = run(plotZAIG),
  BEOI_c = run(plotBEOI, mu=0.3, sigma=2, nu=0.25, to=0.8, n=7),
  BEZI_c = run(plotBEZI, mu=0.4, sigma=0.7, nu=0.35, to=0.9, n=9),
  ZAGA_c = run(plotZAGA, mu=3, sigma=0.5, nu=0.4, to=5, n=9, main="custom ZAGA"),
  ZAIG_c = run(plotZAIG, mu=2, sigma=1.5, nu=0.2, to=6, n=11, main="custom ZAIG"),
  BEOI_from = run(plotBEOI, from=0.5, to=1, n=5),
  BEZI_from = run(plotBEZI, from=0.7, to=0.999, n=5)
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""


def _scalar(value):
    return float(np.asarray(value, dtype=float).ravel()[0])


def _vector(value):
    return np.asarray(value, dtype=float).ravel()


def _strings(value):
    return [str(v) for v in np.asarray(value).ravel()]


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.dist" / "R" / "BEOI.r").exists():
        pytest.skip("vendored gamlss.dist source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "plots.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-400:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _cases():
    return {
        "BEOI": plotBEOI(),
        "BEZI": plotBEZI(),
        "ZAGA": plotZAGA(),
        "ZAIG": plotZAIG(),
        "BEOI_c": plotBEOI(mu=0.3, sigma=2, nu=0.25, to=0.8, n=7),
        "BEZI_c": plotBEZI(mu=0.4, sigma=0.7, nu=0.35, to=0.9, n=9),
        "ZAGA_c": plotZAGA(mu=3, sigma=0.5, nu=0.4, to=5, n=9,
                           main="custom ZAGA"),
        "ZAIG_c": plotZAIG(mu=2, sigma=1.5, nu=0.2, to=6, n=11,
                           main="custom ZAIG"),
        "BEOI_from": plotBEOI(from_=0.5, to=1, n=5),
        "BEZI_from": plotBEZI(from_=0.7, to=0.999, n=5),
    }


# ---------------------------------------------------------------------------
# Live-R parity
# ---------------------------------------------------------------------------


def test_all_plot_data_matches_live_r(r_reference):
    ours = _cases()
    for key, fit in ours.items():
        block = r_reference[key]
        plotted = block["plot"]
        np.testing.assert_allclose(
            fit.y, _vector(plotted["y"]), rtol=0, atol=0, err_msg=f"{key}: grid"
        )
        np.testing.assert_allclose(
            fit.density, _vector(plotted["pdf"]), rtol=1e-10, atol=1e-12,
            err_msg=f"{key}: density",
        )
        assert fit.main == _strings(plotted["main"])[0], key
        np.testing.assert_allclose(
            fit.ylim, _vector(plotted["ylim"]), rtol=1e-10, atol=1e-12,
            err_msg=f"{key}: ylim",
        )
        point = block["points"][0]
        assert fit.point == pytest.approx(_scalar(point["x"])), key
        assert fit.point_density == pytest.approx(_scalar(point["y"]), rel=1e-10), key
        assert len(block["points"]) == 2, key  # R draws it twice: "h" and "p"


def test_the_dead_from_argument_is_ignored_like_r(r_reference):
    """R hard-codes seq(from=0.001, ...); passing from=0.5 changes nothing."""
    ours = _cases()
    for key in ("BEOI_from", "BEZI_from"):
        assert ours[key].y[0] == pytest.approx(
            _scalar(r_reference[key]["plot"]["y"][0]), rel=0, abs=0
        )
    np.testing.assert_allclose(
        ours["BEOI_from"].y, plotBEOI(to=1, n=5).y, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        ours["BEZI_from"].y, plotBEZI(to=0.999, n=5).y, rtol=0, atol=0
    )


def test_plot_beoi_prints_its_point_mass_and_others_do_not(r_reference):
    """R plotBEOI has a leftover print(pr1); the value is our point_density."""
    printed = r_reference["BEOI"]["printed"]
    assert printed, "R plotBEOI should print pr1"
    assert float(printed[0].split()[1]) == pytest.approx(
        _cases()["BEOI"].point_density, rel=1e-10
    )
    for key in ("BEZI", "ZAGA", "ZAIG"):
        assert r_reference[key]["printed"] == [], key


def test_one_inflated_uses_point_one_and_zero_inflated_point_zero():
    assert plotBEOI().point == 1.0
    for factory in (plotBEZI, plotZAGA, plotZAIG):
        assert factory().point == 0.0


# ---------------------------------------------------------------------------
# Pure-Python contracts
# ---------------------------------------------------------------------------


def test_returns_plot_data_without_drawing():
    fit = plotZAGA()
    assert isinstance(fit, InflatedDensityPlot)
    assert fit.family == "ZAGA"
    assert fit.y.shape == fit.density.shape == (101,)
    assert fit.ylim[0] == 0.0
    assert fit.ylim[1] >= float(np.max(fit.density))
    assert fit.point_mass == fit.point_density
    assert fit.parameters == {"mu": 5.0, "sigma": 1.0, "nu": 0.1}


def test_default_titles_and_custom_main():
    assert plotBEOI().main == "One Inflated Beta"
    assert plotBEZI().main == "Zero Inflated Beta"
    assert plotZAGA().main == "Zero Adj. Gamma"
    assert plotZAIG().main == "Zero adjusted IG"
    assert plotZAGA(main="mine").main == "mine"
    assert plotZAIG(main="mine").main == "mine"


def test_n_controls_the_grid_and_must_be_positive():
    assert plotZAGA(n=7).y.shape == (7,)
    assert plotZAGA(n=1).y.shape == (1,)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        plotZAGA(n=0)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        plotBEOI(n=-3)


def test_grid_always_starts_at_0_001_regardless_of_from():
    for factory in (plotBEOI, plotBEZI, plotZAGA, plotZAIG):
        assert factory(from_=0.5, n=4).y[0] == pytest.approx(0.001, rel=0, abs=0)


def test_extra_plot_arguments_are_accepted_and_ignored():
    fit = plotBEOI(col="blue", lwd=2, type="l")
    assert fit.density.shape == (101,)


def test_density_and_point_mass_are_finite_and_nonnegative():
    for factory in (plotBEOI, plotBEZI, plotZAGA, plotZAIG):
        fit = factory()
        assert np.all(np.isfinite(fit.density))
        assert np.all(fit.density >= 0.0)
        assert np.isfinite(fit.point_density) and fit.point_density >= 0.0
