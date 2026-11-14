"""Bivariate scatter / PIT / histogram / density *plot data* from
``gamlss.ggplots`` (Copula_ralated.R).

R source: ``R/gamlss.ggplots/R/Copula_ralated.R`` (the underscore-named
``xy_*`` family).  The R functions draw ggplot2 objects; these ports return the
numerical data the plot layers would have consumed, matching the project
convention used by :mod:`omnilss.bivar_grid` and :mod:`omnilss.ggplots_diag`.

Numeric kernels (all deterministic, live-R verified in
``omnilss/tests/test_copula_xy.py``):

* **PIT transform** ``u = rank(x)/(n+1)``, ``v = rank(y)/(n+1)`` -- used by
  ``xy_pit_scatter`` / ``xy_pit_hist`` / ``xy_pit_density``.  R ``rank`` uses
  ``ties.method="average"`` (== ``scipy.stats.rankdata(method="average")``).
* **xy_median** -- the marker point on the scatter plots, from
  :func:`omnilss.copula_medians.xy_median` (D2.215, including the ``cw``->``pc``
  R bug).  ``xy_pit_scatter`` passes ``(v, u)`` to it, not ``(x, y)``.
* **50% HDR level** -- ``kde2d(..., n=100)`` grid (from
  :func:`omnilss.bivar_grid.kde2d`, D2.218) flattened column-major, sorted
  descending, cumulative mass reaching 0.5 gives the density threshold ``z50``.
  ``xy_scatter`` builds the KDE from ``(x, y)``; ``xy_pit_scatter`` from
  ``(v, u)`` (R source order).

R quirks reproduced: the ``xy_pit_*`` three share one PIT kernel; the two
scatter functions compute the same HDR/median structure (differing only in the
input being raw vs PIT and the ``(v,u)`` swap); ``xy_hist`` / ``xy_density`` /
``xy_density_plain`` are pure data-prep wrappers whose binning / KDE-contour
layers live in ggplot2 stat functions (not a numeric kernel here), so the ports
return just the ``x``/``y`` data frame plus the ``bins`` argument.

The siblings ``bivar_pdf`` / ``bivar_fun`` / ``bivar_fun_plotly`` (grid/plotly
stacks) and the point-named ``xy.density.*`` / ``bivar.fun.*`` (base graphics,
already in :mod:`omnilss.bivar_grid`) are separate and not part of this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .bivar_grid import BivarGrid, kde2d
from .copula_medians import _as_pair, xy_median

__all__ = [
    "PITScores",
    "ScatterData",
    "HistogramData",
    "DensityData",
    "pit_scores",
    "hdr50_level",
    "xy_scatter",
    "xy_pit_scatter",
    "xy_hist",
    "xy_pit_hist",
    "xy_density",
    "xy_density_plain",
    "xy_pit_density",
]


@dataclass(frozen=True)
class PITScores:
    """R ``df <- data.frame(u = rank(x)/(n+1), v = rank(y)/(n+1))``.

    ``u`` corresponds to ``x``, ``v`` to ``y`` (the columns of the R data
    frame).  ``xy_pit_scatter`` then plots ``aes(x=v, y=u)`` -- a swap kept in
    :class:`ScatterData` below, not here.
    """

    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class ScatterData:
    """Data behind ``xy_scatter`` / ``xy_pit_scatter``.

    ``x`` / ``y`` are the plotted coordinates: raw ``(x, y)`` for
    ``xy_scatter``; ``(v, u)`` for ``xy_pit_scatter`` (R plots ``aes(x=v,
    y=u)``).  ``median`` is the ``xy_median`` marker.  ``hdr`` / ``hdr_level``
    are the ``kde2d(n=100)`` grid and the 50% mass density threshold (``None``
    when ``HDR=False``).
    """

    x: np.ndarray
    y: np.ndarray
    median: np.ndarray
    hdr: BivarGrid | None
    hdr_level: float | None


@dataclass(frozen=True)
class HistogramData:
    """Data behind ``xy_hist`` / ``xy_pit_hist`` (``geom_bin2d``)."""

    x: np.ndarray
    y: np.ndarray
    bins: int


@dataclass(frozen=True)
class DensityData:
    """Data behind ``xy_density`` / ``xy_density_plain`` / ``xy_pit_density``.

    Only the ``x``/``y`` columns are pure data; the KDE contours come from
    ggplot2 ``stat_density_2d`` and are deliberately not reproduced here.
    """

    x: np.ndarray
    y: np.ndarray


def pit_scores(x: Any, y: Any = None) -> PITScores:
    """R PIT: ``u = rank(x)/(n+1)``, ``v = rank(y)/(n+1)`` (ties=average)."""
    a, b = _as_pair(x, y)
    n = a.size
    return PITScores(
        u=rankdata(a, method="average") / (n + 1.0),
        v=rankdata(b, method="average") / (n + 1.0),
    )


def hdr50_level(x: Any, y: Any, n: int = 100) -> tuple[BivarGrid, float]:
    """R 50% HDR: ``kde2d(x, y, n=100)`` grid + density threshold ``z50``.

    ``z50`` is the density at which the descending-sorted grid masses first
    reach 50% of the total: ``z50 = z[order][first(cumsum/sum >= 0.5)]``.
    R ``as.vector(kde$z)`` flattens column-major (``ravel(order='F')``).
    """
    grid = kde2d(x, y, n=n)
    z = np.asarray(grid.z, dtype=np.float64).ravel(order="F")
    order = np.argsort(-z, kind="stable")  # R order(decreasing=TRUE), stable ties
    cumprob = np.cumsum(z[order]) / z.sum()
    level_index = int(np.flatnonzero(cumprob >= 0.5)[0])
    return grid, float(z[order][level_index])


def _scatter(a: np.ndarray, b: np.ndarray, method: str, hdr: bool) -> ScatterData:
    """Shared scatter builder; ``(a, b)`` are the plotted (x, y) coordinates."""
    med = np.asarray(xy_median(a, b, method=method), dtype=np.float64)
    if hdr:
        grid, z50 = hdr50_level(a, b, n=100)
        return ScatterData(x=a, y=b, median=med, hdr=grid, hdr_level=z50)
    return ScatterData(x=a, y=b, median=med, hdr=None, hdr_level=None)


def xy_scatter(
    x: Any,
    y: Any = None,
    method: str = "cw",
    HDR: bool = True,
    **_: Any,
) -> ScatterData:
    """R ``xy_scatter`` data: raw ``(x, y)`` + median + 50% HDR from ``(x,y)``."""
    a, b = _as_pair(x, y)
    return _scatter(a, b, method, HDR)


def xy_pit_scatter(
    x: Any,
    y: Any = None,
    method: str = "cw",
    HDR: bool = True,
    **_: Any,
) -> ScatterData:
    """R ``xy_pit_scatter`` data: PIT ``(v, u)`` plotted + median + HDR.

    R computes ``u=rank(x)/(n+1)``, ``v=rank(y)/(n+1)``, then plots
    ``aes(x=v, y=u)`` and calls ``xy_median(v, u)`` / ``kde2d(v, u, n=100)``.
    """
    a, b = _as_pair(x, y)
    pit = pit_scores(a, b)
    return _scatter(pit.v, pit.u, method, HDR)  # (x=v, y=u)


def xy_hist(x: Any, y: Any = None, bins: int = 50, **_: Any) -> HistogramData:
    """R ``xy_hist`` data: raw ``(x, y)`` + ``bins`` (geom_bin2d)."""
    a, b = _as_pair(x, y)
    return HistogramData(x=a, y=b, bins=int(bins))


def xy_pit_hist(x: Any, y: Any = None, bins: int = 50, **_: Any) -> HistogramData:
    """R ``xy_pit_hist`` data: PIT ``(u, v)`` + ``bins``."""
    a, b = _as_pair(x, y)
    pit = pit_scores(a, b)
    return HistogramData(x=pit.u, y=pit.v, bins=int(bins))


def xy_density(x: Any, y: Any = None, **_: Any) -> DensityData:
    """R ``xy_density`` data: raw ``(x, y)`` fed to stat_density_2d."""
    a, b = _as_pair(x, y)
    return DensityData(x=a, y=b)


def xy_density_plain(x: Any, y: Any = None, **_: Any) -> DensityData:
    """R ``xy_density_plain`` data: raw ``(x, y)`` (same data as xy_density)."""
    a, b = _as_pair(x, y)
    return DensityData(x=a, y=b)


def xy_pit_density(x: Any, y: Any = None, **_: Any) -> DensityData:
    """R ``xy_pit_density`` data: PIT ``(u, v)`` fed to stat_density_2d."""
    a, b = _as_pair(x, y)
    pit = pit_scores(a, b)
    return DensityData(x=pit.u, y=pit.v)
