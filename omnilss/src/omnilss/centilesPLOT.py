"""R-aligned centile plotting interfaces.

R source reference:
- file: `gamlss/R/centilesPLOT.R`
- functions: `centiles`, `centiles.fan`, `centiles.split`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import GAMLSSModel
from .getQuantile import get_quantile_data


@dataclass(frozen=True)
class CentileCurveEntry:
    """One staged centile-curve bundle for a single centile percentage."""

    centile: float
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class CentilesResult:
    """Structured staged `centiles` payload."""

    family: str
    xvar: str
    centiles: tuple[float, ...]
    entries: tuple[CentileCurveEntry, ...]
    observed_x: np.ndarray | None
    observed_y: np.ndarray | None
    fixed_at: dict[str, float]


@dataclass(frozen=True)
class CentilesCoverageRow:
    """One staged `centiles(save=TRUE)` summary row."""

    centile: float
    percent_below: float


@dataclass(frozen=True)
class CentilesCoverageResult:
    """Structured staged `centiles(save=TRUE)` payload."""

    family: str
    xvar: str
    rows: tuple[CentilesCoverageRow, ...]
    observed_x: np.ndarray | None
    observed_y: np.ndarray | None


@dataclass(frozen=True)
class CentilesSplitPanel:
    """One staged `centiles.split` panel payload."""

    interval: tuple[float, float]
    centiles: tuple[float, ...]
    entries: tuple[CentileCurveEntry, ...]
    observed_x: np.ndarray | None
    observed_y: np.ndarray | None


@dataclass(frozen=True)
class CentilesSplitResult:
    """Structured staged `centiles.split` payload."""

    family: str
    xvar: str
    centiles: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    panels: tuple[CentilesSplitPanel, ...]
    fixed_at: dict[str, float]


@dataclass(frozen=True)
class CentilesSplitCoverageResult:
    """Structured staged `centiles.split(save=TRUE)` payload."""

    family: str
    xvar: str
    centiles: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    matrix: np.ndarray


def centiles_data(
    object: GAMLSSModel,
    xvar: str,
    cent: list[float] | tuple[float, ...] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
    include_observed: bool = True,
    x_limits: tuple[float, float] | None = None,
) -> CentilesResult:
    """R reference: `gamlss/R/centilesPLOT.R::centiles`.

    Staged behavior:
    - Returns centile-curve data instead of plotting.
    - Delegates distribution-aware quantiles to `get_quantile_data()`.
    - Optionally includes sorted observed points for overlay/report use.
    """

    from .methods import _require_gamlss_method

    _require_gamlss_method(object)
    centiles = tuple(float(value) for value in cent)
    if not centiles:
        raise ValueError("at least one centile is required")
    if any(value <= 0.0 or value >= 100.0 for value in centiles):
        raise ValueError("centiles must lie strictly between 0 and 100")

    quantile_result = get_quantile_data(
        object,
        xvar=xvar,
        probabilities=tuple(value / 100.0 for value in centiles),
        n_points=n_points,
        how=how,
        fixed_at=fixed_at,
        x_limits=x_limits,
    )

    observed_x: np.ndarray | None = None
    observed_y: np.ndarray | None = None
    if include_observed:
        call_data = None if object.call is None else object.call.get("data")
        response_name = object.terms.get("mu", {}).get("response")
        if call_data is not None and response_name is not None and xvar in call_data and response_name in call_data:
            x_values = np.asarray(call_data[xvar], dtype=np.float64).ravel()
            y_values = np.asarray(call_data[response_name], dtype=np.float64).ravel()
            if x_limits is not None:
                lower, upper = float(x_limits[0]), float(x_limits[1])
                mask = (x_values >= lower) & (x_values <= upper)
                x_values = x_values[mask]
                y_values = y_values[mask]
            order = np.argsort(x_values)
            observed_x = x_values[order]
            observed_y = y_values[order]

    entries = tuple(
        CentileCurveEntry(
            centile=centile,
            x=np.asarray(entry.x, dtype=np.float64),
            y=np.asarray(entry.quantile, dtype=np.float64),
        )
        for centile, entry in zip(centiles, quantile_result.entries)
    )
    return CentilesResult(
        family=quantile_result.family,
        xvar=quantile_result.xvar,
        centiles=centiles,
        entries=entries,
        observed_x=observed_x,
        observed_y=observed_y,
        fixed_at=dict(quantile_result.fixed_at),
    )


def centiles_coverage_data(
    object: GAMLSSModel,
    xvar: str,
    cent: list[float] | tuple[float, ...] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
) -> CentilesCoverageResult:
    """R reference: `gamlss/R/centilesPLOT.R::centiles` with `save=TRUE`.

    Staged behavior:
    - Returns the saved centile/coverage table separately from plotting data.
    - Reuses `centiles_data()` and computes the percentage of observed values
      lying below each centile curve.
    """

    centiles_result = centiles_data(
        object,
        xvar=xvar,
        cent=cent,
        n_points=n_points,
        how=how,
        fixed_at=fixed_at,
        include_observed=True,
    )
    observed_x = None if centiles_result.observed_x is None else np.asarray(centiles_result.observed_x, dtype=np.float64)
    observed_y = None if centiles_result.observed_y is None else np.asarray(centiles_result.observed_y, dtype=np.float64)
    rows: list[CentilesCoverageRow] = []
    for entry in centiles_result.entries:
        if observed_x is None or observed_y is None or observed_x.size == 0:
            percent_below = float("nan")
        else:
            curve = np.interp(observed_x, entry.x, entry.y)
            percent_below = float(100.0 * np.mean(observed_y <= curve))
        rows.append(CentilesCoverageRow(centile=float(entry.centile), percent_below=percent_below))
    return CentilesCoverageResult(
        family=centiles_result.family,
        xvar=centiles_result.xvar,
        rows=tuple(rows),
        observed_x=observed_x,
        observed_y=observed_y,
    )


def centiles_split_data(
    object: GAMLSSModel,
    xvar: str,
    xcut_points: Sequence[float] | None = None,
    n_inter: int = 4,
    cent: list[float] | tuple[float, ...] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    overlap: float = 0.0,
    fixed_at: dict[str, float] | None = None,
    include_observed: bool = True,
) -> CentilesSplitResult:
    """R reference: `gamlss/R/centilesPLOT.R::centiles.split`.

    Staged behavior:
    - Returns panel-wise centile-curve data instead of arranging base-R plots.
    - Supports explicit cut points or automatic x-interval construction.
    - Reuses `centiles_data()` so split panels share the same quantile path.
    """

    from .methods import _require_gamlss_method

    _require_gamlss_method(object)
    if n_inter < 1:
        raise ValueError("n_inter must be at least 1")
    overlap_value = float(overlap)
    if overlap_value < 0.0 or overlap_value >= 1.0:
        raise ValueError("overlap must lie in [0, 1)")

    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("centiles_split_data requires call['data']")
    if xvar not in call_data:
        raise KeyError(f"xvar {xvar!r} not found in data")

    x_values = np.asarray(call_data[xvar], dtype=np.float64).ravel()
    if x_values.size == 0:
        raise ValueError("xvar data is empty")
    min_x = float(np.nanmin(x_values))
    max_x = float(np.nanmax(x_values))

    if xcut_points is None:
        probabilities = np.linspace(0.0, 1.0, int(n_inter) + 1, dtype=np.float64)
        edges = np.quantile(x_values, probabilities)
    else:
        cuts = np.asarray(tuple(float(value) for value in xcut_points), dtype=np.float64)
        if cuts.ndim != 1:
            raise ValueError("xcut_points must be one-dimensional")
        if cuts.size == 0:
            raise ValueError("xcut_points cannot be empty")
        if np.any(~np.isfinite(cuts)):
            raise ValueError("xcut_points must be finite")
        if np.any(np.diff(cuts) <= 0.0):
            raise ValueError("xcut_points must be strictly increasing")
        if cuts[0] <= min_x or cuts[-1] >= max_x:
            raise ValueError("xcut_points must lie strictly within the observed x range")
        edges = np.concatenate(([min_x], cuts, [max_x]))

    intervals: list[tuple[float, float]] = []
    for lower_raw, upper_raw in zip(edges[:-1], edges[1:]):
        lower = float(lower_raw)
        upper = float(upper_raw)
        if upper <= lower:
            continue
        width = upper - lower
        pad = 0.5 * overlap_value * width
        interval_lower = max(min_x, lower - pad)
        interval_upper = min(max_x, upper + pad)
        if interval_upper <= interval_lower:
            continue
        intervals.append((interval_lower, interval_upper))
    if not intervals:
        raise ValueError("no valid intervals could be constructed")

    panels: list[CentilesSplitPanel] = []
    chosen_fixed: dict[str, float] | None = None
    for lower, upper in intervals:
        panel = centiles_data(
            object,
            xvar=xvar,
            cent=cent,
            n_points=n_points,
            how=how,
            fixed_at=fixed_at,
            include_observed=include_observed,
            x_limits=(lower, upper),
        )
        if chosen_fixed is None:
            chosen_fixed = dict(panel.fixed_at)
        panels.append(
            CentilesSplitPanel(
                interval=(lower, upper),
                centiles=panel.centiles,
                entries=panel.entries,
                observed_x=panel.observed_x,
                observed_y=panel.observed_y,
            )
        )
    return CentilesSplitResult(
        family=object.family.name,
        xvar=str(xvar),
        centiles=tuple(float(value) for value in cent),
        intervals=tuple(intervals),
        panels=tuple(panels),
        fixed_at={} if chosen_fixed is None else chosen_fixed,
    )


def centiles_split_coverage_data(
    object: GAMLSSModel,
    xvar: str,
    xcut_points: Sequence[float] | None = None,
    n_inter: int = 4,
    cent: list[float] | tuple[float, ...] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    overlap: float = 0.0,
    fixed_at: dict[str, float] | None = None,
) -> CentilesSplitCoverageResult:
    """R reference: `gamlss/R/centilesPLOT.R::centiles.split` with `save=TRUE`.

    Staged behavior:
    - Returns the saved interval-by-centile coverage matrix separately from the
      panel plotting data.
    - Reuses `centiles_split_data()` and computes the same per-panel coverage
      percentages that R stores in matrix `X`.
    """

    split_result = centiles_split_data(
        object,
        xvar=xvar,
        xcut_points=xcut_points,
        n_inter=n_inter,
        cent=cent,
        n_points=n_points,
        how=how,
        overlap=overlap,
        fixed_at=fixed_at,
        include_observed=True,
    )
    matrix = np.full((len(split_result.centiles), len(split_result.panels)), np.nan, dtype=np.float64)
    for column_index, panel in enumerate(split_result.panels):
        observed_x = None if panel.observed_x is None else np.asarray(panel.observed_x, dtype=np.float64)
        observed_y = None if panel.observed_y is None else np.asarray(panel.observed_y, dtype=np.float64)
        for row_index, entry in enumerate(panel.entries):
            if observed_x is None or observed_y is None or observed_x.size == 0:
                percent_below = float("nan")
            else:
                curve = np.interp(observed_x, entry.x, entry.y)
                percent_below = float(100.0 * np.mean(observed_y <= curve))
            matrix[row_index, column_index] = percent_below
    return CentilesSplitCoverageResult(
        family=split_result.family,
        xvar=split_result.xvar,
        centiles=split_result.centiles,
        intervals=split_result.intervals,
        matrix=matrix,
    )


centiles = centiles_data
centiles_split = centiles_split_data

__all__ = [
    "CentileFanBand",
    "CentileCurveEntry",
    "CentilesCoverageResult",
    "CentilesFanResult",
    "CentilesResult",
    "CentilesSplitCoverageResult",
    "CentilesSplitResult",
    "centiles",
    "centiles_coverage_data",
    "centiles_data",
    "centiles_split",
    "centiles_split_coverage_data",
    "centiles_split_data",
]
