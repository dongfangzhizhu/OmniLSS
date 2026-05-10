"""R-aligned centiles comparison interfaces.

R source reference:
- file: `gamlss/R/centilescom.R`
- functions: `centiles.com`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centilesPLOT import CentileCurveEntry, centiles_data
from .model import GAMLSSModel


@dataclass(frozen=True)
class CentilesComparisonModel:
    """One staged `centiles.com` model-comparison payload."""

    index: int
    family: str
    centiles: tuple[float, ...]
    entries: tuple[CentileCurveEntry, ...]
    percent_below: dict[float, float]


@dataclass(frozen=True)
class CentilesComparisonResult:
    """Structured staged `centiles.com` payload."""

    xvar: str
    centiles: tuple[float, ...]
    models: tuple[CentilesComparisonModel, ...]
    observed_x: np.ndarray | None
    observed_y: np.ndarray | None


@dataclass(frozen=True)
class CentilesComparisonCoverageResult:
    """Structured staged centile-comparison coverage matrix."""

    xvar: str
    centiles: tuple[float, ...]
    families: tuple[str, ...]
    matrix: np.ndarray


def centiles_comparison_data(
    obj: GAMLSSModel,
    *others: GAMLSSModel,
    xvar: str,
    cent: list[float] | tuple[float, ...] = (0.4, 10.0, 50.0, 90.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
    include_observed: bool = True,
) -> CentilesComparisonResult:
    """R reference: `gamlss/R/centilescom.R::centiles.com`.

    Staged behavior:
    - Returns per-model centile curves for the same x-axis instead of plotting.
    - Reports the percentage of observed responses below each centile curve.
    - Reuses `centiles_data()` so comparison output stays aligned with the
      single-model centile path.
    """

    from .methods import _require_gamlss_method

    objects = (obj,) + tuple(others)
    if not objects:
        raise ValueError("at least one gamlss object is required")
    for current in objects:
        _require_gamlss_method(current)

    base = objects[0]
    call_data = None if base.call is None else base.call.get("data")
    response_name = base.terms.get("mu", {}).get("response")
    observed_x: np.ndarray | None = None
    observed_y: np.ndarray | None = None
    if include_observed and call_data is not None and response_name is not None and xvar in call_data and response_name in call_data:
        raw_x = np.asarray(call_data[xvar], dtype=np.float64).ravel()
        raw_y = np.asarray(call_data[response_name], dtype=np.float64).ravel()
        order = np.argsort(raw_x)
        observed_x = raw_x[order]
        observed_y = raw_y[order]

    models: list[CentilesComparisonModel] = []
    centiles = tuple(float(value) for value in cent)
    for index, current in enumerate(objects, start=1):
        result = centiles_data(
            current,
            xvar=xvar,
            cent=centiles,
            n_points=n_points,
            how=how,
            fixed_at=fixed_at,
            include_observed=False,
        )
        current_call_data = None if current.call is None else current.call.get("data")
        current_response = current.terms.get("mu", {}).get("response")
        percent_below: dict[float, float] = {}
        if current_call_data is not None and current_response is not None and xvar in current_call_data and current_response in current_call_data:
            current_x = np.asarray(current_call_data[xvar], dtype=np.float64).ravel()
            current_y = np.asarray(current_call_data[current_response], dtype=np.float64).ravel()
            order = np.argsort(current_x)
            sorted_x = current_x[order]
            sorted_y = current_y[order]
            for entry in result.entries:
                curve = np.interp(sorted_x, entry.x, entry.y)
                percent_below[float(entry.centile)] = float(100.0 * np.mean(sorted_y <= curve))
        models.append(
            CentilesComparisonModel(
                index=index,
                family=current.family.name,
                centiles=result.centiles,
                entries=result.entries,
                percent_below=percent_below,
            )
        )
    return CentilesComparisonResult(
        xvar=str(xvar),
        centiles=centiles,
        models=tuple(models),
        observed_x=observed_x,
        observed_y=observed_y,
    )


def centiles_comparison_coverage_data(
    obj: GAMLSSModel,
    *others: GAMLSSModel,
    xvar: str,
    cent: list[float] | tuple[float, ...] = (0.4, 10.0, 50.0, 90.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
) -> CentilesComparisonCoverageResult:
    """R reference: `gamlss/R/centilescom.R::centiles.com`.

    Staged behavior:
    - Returns the `% below centile` summaries as a model-by-centile matrix.
    - Reuses `centiles_comparison_data()` so the matrix stays aligned with the
      staged comparison curves and coverage calculations.
    """

    comparison = centiles_comparison_data(
        obj,
        *others,
        xvar=xvar,
        cent=cent,
        n_points=n_points,
        how=how,
        fixed_at=fixed_at,
        include_observed=False,
    )
    matrix = np.full((len(comparison.models), len(comparison.centiles)), np.nan, dtype=np.float64)
    families: list[str] = []
    for row_index, model in enumerate(comparison.models):
        families.append(model.family)
        for column_index, centile in enumerate(comparison.centiles):
            matrix[row_index, column_index] = model.percent_below.get(float(centile), np.nan)
    return CentilesComparisonCoverageResult(
        xvar=comparison.xvar,
        centiles=comparison.centiles,
        families=tuple(families),
        matrix=matrix,
    )


centiles_com = centiles_comparison_data


__all__ = [
    "CentilesComparisonModel",
    "CentilesComparisonResult",
    "CentilesComparisonCoverageResult",
    "centiles_com",
    "centiles_comparison_data",
    "centiles_comparison_coverage_data",
]
