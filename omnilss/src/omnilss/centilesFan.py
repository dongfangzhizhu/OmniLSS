"""R-aligned centile fan interfaces.

R source reference:
- file: `gamlss/R/centilesFan.R`
- functions: `centiles.fan`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centilesPLOT import CentileCurveEntry, centiles_data
from .model import GAMLSSModel


@dataclass(frozen=True)
class CentileFanBand:
    """One staged `centiles.fan` symmetric band payload."""

    lower_centile: float
    upper_centile: float
    x: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    color_index: int


@dataclass(frozen=True)
class CentilesFanResult:
    """Structured staged `centiles.fan` payload."""

    family: str
    xvar: str
    centiles: tuple[float, ...]
    bands: tuple[CentileFanBand, ...]
    median: CentileCurveEntry | None
    observed_x: np.ndarray | None
    observed_y: np.ndarray | None
    fixed_at: dict[str, float]


def centiles_fan_data(
    object: GAMLSSModel,
    xvar: str,
    cent: list[float] | tuple[float, ...] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
    include_observed: bool = False,
    include_median: bool = True,
) -> CentilesFanResult:
    """R reference: `gamlss/R/centilesFan.R::centiles.fan`.

    Staged behavior:
    - Returns symmetric centile-band data instead of drawing polygons.
    - Reuses `centiles_data()` so fan charts and centile curves share the same
      distribution-aware quantile approximation path.
    """

    centiles_result = centiles_data(
        object,
        xvar=xvar,
        cent=cent,
        n_points=n_points,
        how=how,
        fixed_at=fixed_at,
        include_observed=include_observed,
    )
    entry_map = {float(entry.centile): entry for entry in centiles_result.entries}
    sorted_centiles = tuple(sorted(float(value) for value in centiles_result.centiles))
    bands: list[CentileFanBand] = []
    band_count = len(sorted_centiles) // 2
    for index in range(band_count):
        lower_centile = float(sorted_centiles[index])
        upper_centile = float(sorted_centiles[-(index + 1)])
        if lower_centile >= upper_centile:
            continue
        lower_entry = entry_map[lower_centile]
        upper_entry = entry_map[upper_centile]
        bands.append(
            CentileFanBand(
                lower_centile=lower_centile,
                upper_centile=upper_centile,
                x=np.asarray(lower_entry.x, dtype=np.float64),
                lower=np.asarray(lower_entry.y, dtype=np.float64),
                upper=np.asarray(upper_entry.y, dtype=np.float64),
                color_index=index,
            )
        )
    median_curve = None
    if include_median and 50.0 in entry_map:
        median_entry = entry_map[50.0]
        median_curve = CentileCurveEntry(
            centile=median_entry.centile,
            x=np.asarray(median_entry.x, dtype=np.float64),
            y=np.asarray(median_entry.y, dtype=np.float64),
        )
    return CentilesFanResult(
        family=centiles_result.family,
        xvar=centiles_result.xvar,
        centiles=centiles_result.centiles,
        bands=tuple(bands),
        median=median_curve,
        observed_x=centiles_result.observed_x,
        observed_y=centiles_result.observed_y,
        fixed_at=dict(centiles_result.fixed_at),
    )


centiles_fan = centiles_fan_data

__all__ = [
    "CentileFanBand",
    "CentilesFanResult",
    "centiles_fan",
    "centiles_fan_data",
]
