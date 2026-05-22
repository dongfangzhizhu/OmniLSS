"""R-aligned parameter diagnostics payload builder.

R reference: `gamlss/R/par_plot-21-5-13.R::par.plot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class ParPlotSeries:
    parameter: str
    fitted: np.ndarray
    linear_predictor: np.ndarray


@dataclass(frozen=True)
class ParPlotResult:
    y: np.ndarray
    series: tuple[ParPlotSeries, ...]


def par_plot(object: GAMLSSModel, parameters: tuple[str, ...] | None = None) -> ParPlotResult:
    """Return data payload for parameter diagnostics plotting."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")

    selected = tuple(parameters or object.par)
    y = np.asarray(object.y, dtype=np.float64)
    series: list[ParPlotSeries] = []
    for p in selected:
        if p not in object.par:
            raise ValueError(f"{p!r} is not a parameter in the object")
        series.append(
            ParPlotSeries(
                parameter=p,
                fitted=np.asarray(object.fitted_values[p], dtype=np.float64),
                linear_predictor=np.asarray(object.linear_predictors[p], dtype=np.float64),
            )
        )
    return ParPlotResult(y=y, series=tuple(series))


__all__ = ["par_plot", "ParPlotResult", "ParPlotSeries"]
