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
    response_residual: np.ndarray
    weighted_response_residual: np.ndarray


@dataclass(frozen=True)
class ParPlotResult:
    y: np.ndarray
    weights: np.ndarray
    series: tuple[ParPlotSeries, ...]


def par_plot(
    object: GAMLSSModel, parameters: tuple[str, ...] | None = None
) -> ParPlotResult:
    """Return data payload for parameter diagnostics plotting."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")

    selected = tuple(parameters or object.par)
    y = np.asarray(object.y, dtype=np.float64)
    if object.weights is None:
        weights = np.ones_like(y, dtype=np.float64)
    else:
        weights = np.asarray(object.weights, dtype=np.float64)
        if weights.shape != y.shape:
            raise ValueError("weights must have same shape as response")

    series: list[ParPlotSeries] = []
    for p in selected:
        if p not in object.par:
            raise ValueError(f"{p!r} is not a parameter in the object")
        fitted = np.asarray(object.fitted_values[p], dtype=np.float64)
        if fitted.shape != y.shape:
            raise ValueError(
                f"fitted values for {p!r} must have same shape as response"
            )
        linear_predictor = np.asarray(object.linear_predictors[p], dtype=np.float64)
        response_residual = y - fitted
        series.append(
            ParPlotSeries(
                parameter=p,
                fitted=fitted,
                linear_predictor=linear_predictor,
                response_residual=response_residual,
                weighted_response_residual=weights * response_residual,
            )
        )
    return ParPlotResult(y=y, weights=weights, series=tuple(series))


__all__ = ["par_plot", "ParPlotResult", "ParPlotSeries"]
