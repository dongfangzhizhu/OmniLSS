"""R-aligned histogram-versus-distribution interfaces.

R source reference:
- file: `gamlss/R/HistDist-03-10-13.R`
- functions: `histDist`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .distributions import resolve_family
from .operations import fitted
from .pdfplot import _staged_density_curve
from .plot import _kernel_density


@dataclass(frozen=True)
class HistDistResult:
    """Structured staged `histDist` payload."""

    family: str
    distribution_type: str
    histogram_x: np.ndarray
    histogram_y: np.ndarray
    fitted_x: np.ndarray
    fitted_y: np.ndarray
    fitted_parameters: dict[str, float]
    density_x: np.ndarray | None
    density_y: np.ndarray | None
    used_weights: bool
    approximation: str


def hist_dist_data(
    y: Sequence[float] | np.ndarray,
    family: str | Any = "NO",
    freq: Sequence[float] | np.ndarray | None = None,
    density: bool = False,
    nbins: int = 10,
    xlim: tuple[float, float] | None = None,
) -> HistDistResult:
    """R reference: `gamlss/R/HistDist-03-10-13.R::histDist`.

    Staged behavior:
    - Fits an intercept-only staged `gamlss_ml()` model for the requested family.
    - Returns histogram/barplot coordinates plus the fitted density/pmf curve.
    - Optionally adds a kernel density estimate for the observed sample on the
      continuous path.
    """

    from .fitting import gamlss_ml

    resolved_family = resolve_family(family)
    distribution_type = str(getattr(resolved_family, "type", "Continuous"))
    y_array = np.asarray(y, dtype=np.float64).ravel()
    if y_array.size == 0:
        raise ValueError("y must contain at least one observation")
    weights = None if freq is None else np.asarray(freq, dtype=np.float64).ravel()
    if weights is not None and weights.size != y_array.size:
        raise ValueError("freq must have the same length as y")

    fit = gamlss_ml(
        formula="y ~ 1",
        family=resolved_family,
        data={"y": y_array},
        weights=weights,
    )
    parameter_values = {
        parameter: float(np.asarray(fitted(fit, parameter), dtype=np.float64).ravel()[0])
        for parameter in fit.par
    }

    lower = float(np.min(y_array) if xlim is None else xlim[0])
    upper = float(np.max(y_array) if xlim is None else xlim[1])
    if upper <= lower:
        upper = lower + 1.0

    if distribution_type.lower() == "discrete":
        support = np.arange(math.floor(lower), math.ceil(upper) + 1, dtype=np.float64)
        if support.size == 0:
            support = np.arange(math.floor(np.min(y_array)), math.ceil(np.max(y_array)) + 1, dtype=np.float64)
        count_weights = np.ones_like(y_array) if weights is None else weights
        totals = np.zeros_like(support, dtype=np.float64)
        for index, value in enumerate(support):
            totals[index] = float(np.sum(count_weights[np.isclose(y_array, value)]))
        total_weight = float(np.sum(count_weights))
        histogram_y = totals / total_weight if total_weight > 0.0 else totals
        fitted_y = _staged_density_curve(resolved_family, support, parameter_values)
        density_x = None
        density_y = None
        histogram_x = support
        fitted_x = support
    else:
        hist_weights = None if weights is None else weights
        histogram_y, edges = np.histogram(y_array, bins=int(nbins), range=(lower, upper), density=True, weights=hist_weights)
        histogram_x = 0.5 * (edges[:-1] + edges[1:])
        fitted_x = np.linspace(lower, upper, 201, dtype=np.float64)
        fitted_y = _staged_density_curve(resolved_family, fitted_x, parameter_values)
        if density:
            expanded = y_array if weights is None else np.repeat(y_array, np.maximum(np.rint(weights).astype(int), 0))
            if expanded.size >= 2:
                density_x, density_y = _kernel_density(np.asarray(expanded, dtype=np.float64))
            else:
                density_x = None
                density_y = None
        else:
            density_x = None
            density_y = None

    return HistDistResult(
        family=str(resolved_family.name),
        distribution_type=distribution_type,
        histogram_x=np.asarray(histogram_x, dtype=np.float64),
        histogram_y=np.asarray(histogram_y, dtype=np.float64),
        fitted_x=np.asarray(fitted_x, dtype=np.float64),
        fitted_y=np.asarray(fitted_y, dtype=np.float64),
        fitted_parameters=parameter_values,
        density_x=None if density_x is None else np.asarray(density_x, dtype=np.float64),
        density_y=None if density_y is None else np.asarray(density_y, dtype=np.float64),
        used_weights=weights is not None,
        approximation="staged intercept-only fit plus fitted density/histogram coordinates",
    )


histDist = hist_dist_data

__all__ = [
    "HistDistResult",
    "histDist",
    "hist_dist_data",
]
