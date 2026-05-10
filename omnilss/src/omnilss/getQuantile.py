"""R-aligned quantile curve interfaces.

R source reference:
- file: `gamlss/R/getQuantile.R`
- functions: `getQuantile`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import GAMLSSModel


@dataclass(frozen=True)
class QuantileCurveEntry:
    """One staged quantile-curve data bundle for a single probability level."""

    probability: float
    x: np.ndarray
    quantile: np.ndarray


@dataclass(frozen=True)
class QuantileCurveResult:
    """Structured staged `getQuantile` style payload."""

    family: str
    xvar: str
    probabilities: tuple[float, ...]
    entries: tuple[QuantileCurveEntry, ...]
    fixed_at: dict[str, float]
    approximation: str


def get_quantile_data(
    object: GAMLSSModel,
    xvar: str,
    probabilities: list[float] | tuple[float, ...] = (0.1, 0.5, 0.9),
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
    x_limits: tuple[float, float] | None = None,
) -> QuantileCurveResult:
    """R reference: `gamlss/R/getQuantile.R::getQuantile`.

    Staged behavior:
    - Builds quantile curves over one selected covariate.
    - Fixes all remaining covariates at median or last observed values.
    - Uses family-aware staged quantile approximations based on predicted parameters.
    """

    from .methods import (
        _family_quantile_values,
        _predict_parameter_frame,
        _require_gamlss_method,
        _rhs_terms,
    )

    _require_gamlss_method(object)
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    how_name = str(how).strip().lower()
    if how_name not in {"median", "last"}:
        raise ValueError("how must be 'median' or 'last'")

    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("quantile data requires call['data']")
    if xvar not in call_data:
        raise KeyError(f"xvar {xvar!r} not found in data")

    probs = tuple(float(probability) for probability in probabilities)
    if not probs:
        raise ValueError("at least one probability is required")
    if any(probability <= 0.0 or probability >= 1.0 for probability in probs):
        raise ValueError("probabilities must lie strictly between 0 and 1")

    x_values = np.asarray(call_data[xvar], dtype=np.float64).ravel()
    x_lower = float(np.nanmin(x_values))
    x_upper = float(np.nanmax(x_values))
    if x_limits is not None:
        x_lower = float(x_limits[0])
        x_upper = float(x_limits[1])
        if not np.isfinite(x_lower) or not np.isfinite(x_upper) or x_lower >= x_upper:
            raise ValueError("x_limits must contain two finite increasing values")
    x_grid = np.linspace(x_lower, x_upper, int(n_points), dtype=np.float64)
    fixed_values = {} if fixed_at is None else {str(key): float(value) for key, value in fixed_at.items()}
    if how_name == "last":
        predictor_names: set[str] = set()
        for parameter in object.par:
            formula_text = object.formulas.get(parameter)
            if formula_text is None:
                continue
            predictor_names.update(_rhs_terms(formula_text))
        predictor_names.discard(xvar)
        predictor_names.discard(".")
        for name in sorted(predictor_names):
            if name not in fixed_values:
                source = np.asarray(call_data[name], dtype=np.float64).ravel()
                fixed_values[name] = float(source[-1])

    predicted, chosen_fixed = _predict_parameter_frame(object, xvar, x_grid, fixed_at=fixed_values)
    family_name = getattr(object.family, "name", str(object.family))
    quantile_values = _family_quantile_values(
        family_name,
        object.par,
        predicted["mu"],
        predicted["sigma"],
        predicted["nu"],
        predicted["tau"],
        probs,
    )

    entries: list[QuantileCurveEntry] = []
    for probability, quantile in zip(probs, quantile_values):
        entries.append(
            QuantileCurveEntry(
                probability=float(probability),
                x=x_grid.copy(),
                quantile=np.asarray(quantile, dtype=np.float64).ravel(),
            )
        )

    return QuantileCurveResult(
        family=family_name,
        xvar=str(xvar),
        probabilities=probs,
        entries=tuple(entries),
        fixed_at=chosen_fixed,
        approximation="family-aware staged quantile approximation",
    )


getQuantile = get_quantile_data

__all__ = [
    "QuantileCurveEntry",
    "QuantileCurveResult",
    "getQuantile",
    "get_quantile_data",
]
