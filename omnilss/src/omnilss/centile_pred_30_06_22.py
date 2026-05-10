"""R-aligned centile prediction interfaces.

R source reference:
- file: `gamlss/R/centile-pred_30_06_22.R`
- functions: `centiles.pred`
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .model import GAMLSSModel


@dataclass(frozen=True)
class CentilePredEntry:
    """One staged `centiles.pred` output column."""

    label: str
    values: np.ndarray
    probability: float | None = None


@dataclass(frozen=True)
class CentilePredResult:
    """Structured staged `centiles.pred` payload."""

    type: str
    xname: str
    x: np.ndarray
    entries: tuple[CentilePredEntry, ...]
    y: np.ndarray | None
    calibration_applied: bool
    fixed_at: dict[str, float]
    approximation: str


def centile_pred_data(
    object: GAMLSSModel,
    type: str = "centiles",
    xname: str | None = None,
    xvalues: Sequence[float] | None = None,
    power: float | None = None,
    yval: Sequence[float] | None = None,
    cent: Sequence[float] = (0.4, 2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0, 99.6),
    dev: Sequence[float] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
    calibration: bool = False,
    fixed_at: dict[str, float] | None = None,
    probability_grid_size: int = 401,
) -> CentilePredResult:
    """R reference: `gamlss/R/centile-pred_30_06_22.R::centiles.pred`.

    Staged behavior:
    - Returns structured prediction payloads instead of plotting.
    - Supports `centiles`, `z-scores`, and `standard-centiles`.
    - Uses family-aware staged quantiles and an interpolated probability grid
      for generic z-score inversion.
    """

    from .methods import (
        _family_quantile_values,
        _ordered_residuals,
        _predict_parameter_frame,
        _require_gamlss_method,
    )

    _require_gamlss_method(object)
    if xname is None or not str(xname).strip():
        raise ValueError("xname argument is required")
    if xvalues is None:
        raise ValueError("xvalues argument is required")
    selected_type = str(type).strip().lower()
    if selected_type not in {"centiles", "z-scores", "standard-centiles"}:
        raise ValueError("type must be one of 'centiles', 'z-scores', or 'standard-centiles'")

    x_output = np.asarray(xvalues, dtype=np.float64).ravel()
    x_input = np.asarray(np.power(x_output, power), dtype=np.float64) if power is not None else x_output.copy()
    predicted, chosen_fixed = _predict_parameter_frame(object, str(xname), x_input, fixed_at=fixed_at)
    family_name = getattr(object.family, "name", str(object.family))
    approximation = "family-aware staged quantile approximation"
    normal = NormalDist()

    if selected_type == "centiles":
        cent_values = np.asarray(tuple(float(value) for value in cent), dtype=np.float64)
        if cent_values.size == 0:
            raise ValueError("cent must contain at least one centile")
        if np.any((cent_values <= 0.0) | (cent_values >= 100.0)):
            raise ValueError("cent must lie strictly between 0 and 100")
        if calibration:
            z = np.quantile(_ordered_residuals(object), cent_values / 100.0)
            cent_values = np.array([100.0 * normal.cdf(float(value)) for value in z], dtype=np.float64)
            eps_pct = 100.0 * np.finfo(np.float64).eps
            cent_values = np.clip(cent_values, eps_pct, 100.0 - eps_pct)
        probs = tuple(float(value / 100.0) for value in cent_values)
        quantiles = _family_quantile_values(
            family_name,
            object.par,
            predicted["mu"],
            predicted["sigma"],
            predicted["nu"],
            predicted["tau"],
            probs,
        )
        entries = tuple(
            CentilePredEntry(
                label=str(round(float(centile), 6)).rstrip("0").rstrip("."),
                values=np.asarray(values, dtype=np.float64).ravel(),
                probability=float(probability),
            )
            for centile, probability, values in zip(cent_values, probs, quantiles)
        )
        return CentilePredResult(
            type=selected_type,
            xname=str(xname),
            x=np.asarray(x_output, dtype=np.float64),
            entries=entries,
            y=None,
            calibration_applied=bool(calibration),
            fixed_at=chosen_fixed,
            approximation=approximation,
        )

    if selected_type == "standard-centiles":
        dev_values = np.asarray(tuple(float(value) for value in dev), dtype=np.float64)
        probs = tuple(float(normal.cdf(float(value))) for value in dev_values)
        quantiles = _family_quantile_values(
            family_name,
            object.par,
            predicted["mu"],
            predicted["sigma"],
            predicted["nu"],
            predicted["tau"],
            probs,
        )
        entries = tuple(
            CentilePredEntry(
                label=str(int(value)) if float(value).is_integer() else str(float(value)),
                values=np.asarray(values, dtype=np.float64).ravel(),
                probability=float(probability),
            )
            for value, probability, values in zip(dev_values, probs, quantiles)
        )
        return CentilePredResult(
            type=selected_type,
            xname=str(xname),
            x=np.asarray(x_output, dtype=np.float64),
            entries=entries,
            y=None,
            calibration_applied=False,
            fixed_at=chosen_fixed,
            approximation=approximation,
        )

    if calibration:
        raise ValueError("calibration is not implemented for z-scores")
    if yval is None:
        raise ValueError("yval must be supplied for z-scores")
    y_values = np.asarray(yval, dtype=np.float64).ravel()
    if y_values.shape != x_output.shape:
        raise ValueError("length of xvalues and yval is not the same")
    probability_grid = np.linspace(1e-3, 1.0 - 1e-3, int(probability_grid_size), dtype=np.float64)
    quantile_grid = _family_quantile_values(
        family_name,
        object.par,
        predicted["mu"],
        predicted["sigma"],
        predicted["nu"],
        predicted["tau"],
        tuple(float(value) for value in probability_grid),
    )
    quantile_matrix = np.column_stack([np.asarray(values, dtype=np.float64).ravel() for values in quantile_grid])
    z_scores = np.empty_like(y_values, dtype=np.float64)
    for idx in range(y_values.size):
        curve = np.asarray(quantile_matrix[idx], dtype=np.float64)
        order = np.argsort(curve)
        sorted_curve = curve[order]
        sorted_probs = probability_grid[order]
        clipped_prob = np.interp(float(y_values[idx]), sorted_curve, sorted_probs, left=float(sorted_probs[0]), right=float(sorted_probs[-1]))
        clipped_prob = float(np.clip(clipped_prob, 1e-6, 1.0 - 1e-6))
        z_scores[idx] = normal.inv_cdf(clipped_prob)
    return CentilePredResult(
        type=selected_type,
        xname=str(xname),
        x=np.asarray(x_output, dtype=np.float64),
        entries=(CentilePredEntry(label="z_scores", values=z_scores),),
        y=y_values,
        calibration_applied=False,
        fixed_at=chosen_fixed,
        approximation=f"{approximation}; z-scores via interpolated probability grid",
    )


centiles_pred = centile_pred_data


__all__ = [
    "CentilePredEntry",
    "CentilePredResult",
    "centiles_pred",
    "centile_pred_data",
]
