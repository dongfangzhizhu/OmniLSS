"""R-aligned fitted plot interfaces.

R source reference:
- file: `gamlss/R/fitted-plot.R`
- functions: fitted plot helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .model import GAMLSSModel


@dataclass(frozen=True)
class FittedPlotEntry:
    """One staged fitted-plot data bundle for a single distribution parameter."""

    parameter: str
    x: np.ndarray
    fit: np.ndarray
    se_fit: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None


@dataclass(frozen=True)
class FittedPlotResult:
    """Structured staged `fitted.plot` data payload."""

    xvar: str
    type: str
    entries: tuple[FittedPlotEntry, ...]


def fitted_plot_data(
    object: GAMLSSModel,
    xvar: str,
    parameters: list[str] | tuple[str, ...] | None = None,
    type: str = "response",
    se_fit: bool = True,
    level: float = 0.95,
) -> FittedPlotResult:
    """R reference: `gamlss/R/fitted-plot.R`.

    Staged behavior:
    - Returns fitted-curve data instead of drawing parameter panels.
    - Uses observed data order, sorted by the requested `xvar`.
    - Optionally returns pointwise standard errors and intervals.
    """

    from .methods import _require_gamlss_method
    from .predictAll_22_08_22 import PredictAllResult, predict_all

    _require_gamlss_method(object)
    selected_type = str(type).strip().lower()
    if selected_type not in {"response", "link"}:
        raise ValueError("type must be 'response' or 'link'")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("fitted plot data requires call['data']")
    if xvar not in call_data:
        raise KeyError(f"xvar {xvar!r} not found in data")

    requested = [str(parameter).strip().lower() for parameter in (parameters or object.par) if str(parameter).strip()]
    if not requested:
        raise ValueError("no parameters available for fitted plot data")
    invalid = [parameter for parameter in requested if parameter not in object.par]
    if invalid:
        raise ValueError(f"parameters not found in object: {', '.join(invalid)}")

    x_values = np.asarray(call_data[xvar], dtype=np.float64).ravel()
    order = np.argsort(x_values)
    ordered_x = x_values[order]
    alpha = (1.0 - level) / 2.0
    z_value = NormalDist().inv_cdf(1.0 - alpha)

    if se_fit:
        prediction = predict_all(object, type=selected_type, output="list", se_fit=True)
        values = prediction.values if isinstance(prediction, PredictAllResult) else prediction
    else:
        prediction = predict_all(object, type=selected_type, output="list", se_fit=False)
        values = prediction.values if isinstance(prediction, PredictAllResult) else prediction

    entries: list[FittedPlotEntry] = []
    link_map = getattr(object.family, "links", {}) or {}
    for parameter in requested:
        value = values[parameter]
        if isinstance(value, dict):
            fit_array = np.asarray(value["fit"], dtype=np.float64).ravel()
            se_array = np.asarray(value["se.fit"], dtype=np.float64).ravel()
        else:
            fit_array = np.asarray(value, dtype=np.float64).ravel()
            se_array = None
        fit_array = np.nan_to_num(fit_array, nan=0.0, posinf=1e12, neginf=-1e12)
        if se_array is not None:
            se_array = np.nan_to_num(se_array, nan=np.nan, posinf=1e12, neginf=np.nan)
        if fit_array.size == 1 and x_values.size > 1:
            fit_array = np.full(x_values.shape, fit_array.item(), dtype=np.float64)
        if se_array is not None and se_array.size == 1 and x_values.size > 1:
            se_array = np.full(x_values.shape, se_array.item(), dtype=np.float64)
        link_name = str(link_map.get(parameter, "")).lower()
        if selected_type == "response" and link_name == "log":
            fit_array = np.maximum(fit_array, np.finfo(np.float64).eps)
        if selected_type == "response" and link_name == "logshift2":
            fit_array = np.maximum(fit_array, 2.0 + np.finfo(np.float64).eps)
        if se_array is not None:
            se_array = np.where(np.isfinite(se_array), np.maximum(se_array, 0.0), np.nan)
        ordered_fit = fit_array[order]
        if se_array is None:
            ordered_se = None
            lower = None
            upper = None
        else:
            ordered_se = se_array[order]
            lower = ordered_fit - z_value * ordered_se
            upper = ordered_fit + z_value * ordered_se
        entries.append(
            FittedPlotEntry(
                parameter=parameter,
                x=ordered_x,
                fit=ordered_fit,
                se_fit=ordered_se,
                lower=lower,
                upper=upper,
            )
        )

    return FittedPlotResult(
        xvar=str(xvar),
        type=selected_type,
        entries=tuple(entries),
    )


fitted_plot = fitted_plot_data

__all__ = [
    "FittedPlotEntry",
    "FittedPlotResult",
    "fitted_plot",
    "fitted_plot_data",
]
