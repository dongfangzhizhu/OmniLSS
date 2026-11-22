"""R-aligned fitted plot interfaces.

R source reference:
- file: `gamlss/R/fitted-plot.R`
- functions: fitted plot helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

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

    requested = [
        str(parameter).strip().lower()
        for parameter in (parameters or object.par)
        if str(parameter).strip()
    ]
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
        values = (
            prediction.values
            if isinstance(prediction, PredictAllResult)
            else prediction
        )
    else:
        prediction = predict_all(
            object, type=selected_type, output="list", se_fit=False
        )
        values = (
            prediction.values
            if isinstance(prediction, PredictAllResult)
            else prediction
        )

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
        if selected_type == "response" and link_name == "logshiftto2":
            fit_array = np.maximum(fit_array, 2.0 + np.finfo(np.float64).eps)
        if se_array is not None:
            se_array = np.where(
                np.isfinite(se_array), np.maximum(se_array, 0.0), np.nan
            )
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


@dataclass(frozen=True)
class FittedPlotComparisonResult:
    """Structured result for R ``gamlss::fittedPlot`` (single or multi-model)."""

    x: np.ndarray
    xlab: str
    parameters: dict[str, np.ndarray]
    model_count: int
    figure: Any = None


def fittedPlot(
    object: GAMLSSModel,
    *others: GAMLSSModel,
    x: Any = None,
    color: bool = True,
    line_type: bool = False,
    xlab: str | None = None,
    plot: bool = False,
) -> FittedPlotComparisonResult:
    """R reference: ``gamlss/R/fitted-plot.R::fittedPlot``.

    Sorts fitted distribution parameters (``mu``, ``sigma``, ``nu``, ``tau``)
    by ``x`` for one or more fitted ``GAMLSSModel`` objects, snapping
    near-constant curves (``|v - v[0]| < 1e-4``) identically to R.
    """
    from .operations import fitted, is_gamlss

    models = (object, *others)
    if not all(is_gamlss(m) for m in models):
        raise TypeError(
            "all objects passed to fittedPlot must be GAMLSSModel instances"
        )
    if x is None:
        raise ValueError("The x-variable argument is not specified")

    if isinstance(x, str):
        call_data = None if object.call is None else object.call.get("data")
        if call_data is None or x not in call_data:
            raise KeyError(f"x column {x!r} not found in object.call['data']")
        x_arr = np.asarray(call_data[x], dtype=np.float64).ravel()
        label = x if xlab is None else str(xlab)
    else:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        label = "x" if xlab is None else str(xlab)

    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]

    param_columns: dict[str, list[np.ndarray]] = {}
    for model in models:
        active = tuple(model.parameters or model.par)
        for pname in ("mu", "sigma", "nu", "tau"):
            if pname not in active:
                continue
            vals = np.asarray(fitted(model, pname), dtype=np.float64).ravel()
            if vals.size == 1 and x_arr.size > 1:
                vals = np.full(x_arr.shape, vals.item(), dtype=np.float64)
            if vals.size != x_arr.size:
                raise ValueError(
                    f"fitted {pname!r} length ({vals.size}) does not match x length ({x_arr.size})"
                )
            ordered = vals[order]
            # R fitted-plot.R:18-19 / 156-157 constant snapping
            if ordered.size > 0 and np.all(np.abs(ordered - ordered[0]) < 1e-4):
                ordered = np.full(ordered.shape, ordered[0], dtype=np.float64)
            param_columns.setdefault(pname, []).append(ordered)

    matrices: dict[str, np.ndarray] = {
        pname: np.column_stack(cols) for pname, cols in param_columns.items()
    }

    figure = None
    if plot and matrices:
        try:
            import matplotlib.pyplot as plt

            names = list(matrices.keys())
            layout = {1: (1, 1), 2: (2, 1), 3: (3, 1)}.get(len(names), (2, 2))
            fig, axes = plt.subplots(*layout, squeeze=False)
            panel_labels = ("(a)", "(b)", "(c)", "(d)")
            for idx, pname in enumerate(names):
                ax = axes[idx // layout[1]][idx % layout[1]]
                mat = matrices[pname]
                for col in range(mat.shape[1]):
                    linestyle = "--" if (line_type and col % 2 == 1) else "-"
                    plot_kwargs = {"linestyle": linestyle}
                    if not color:
                        plot_kwargs["color"] = "black"
                    ax.plot(x_sorted, mat[:, col], **plot_kwargs)
                ax.set_xlabel(label)
                ax.set_ylabel(pname)
                ax.set_title(panel_labels[idx] if idx < len(panel_labels) else pname)
            figure = fig
        except ImportError:
            figure = None

    return FittedPlotComparisonResult(
        x=x_sorted,
        xlab=label,
        parameters=matrices,
        model_count=len(models),
        figure=figure,
    )


__all__ = [
    "FittedPlotComparisonResult",
    "FittedPlotEntry",
    "FittedPlotResult",
    "fittedPlot",
    "fitted_plot",
    "fitted_plot_data",
]
