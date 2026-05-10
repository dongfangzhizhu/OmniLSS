"""R-aligned predictive distribution fitting interfaces.

R source reference:
- file: `gamlss/R/fitDistPred.R`
- functions: `fitDistPred`, `gamlssMLpred`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .families import FamilyDefinition
from .fitDist import _fit_dist_available_families
from .model import GAMLSSModel
from .predictAll_22_08_22 import PredictAllResult, predict_all


@dataclass(frozen=True)
class FitDistPredRow:
    """One staged `fitDistPred` ranking row."""

    family: str
    validation_global_deviance: float
    prediction_error: float
    df_fit: float


@dataclass(frozen=True)
class FitDistPredResult:
    """Structured staged `fitDistPred` payload."""

    type: str
    best_family: str
    best_model: GAMLSSModel
    rows: tuple[FitDistPredRow, ...]
    failed: tuple[str, ...]
    validation_size: int


@dataclass(frozen=True)
class GAMLSSMLPredResult:
    """Structured staged `gamlssMLpred` payload."""

    family: str
    model: GAMLSSModel
    validation_global_deviance: float
    validation_increment: np.ndarray
    prediction_error: float
    validation_residuals: np.ndarray | None
    validation_size: int


def fit_dist_pred_data(
    y: Sequence[float] | np.ndarray,
    type: str = "realAll",
    rand: Sequence[int] | np.ndarray | None = None,
    newdata: Mapping[str, Any] | None = None,
    extra: Sequence[str] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> FitDistPredResult:
    """R reference: `gamlss/R/fitDistPred.R::fitDistPred`.

    Staged behavior:
    - Fits currently supported staged families on a training subset or full `y`.
    - Scores candidate families by validation global deviance on held-out data.
    - Returns the best model together with ranked validation scores.
    """

    from .fitting import gamlss_ml

    if (rand is None) == (newdata is None):
        raise ValueError("exactly one of rand or newdata must be provided")

    available = _fit_dist_available_families()
    selected_type = str(type).strip()
    key = selected_type.lower()
    if key not in available:
        raise ValueError("type must be one of 'realAll', 'realline', 'realplus', 'real0to1', 'counts', or 'binom'")

    family_names = list(available[key])
    if extra is not None:
        family_names.extend(str(name).strip().upper() for name in extra if str(name).strip())
    family_names = list(dict.fromkeys(family_names))

    y_array = np.asarray(y, dtype=np.float64).ravel()
    if y_array.size == 0:
        raise ValueError("y must contain at least one observation")
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w is not None and w.size != y_array.size:
        raise ValueError("weights must have the same length as y")

    if rand is not None:
        split = np.asarray(rand)
        if split.shape[0] != y_array.shape[0]:
            raise ValueError("rand must have the same length as y")
        if not np.all(np.isin(split, [1, 2])):
            raise ValueError("rand values must be 1 or 2")
        train_mask = split == 1
        valid_mask = split == 2
        if not np.any(train_mask) or not np.any(valid_mask):
            raise ValueError("rand must contain both training (1) and validation (2) rows")
        train_y = y_array[train_mask]
        train_w = None if w is None else w[train_mask]
        validation_y = y_array[valid_mask]
        validation_data: dict[str, Any] = {"y": validation_y}
    else:
        train_y = y_array
        train_w = w
        if newdata is None or "y" not in newdata:
            raise ValueError("newdata must contain response column 'y'")
        validation_y = np.asarray(newdata["y"], dtype=np.float64).ravel()
        validation_data = dict(newdata)

    rows: list[FitDistPredRow] = []
    failed: list[str] = []
    best_model: GAMLSSModel | None = None
    best_score = float("inf")

    for family_name in family_names:
        try:
            fit = gamlss_ml(
                formula="y ~ 1",
                family=family_name,
                data={"y": train_y},
                weights=train_w,
            )
            predicted = predict_all(
                fit,
                newdata=validation_data,
                type="response",
                output="list",
                se_fit=False,
            )
            values = predicted.values if isinstance(predicted, PredictAllResult) else dict(predicted)
            deviance_kwargs: dict[str, Any] = {"y": np.asarray(validation_y, dtype=np.float64)}
            for parameter in fit.par:
                deviance_kwargs[parameter] = np.asarray(values[parameter], dtype=np.float64)
            deviance_increment = np.asarray(fit.family.g_dev_inc(**deviance_kwargs), dtype=np.float64)
            validation_global_deviance = float(np.sum(deviance_increment))
            prediction_error = float(validation_global_deviance / validation_y.size)
        except Exception:
            failed.append(family_name)
            continue
        row = FitDistPredRow(
            family=str(getattr(fit.family, "name", family_name)),
            validation_global_deviance=validation_global_deviance,
            prediction_error=prediction_error,
            df_fit=float(fit.df_fit),
        )
        rows.append(row)
        if validation_global_deviance < best_score:
            best_score = validation_global_deviance
            best_model = fit

    if best_model is None:
        raise ValueError("no candidate family could be fitted")

    ordered_rows = tuple(sorted(rows, key=lambda row: (row.validation_global_deviance, row.family)))
    return FitDistPredResult(
        type=selected_type,
        best_family=str(getattr(best_model.family, "name", best_model.family)),
        best_model=best_model,
        rows=ordered_rows,
        failed=tuple(failed),
        validation_size=int(validation_y.size),
    )


def fit_dist_pred(
    y: Sequence[float] | np.ndarray,
    type: str = "realAll",
    rand: Sequence[int] | np.ndarray | None = None,
    newdata: Mapping[str, Any] | None = None,
    extra: Sequence[str] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> FitDistPredResult:
    """R reference: `gamlss/R/fitDistPred.R::fitDistPred`."""

    return fit_dist_pred_data(
        y=y,
        type=type,
        rand=rand,
        newdata=newdata,
        extra=extra,
        weights=weights,
    )


def gamlss_ml_pred_data(
    y: Sequence[float] | np.ndarray,
    family: FamilyDefinition | str | None = None,
    rand: Sequence[int] | np.ndarray | None = None,
    newdata: Mapping[str, Any] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> GAMLSSMLPredResult:
    """R reference: `gamlss/R/fitDistPred.R::gamlssMLpred`.

    Staged behavior:
    - Fits a single-family intercept-only `gamlss_ml()` model.
    - Scores the fit by validation global deviance on held-out data.
    - Returns validation deviance increments, prediction error, and residuals.
    """

    from .fitting import gamlss_ml

    if (rand is None) == (newdata is None):
        raise ValueError("exactly one of rand or newdata must be provided")

    y_array = np.asarray(y, dtype=np.float64).ravel()
    if y_array.size == 0:
        raise ValueError("y must contain at least one observation")
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w is not None and w.size != y_array.size:
        raise ValueError("weights must have the same length as y")

    if rand is not None:
        split = np.asarray(rand)
        if split.shape[0] != y_array.shape[0]:
            raise ValueError("rand must have the same length as y")
        if not np.all(np.isin(split, [1, 2])):
            raise ValueError("rand values must be 1 or 2")
        train_mask = split == 1
        valid_mask = split == 2
        if not np.any(train_mask) or not np.any(valid_mask):
            raise ValueError("rand must contain both training (1) and validation (2) rows")
        train_y = y_array[train_mask]
        train_w = None if w is None else w[train_mask]
        validation_y = y_array[valid_mask]
        validation_data: dict[str, Any] = {"y": validation_y}
    else:
        train_y = y_array
        train_w = w
        if newdata is None or "y" not in newdata:
            raise ValueError("newdata must contain response column 'y'")
        validation_y = np.asarray(newdata["y"], dtype=np.float64).ravel()
        validation_data = dict(newdata)

    fit = gamlss_ml(
        formula="y ~ 1",
        family=family,
        data={"y": train_y},
        weights=train_w,
    )
    predicted = predict_all(
        fit,
        newdata=validation_data,
        type="response",
        output="list",
        se_fit=False,
    )
    values = predicted.values if isinstance(predicted, PredictAllResult) else dict(predicted)
    deviance_kwargs: dict[str, Any] = {"y": validation_y}
    for parameter in fit.par:
        deviance_kwargs[parameter] = np.asarray(values[parameter], dtype=np.float64)
    validation_increment = np.asarray(fit.family.g_dev_inc(**deviance_kwargs), dtype=np.float64)
    validation_global_deviance = float(np.sum(validation_increment))
    prediction_error = float(validation_global_deviance / validation_y.size)

    validation_residuals: np.ndarray | None = None
    if fit.rqres is not None:
        rqres_kwargs = {"y": validation_y}
        for parameter in fit.par:
            rqres_kwargs[parameter] = np.asarray(values[parameter], dtype=np.float64)
        try:
            validation_residuals = np.asarray(fit.rqres(**rqres_kwargs), dtype=np.float64)
        except Exception:
            validation_residuals = None

    return GAMLSSMLPredResult(
        family=str(getattr(fit.family, "name", fit.family)),
        model=fit,
        validation_global_deviance=validation_global_deviance,
        validation_increment=validation_increment,
        prediction_error=prediction_error,
        validation_residuals=validation_residuals,
        validation_size=int(validation_y.size),
    )


def gamlss_ml_pred(
    y: Sequence[float] | np.ndarray,
    family: FamilyDefinition | str | None = None,
    rand: Sequence[int] | np.ndarray | None = None,
    newdata: Sequence[float] | np.ndarray | None = None,
) -> GAMLSSMLPredResult:
    """R reference: `gamlss/R/fitDistPred.R::gamlssMLpred`."""

    return gamlss_ml_pred_data(y=y, family=family, rand=rand, newdata=newdata)

fitDistPred = fit_dist_pred
gamlssMLpred = gamlss_ml_pred

__all__ = [
    "FitDistPredResult",
    "FitDistPredRow",
    "GAMLSSMLPredResult",
    "fitDistPred",
    "fit_dist_pred",
    "fit_dist_pred_data",
    "gamlssMLpred",
    "gamlss_ml_pred",
    "gamlss_ml_pred_data",
]
