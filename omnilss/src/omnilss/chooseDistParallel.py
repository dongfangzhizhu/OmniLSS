"""R-aligned distribution comparison interfaces.

R source reference:
- file: `gamlss/R/chooseDistParallel.R`
- functions: `chooseDist`, `chooseDistPred`, `getOrder`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .fitDist import _fit_dist_available_families
from .model import GAMLSSModel
from .operations import formula, gaic, is_gamlss
from .predictAll_22_08_22 import PredictAllResult, predict_all


@dataclass(frozen=True)
class ChooseDistResult:
    """Structured staged `chooseDist` payload."""

    type: str
    penalties: tuple[float, ...]
    families: tuple[str, ...]
    matrix: np.ndarray
    minima: dict[float, str]
    failed: tuple[str, ...]


@dataclass(frozen=True)
class ChooseDistPredResult:
    """Structured staged `chooseDistPred` payload."""

    type: str
    families: tuple[str, ...]
    scores: np.ndarray
    best_family: str
    failed: tuple[str, ...]
    validation_size: int


@dataclass(frozen=True)
class ChooseDistOrderRow:
    """One staged `getOrder` ranking row."""

    family: str
    value: float


@dataclass(frozen=True)
class ChooseDistOrderResult:
    """Structured staged `getOrder` payload."""

    column_index: int
    penalty: float
    rows: tuple[ChooseDistOrderRow, ...]


def _require_gamlss_method(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _subset_mapping_rows(data: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != mask.shape[0]:
            subset[key] = value
            continue
        subset[key] = array[mask]
    return subset


def choose_dist_data(
    object: GAMLSSModel,
    k: Sequence[float] = (2.0, 3.84),
    type: str = "realAll",
    extra: Sequence[str] | None = None,
) -> ChooseDistResult:
    """R reference: `gamlss/R/chooseDistParallel.R::chooseDist`."""

    from .fitting import gamlss

    _require_gamlss_method(object)
    penalties = tuple(float(value) for value in k)
    if not penalties:
        raise ValueError("k must contain at least one penalty value")

    available = _fit_dist_available_families()
    selected_type = str(type).strip()
    key = selected_type.lower()
    if key == "extra" and not extra:
        raise ValueError("extra families must be provided when type='extra'")
    if key not in available and key != "extra":
        raise ValueError("type must be one of 'realAll', 'realline', 'realplus', 'real0to1', 'counts', 'binom', or 'extra'")

    family_names = list(extra if key == "extra" else available[key])
    if extra is not None and key != "extra":
        family_names.extend(str(name).strip().upper() for name in extra if str(name).strip())
    family_names = list(dict.fromkeys(str(name).strip().upper() for name in family_names if str(name).strip()))

    call = object.call or {}
    data = call.get("data")
    if data is None:
        raise ValueError("choose_dist_data requires call['data']")
    response = object.terms.get("mu", {}).get("response")
    if response is None:
        raise ValueError("choose_dist_data requires a stored response name")

    formula_text = str(call.get("formula", formula(object, what="mu")))
    sigma_formula = str(call.get("sigma_formula", formula(object, what="sigma"))) if "sigma" in object.par else "~1"
    parameter_formulas = dict(call.get("parameter_formulas", {}))
    method = str(call.get("method", object.additional_slots.get("method", "RS")))
    weights = call.get("weights", object.weights)

    matrix = np.full((len(family_names), len(penalties)), np.nan, dtype=np.float64)
    failed: list[str] = []
    for row_index, family_name in enumerate(family_names):
        try:
            fitted_model = gamlss(
                formula=formula_text,
                family=family_name,
                data=data,
                sigma_formula=sigma_formula,
                parameter_formulas=parameter_formulas if parameter_formulas else None,
                method=method,
                weights=weights,
            )
        except Exception:
            failed.append(family_name)
            continue
        for column_index, penalty in enumerate(penalties):
            matrix[row_index, column_index] = float(gaic(fitted_model, k=penalty))

    minima: dict[float, str] = {}
    for column_index, penalty in enumerate(penalties):
        column = matrix[:, column_index]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue
        best_index = int(np.nanargmin(column))
        minima[penalty] = family_names[best_index]

    return ChooseDistResult(
        type=selected_type,
        penalties=penalties,
        families=tuple(family_names),
        matrix=matrix,
        minima=minima,
        failed=tuple(failed),
    )


def get_order(
    obj: ChooseDistResult,
    column: int | float | str = 1,
) -> ChooseDistOrderResult:
    """R reference: `gamlss/R/chooseDistParallel.R::getOrder`."""

    if not isinstance(obj, ChooseDistResult):
        raise ValueError("get_order expects a ChooseDistResult produced by choose_dist_data()")
    matrix = np.asarray(obj.matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("choose_dist_data result must contain a 2D matrix")

    if isinstance(column, str):
        labels = [str(value) for value in obj.penalties]
        if column not in labels:
            raise ValueError("column label not found in penalties")
        column_index = labels.index(column)
    elif isinstance(column, (float, np.floating)):
        penalties = np.asarray(obj.penalties, dtype=np.float64)
        matches = np.where(np.isclose(penalties, float(column)))[0]
        if matches.size == 0:
            raise ValueError("penalty value not found in penalties")
        column_index = int(matches[0])
    else:
        column_index = int(column) - 1

    if column_index < 0 or column_index >= matrix.shape[1]:
        raise ValueError("column index out of range")

    column_values = np.asarray(matrix[:, column_index], dtype=np.float64)
    finite_mask = np.isfinite(column_values)
    finite_order = np.argsort(column_values[finite_mask], kind="mergesort")
    finite_indices = np.nonzero(finite_mask)[0][finite_order]
    non_finite_indices = np.nonzero(~finite_mask)[0]
    order = np.concatenate([finite_indices, non_finite_indices])
    rows = tuple(
        ChooseDistOrderRow(
            family=str(obj.families[index]),
            value=float(column_values[index]),
        )
        for index in order
    )
    return ChooseDistOrderResult(
        column_index=column_index + 1,
        penalty=float(obj.penalties[column_index]),
        rows=rows,
    )


def choose_dist_pred_data(
    object: GAMLSSModel,
    type: str = "realAll",
    extra: Sequence[str] | None = None,
    newdata: Mapping[str, Any] | None = None,
    rand: Sequence[int] | np.ndarray | None = None,
) -> ChooseDistPredResult:
    """R reference: `gamlss/R/chooseDistParallel.R::chooseDistPred`."""

    from .fitting import gamlss

    _require_gamlss_method(object)
    if (rand is None) == (newdata is None):
        raise ValueError("exactly one of rand or newdata must be provided")

    available = _fit_dist_available_families()
    selected_type = str(type).strip()
    key = selected_type.lower()
    if key == "extra" and not extra:
        raise ValueError("extra families must be provided when type='extra'")
    if key not in available and key != "extra":
        raise ValueError("type must be one of 'realAll', 'realline', 'realplus', 'real0to1', 'counts', 'binom', or 'extra'")

    family_names = list(extra if key == "extra" else available[key])
    if extra is not None and key != "extra":
        family_names.extend(str(name).strip().upper() for name in extra if str(name).strip())
    family_names = list(dict.fromkeys(str(name).strip().upper() for name in family_names if str(name).strip()))

    call = object.call or {}
    data = call.get("data")
    if data is None:
        raise ValueError("choose_dist_pred_data requires call['data']")
    response = object.terms.get("mu", {}).get("response")
    if response is None:
        raise ValueError("choose_dist_pred_data requires a stored response name")

    formula_text = str(call.get("formula", formula(object, what="mu")))
    sigma_formula = str(call.get("sigma_formula", formula(object, what="sigma"))) if "sigma" in object.par else "~1"
    parameter_formulas = dict(call.get("parameter_formulas", {}))
    method = str(call.get("method", object.additional_slots.get("method", "RS")))
    weights = call.get("weights", object.weights)

    if rand is not None:
        split = np.asarray(rand)
        y_full = np.asarray(data[response])
        if split.shape[0] != y_full.shape[0]:
            raise ValueError("rand must have the same length as the stored response")
        if not np.all(np.isin(split, [1, 2])):
            raise ValueError("rand values must be 1 or 2")
        train_mask = split == 1
        valid_mask = split == 2
        if not np.any(train_mask) or not np.any(valid_mask):
            raise ValueError("rand must contain both training (1) and validation (2) rows")
        train_data = _subset_mapping_rows(data, train_mask)
        validation_data = _subset_mapping_rows(data, valid_mask)
        validation_y = np.asarray(validation_data[response], dtype=np.float64).ravel()
        train_weights = None if weights is None else np.asarray(weights)[train_mask]
    else:
        train_data = data
        validation_data = dict(newdata)
        if response not in validation_data:
            raise ValueError(f"newdata must contain response column {response!r}")
        validation_y = np.asarray(validation_data[response], dtype=np.float64).ravel()
        train_weights = weights

    scores = np.full(len(family_names), np.nan, dtype=np.float64)
    failed: list[str] = []
    for index, family_name in enumerate(family_names):
        try:
            fitted_model = gamlss(
                formula=formula_text,
                family=family_name,
                data=train_data,
                sigma_formula=sigma_formula,
                parameter_formulas=parameter_formulas if parameter_formulas else None,
                method=method,
                weights=train_weights,
            )
            predicted = predict_all(
                fitted_model,
                newdata=validation_data,
                type="response",
                output="list",
                se_fit=False,
            )
            values = predicted.values if isinstance(predicted, PredictAllResult) else dict(predicted)
            deviance_kwargs: dict[str, Any] = {"y": validation_y}
            for parameter in fitted_model.par:
                deviance_kwargs[parameter] = np.asarray(values[parameter], dtype=np.float64)
            scores[index] = float(np.sum(np.asarray(fitted_model.family.g_dev_inc(**deviance_kwargs), dtype=np.float64)))
        except Exception:
            failed.append(family_name)

    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        raise ValueError("no candidate family produced a validation score")
    best_index = int(np.nanargmin(scores))
    return ChooseDistPredResult(
        type=selected_type,
        families=tuple(family_names),
        scores=scores,
        best_family=family_names[best_index],
        failed=tuple(failed),
        validation_size=int(validation_y.size),
    )

choose_dist_pred = choose_dist_pred_data
chooseDist = choose_dist_data
chooseDistPred = choose_dist_pred_data
getOrder = get_order

__all__ = [
    "ChooseDistOrderResult",
    "ChooseDistOrderRow",
    "ChooseDistPredResult",
    "ChooseDistResult",
    "chooseDist",
    "chooseDistPred",
    "choose_dist_data",
    "choose_dist_pred",
    "choose_dist_pred_data",
    "getOrder",
    "get_order",
]
