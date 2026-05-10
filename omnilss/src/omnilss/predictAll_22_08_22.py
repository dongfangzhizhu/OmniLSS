"""R-aligned multi-parameter prediction interfaces.

R source reference:
- file: `gamlss/R/predictAll_22_08_22.R`
- functions: `predictAll`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class PredictAllResult:
    """Structured staged `predictAll` payload."""

    family: str
    output: str
    values: dict[str, Any]


def _require_gamlss_method(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _slice_predict_all_values(
    values: dict[str, Any],
    start: int,
    stop: int,
) -> dict[str, Any]:
    """Slice staged `predictAll` values to a target row interval."""

    sliced: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict) and "fit" in value:
            fit_array = np.asarray(value["fit"], dtype=np.float64)
            entry: dict[str, Any] = {"fit": fit_array[start:stop]}
            if value.get("se.fit") is not None:
                se_array = np.asarray(value["se.fit"], dtype=np.float64)
                entry["se.fit"] = se_array[start:stop]
            else:
                entry["se.fit"] = None
            sliced[key] = entry
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.ndim >= 1 and array.shape[0] >= stop:
            sliced[key] = array[start:stop]
        else:
            sliced[key] = array
    return sliced


def _predict_all_with_weights_refit(
    object: GAMLSSModel,
    newdata: dict[str, Any],
    type: str,
    terms: list[str] | tuple[str, ...] | None,
    se_fit: bool,
) -> dict[str, Any]:
    """Staged `predictAll(..., use.weights=TRUE)` refit path.

    R reference: `gamlss/R/predictAll_22_08_22.R::predictAll`.

    Current staged behavior:
    - Rebuilds a combined old/new dataset.
    - Assigns tiny positive weights to the new rows.
    - Refits with the stored family, formulas, and method.
    - Returns predictions restricted to the appended new-data rows.
    """

    from .fitting import gamlss, gamlss_ml
    from .methods import predict

    call = object.call or {}
    original_data = call.get("data")
    if original_data is None:
        raise ValueError("predict_all(use_weights=True) requires call['data']")

    response_name = object.terms.get("mu", {}).get("response")
    if response_name is None:
        raise ValueError("predict_all(use_weights=True) requires a stored response name")

    original_y = np.asarray(original_data[response_name], dtype=np.float64).ravel()
    old_n = original_y.shape[0]

    new_lengths = [
        np.asarray(value).shape[0]
        for key, value in newdata.items()
        if key != response_name and np.asarray(value).ndim > 0
    ]
    if response_name in newdata:
        new_lengths.append(np.asarray(newdata[response_name]).shape[0])
    if not new_lengths:
        raise ValueError("predict_all(use_weights=True) requires at least one column in newdata")
    new_n = int(new_lengths[0])
    if any(length != new_n for length in new_lengths):
        raise ValueError("all newdata columns must share the same length")

    combined: dict[str, np.ndarray] = {}
    for key, original_value in original_data.items():
        original_array = np.asarray(original_value)
        if original_array.ndim == 0:
            original_array = np.repeat(original_array.reshape(1), old_n)
        if key == response_name:
            if key in newdata:
                new_array = np.asarray(newdata[key], dtype=np.float64).ravel()
            else:
                filler = float(np.mean(original_y)) if original_y.size else 0.0
                new_array = np.full(new_n, filler, dtype=np.float64)
            combined[key] = np.concatenate(
                [np.asarray(original_array, dtype=np.float64).ravel(), new_array]
            ).astype(np.float64, copy=False)
            continue

        if key in newdata:
            new_array = np.asarray(newdata[key], dtype=np.float64).ravel()
        else:
            filler = np.asarray(original_array).ravel()
            if filler.size == 0:
                new_array = np.zeros(new_n, dtype=np.float64)
            else:
                new_array = np.repeat(filler[-1], new_n).astype(np.float64, copy=False)
        combined[key] = np.concatenate(
            [np.asarray(original_array, dtype=np.float64).ravel(), new_array]
        ).astype(np.float64, copy=False)

    for key, value in newdata.items():
        if key in combined:
            continue
        new_array = np.asarray(value, dtype=np.float64).ravel()
        if new_array.shape[0] != new_n:
            raise ValueError("all newdata columns must share the same length")
        combined[key] = np.concatenate(
            [np.repeat(new_array[0] if new_array.size else 0.0, old_n), new_array]
        ).astype(np.float64, copy=False)

    old_weights = np.asarray(
        object.call.get("weights", np.ones(old_n, dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    if old_weights.shape[0] != old_n:
        old_weights = np.ones(old_n, dtype=np.float64)
    tiny_weight = np.finfo(np.float64).tiny
    combined_weights = np.concatenate(
        [old_weights, np.full(new_n, tiny_weight, dtype=np.float64)]
    ).astype(np.float64, copy=False)

    mu_formula = object.formulas.get("mu")
    if mu_formula is None:
        raise ValueError("predict_all(use_weights=True) requires a stored mu formula")
    sigma_formula = object.formulas.get("sigma", "~1")
    method_name = str(object.additional_slots.get("method", call.get("method", "RS"))).upper()

    if method_name == "ML":
        refit_model = gamlss_ml(
            formula=mu_formula,
            sigma_formula=sigma_formula,
            family=object.family,
            data=combined,
            weights=combined_weights,
        )
    else:
        refit_model = gamlss(
            formula=mu_formula,
            sigma_formula=sigma_formula,
            family=object.family,
            data=combined,
            weights=combined_weights,
            method=method_name,
        )

    refit_result = predict_all(
        refit_model,
        newdata=None,
        type=type,
        terms=terms,
        se_fit=se_fit,
        output="list",
        use_weights=False,
    )
    sliced = _slice_predict_all_values(refit_result.values, start=old_n, stop=old_n + new_n)
    if se_fit:
        for parameter in object.par:
            value = sliced.get(parameter)
            if not isinstance(value, dict) or "fit" not in value:
                continue
            se_value = value.get("se.fit")
            se_array = None if se_value is None else np.asarray(se_value, dtype=np.float64)
            needs_fallback = se_array is None or not np.isfinite(se_array).all()
            if not needs_fallback:
                continue
            fallback = predict(
                object,
                what=parameter,
                type=type,
                terms=terms,
                se_fit=True,
                newdata=newdata,
            )
            if isinstance(fallback, dict) and fallback.get("se.fit") is not None:
                value["se.fit"] = np.asarray(fallback["se.fit"], dtype=np.float64)
    return sliced


def _predict_all_term_names(
    object: GAMLSSModel,
    parameter: str,
    count: int,
    requested_terms: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Resolve staged column names for `predict_all(type="terms")`."""

    if requested_terms is not None:
        base_names = [str(term).strip() for term in requested_terms]
    else:
        info = object.terms.get(parameter, {})
        labels = info.get("column_names") or info.get("term_labels") or []
        base_names = [str(label).strip() for label in labels]
    if not base_names:
        base_names = [f"term{index + 1}" for index in range(count)]
    if len(base_names) < count:
        base_names = base_names + [f"term{index + 1}" for index in range(len(base_names), count)]
    return [f"{parameter}_{name}" for name in base_names[:count]]


def predict_all(
    object: GAMLSSModel,
    newdata: dict[str, Any] | None = None,
    type: str = "response",
    terms: list[str] | tuple[str, ...] | None = None,
    se_fit: bool = False,
    output: str = "list",
    use_weights: bool = False,
) -> PredictAllResult | dict[str, Any] | np.ndarray:
    """R reference: `gamlss/R/predictAll_22_08_22.R::predictAll`.

    Staged behavior:
    - Aggregates `predict()` across fitted parameters.
    - Supports `output` as `list`, `data.frame`, or `matrix`.
    - Supports a staged `use_weights=True` refit path for `newdata`.
    """

    from .methods import predict

    _require_gamlss_method(object)
    requested_type = type.strip().lower()
    if requested_type not in {"response", "link", "terms"}:
        raise ValueError("type must be one of 'response', 'link', or 'terms'")
    output_name = output.strip().lower()
    if output_name not in {"list", "data.frame", "matrix"}:
        raise ValueError("output must be one of 'list', 'data.frame', or 'matrix'")

    values: dict[str, Any] = {}
    response_name = object.terms.get("mu", {}).get("response")
    if use_weights and newdata is None:
        raise ValueError("predict_all(use_weights=True) requires newdata")

    if use_weights and newdata is not None:
        if requested_type == "terms":
            raise ValueError("predict_all(use_weights=True) does not support type='terms' yet")
        values = _predict_all_with_weights_refit(
            object,
            newdata=newdata,
            type=requested_type,
            terms=terms,
            se_fit=se_fit,
        )
        if response_name is not None and response_name in newdata:
            values["y"] = np.asarray(newdata[response_name], dtype=np.float64)
    elif newdata is None:
        if response_name is not None and object.call is not None and "data" in object.call and response_name in object.call["data"]:
            values["y"] = np.asarray(object.call["data"][response_name], dtype=np.float64)
        for parameter in object.par:
            values[parameter] = predict(object, what=parameter, type=requested_type, terms=terms, se_fit=se_fit)
    else:
        if response_name is not None and response_name in newdata:
            values["y"] = np.asarray(newdata[response_name], dtype=np.float64)
        for parameter in object.par:
            values[parameter] = predict(
                object,
                newdata=newdata,
                what=parameter,
                type=requested_type,
                terms=terms,
                se_fit=se_fit,
            )

    family_name = getattr(object.family, "name", str(object.family))
    result = PredictAllResult(family=family_name, output=output_name, values=values)
    if output_name == "list":
        return result

    normalized: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if isinstance(value, dict) and "fit" in value:
            fit_array = np.asarray(value["fit"], dtype=np.float64)
            if fit_array.ndim == 0:
                fit_array = fit_array.reshape(1)
            if fit_array.ndim > 1:
                if requested_type == "terms" and fit_array.ndim == 2:
                    if fit_array.shape[1] == 0:
                        continue
                    term_names = _predict_all_term_names(object, key, fit_array.shape[1], terms)
                    for index, name in enumerate(term_names):
                        normalized[name] = fit_array[:, index]
                    se_value = value.get("se.fit")
                    if se_value is not None:
                        se_array = np.asarray(se_value, dtype=np.float64)
                        if se_array.ndim == 1:
                            se_array = se_array[:, None]
                        if se_array.ndim != 2 or se_array.shape[1] != fit_array.shape[1]:
                            raise ValueError("terms standard errors must match the term prediction matrix")
                        for index, name in enumerate(term_names):
                            normalized[f"{name}_se"] = se_array[:, index]
                    elif se_fit:
                        for name in term_names:
                            normalized[f"{name}_se"] = np.full(fit_array.shape[0], np.nan, dtype=np.float64)
                    continue
                if fit_array.ndim == 2 and fit_array.shape[1] == 1:
                    fit_array = fit_array[:, 0]
                else:
                    fit_array = fit_array.reshape(fit_array.shape[0], -1)
                    if fit_array.shape[1] != 1:
                        raise ValueError("matrix/data.frame output currently requires one-dimensional parameter predictions")
                    fit_array = fit_array[:, 0]
            normalized[key] = fit_array
            se_value = value.get("se.fit")
            if se_value is not None:
                se_array = np.asarray(se_value, dtype=np.float64)
                if se_array.ndim == 0:
                    se_array = se_array.reshape(1)
                if se_array.ndim > 1:
                    if se_array.ndim == 2 and se_array.shape[1] == 1:
                        se_array = se_array[:, 0]
                    else:
                        raise ValueError("matrix/data.frame output currently requires one-dimensional standard errors")
                normalized[f"{key}_se"] = se_array
            elif se_fit:
                normalized[f"{key}_se"] = np.full(fit_array.shape[0], np.nan, dtype=np.float64)
            continue

        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim > 1:
            if requested_type == "terms" and array.ndim == 2:
                if array.shape[1] == 0:
                    continue
                term_names = _predict_all_term_names(object, key, array.shape[1], terms)
                for index, name in enumerate(term_names):
                    normalized[name] = array[:, index]
                continue
            if array.ndim == 2 and array.shape[1] == 1:
                array = array[:, 0]
            else:
                array = array.reshape(array.shape[0], -1)
                if array.shape[1] != 1:
                    raise ValueError("matrix/data.frame output currently requires one-dimensional parameter predictions")
                array = array[:, 0]
        normalized[key] = array

    column_names = list(normalized.keys())
    lengths = {name: value.shape[0] for name, value in normalized.items()}
    target_length = max(lengths.values())
    expanded: dict[str, np.ndarray] = {}
    for name in column_names:
        value = normalized[name]
        if value.shape[0] == target_length:
            expanded[name] = value.astype(np.float64, copy=False)
            continue
        if value.shape[0] == 1:
            expanded[name] = np.repeat(value, target_length).astype(np.float64, copy=False)
            continue
        raise ValueError("data.frame/matrix output requires predictions to share a common length or be scalar")

    if output_name == "data.frame":
        return expanded

    matrix = np.column_stack([expanded[name] for name in column_names]).astype(np.float64, copy=False)
    return matrix

predictAll = predict_all

__all__ = [
    "PredictAllResult",
    "predictAll",
    "predict_all",
]
