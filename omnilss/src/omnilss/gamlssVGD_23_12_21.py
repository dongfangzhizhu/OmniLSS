"""R-aligned validation global deviance interfaces.

R source reference:
- file: `gamlss/R/gamlssVGD_23_12_21.R`
- functions: `gamlssVGD`, `VGD`, `is.gamlssVGD`, `getTGD`, `gamlssCV`, `CV`, `is.CV`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .families import FamilyDefinition
from .fitDistPred import GAMLSSMLPredResult, gamlss_ml_pred, gamlss_ml_pred_data
from .model import GAMLSSModel
from .operations import is_gamlss
from .predictAll_22_08_22 import PredictAllResult, predict_all


@dataclass(frozen=True)
class TGDResult:
    """Structured staged `getTGD` payload."""

    family: str
    test_global_deviance: float
    prediction_error: float
    deviance_increment: np.ndarray
    residuals: np.ndarray | None
    validation_size: int


@dataclass(frozen=True)
class GAMLSSVGDResult:
    """Structured staged `gamlssVGD` payload."""

    family: str
    model: GAMLSSModel
    validation_global_deviance: float
    validation_increment: np.ndarray
    prediction_error: float
    validation_residuals: np.ndarray | None
    validation_size: int


@dataclass(frozen=True)
class VGDComparisonRow:
    """One row in a staged `VGD` comparison result."""

    family: str
    pred_gd: float


@dataclass(frozen=True)
class VGDComparisonResult:
    """Structured staged `VGD` comparison payload."""

    rows: tuple[VGDComparisonRow, ...]


@dataclass(frozen=True)
class GAMLSSCVResult:
    """Structured staged `gamlssCV` payload."""

    family: str
    model: GAMLSSModel
    cv: float
    all_cv: np.ndarray
    resid_cv: np.ndarray
    folds: np.ndarray
    k_fold: int


@dataclass(frozen=True)
class CVComparisonRow:
    """One row in a staged `CV` comparison result."""

    family: str
    cv: float


@dataclass(frozen=True)
class CVComparisonResult:
    """Structured staged `CV` comparison payload."""

    rows: tuple[CVComparisonRow, ...]


def _require_gamlss_method(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _unwrap_validation_model(object: Any) -> GAMLSSModel:
    if isinstance(object, GAMLSSModel):
        return object
    model = getattr(object, "model", None)
    if isinstance(model, GAMLSSModel):
        return model
    raise TypeError("This is not an gamlss object")


def _normalize_formula_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _response_name_from_formula(formula_text: str) -> str:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        raise ValueError("formula must contain '~'")
    return normalized.split("~", 1)[0].strip()


def _subset_mapping_rows(data: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != mask.shape[0]:
            subset[key] = value
            continue
        subset[key] = array[mask]
    return subset


def is_gamlss_vgd(x: Any) -> bool:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::is.gamlssVGD`."""

    try:
        model = _unwrap_validation_model(x)
    except TypeError:
        return False
    return "VGD" in model.additional_slots


def is_gamlss_cv(x: Any) -> bool:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::is.CV`."""

    try:
        model = _unwrap_validation_model(x)
    except TypeError:
        return False
    return "CV" in model.additional_slots


def get_tgd_data(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
) -> TGDResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::getTGD`.

    Staged behavior:
    - Requires a fitted `gamlss` model plus `newdata` containing the response.
    - Computes test global deviance and prediction error using staged `predict_all()`.
    - Returns validation residuals via stored `rqres` when available, otherwise
      falls back to signed square-root deviance residuals.
    """

    _require_gamlss_method(object)
    if newdata is None:
        raise ValueError("newdata is required")

    response = object.terms.get("mu", {}).get("response")
    if response is None:
        raise ValueError("get_tgd_data requires a stored response name")
    if response not in newdata:
        raise ValueError(f"newdata must contain response column {response!r}")

    validation_y = np.asarray(newdata[response], dtype=np.float64).ravel()
    if validation_y.size == 0:
        raise ValueError("newdata response must contain at least one observation")

    predicted = predict_all(
        object,
        newdata=dict(newdata),
        type="response",
        output="list",
        se_fit=False,
    )
    values = predicted.values if isinstance(predicted, PredictAllResult) else dict(predicted)
    deviance_kwargs: dict[str, Any] = {"y": validation_y}
    for parameter in object.par:
        deviance_kwargs[parameter] = np.asarray(values[parameter], dtype=np.float64)
    deviance_increment = np.asarray(object.family.g_dev_inc(**deviance_kwargs), dtype=np.float64)
    test_global_deviance = float(np.sum(deviance_increment))
    prediction_error = float(test_global_deviance / validation_y.size)

    residuals_value: np.ndarray | None
    if object.rqres is not None:
        rqres_kwargs = {"y": validation_y}
        for parameter in object.par:
            rqres_kwargs[parameter] = np.asarray(values[parameter], dtype=np.float64)
        try:
            residuals_value = np.asarray(object.rqres(**rqres_kwargs), dtype=np.float64)
        except Exception:
            residuals_value = None
    else:
        mu = np.asarray(values.get("mu", validation_y), dtype=np.float64).ravel()
        signed = np.sign(validation_y - mu)
        residuals_value = signed * np.sqrt(np.maximum(deviance_increment, 0.0))

    return TGDResult(
        family=str(getattr(object.family, "name", object.family)),
        test_global_deviance=test_global_deviance,
        prediction_error=prediction_error,
        deviance_increment=deviance_increment,
        residuals=residuals_value,
        validation_size=int(validation_y.size),
    )


def get_tgd(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
) -> TGDResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::getTGD`."""

    return get_tgd_data(object=object, newdata=newdata)


def gamlss_vgd_data(
    formula: str,
    data: Mapping[str, Any] | None = None,
    family: FamilyDefinition | str | None = None,
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    rand: Sequence[int] | np.ndarray | None = None,
    newdata: Mapping[str, Any] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    method: str = "RS",
) -> GAMLSSVGDResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::gamlssVGD`."""

    from .fitting import gamlss

    if data is None:
        raise ValueError("data is required")
    if (rand is None) == (newdata is None):
        raise ValueError("exactly one of rand or newdata must be provided")

    response_name = _response_name_from_formula(_normalize_formula_text(formula))
    if rand is not None:
        response = np.asarray(data[response_name], dtype=np.float64).ravel()
        split = np.asarray(rand)
        if split.shape[0] != response.shape[0]:
            raise ValueError("rand must have the same length as the stored response")
        if not np.all(np.isin(split, [1, 2])):
            raise ValueError("rand values must be 1 or 2")
        train_mask = split == 1
        valid_mask = split == 2
        if not np.any(train_mask) or not np.any(valid_mask):
            raise ValueError("rand must contain both training (1) and validation (2) rows")
        train_data = _subset_mapping_rows(data, train_mask)
        validation_data = _subset_mapping_rows(data, valid_mask)
        train_weights = None if weights is None else np.asarray(weights, dtype=np.float64)[train_mask]
    else:
        train_data = dict(data)
        validation_data = dict(newdata)
        train_weights = weights

    fit = gamlss(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        family=family,
        data=train_data,
        weights=train_weights,
        method=method,
    )
    tgd_result = get_tgd_data(fit, newdata=validation_data)
    fit.additional_slots["VGD"] = float(tgd_result.test_global_deviance)
    fit.additional_slots["IncrVGD"] = np.asarray(tgd_result.deviance_increment, dtype=np.float64)
    fit.additional_slots["predictError"] = float(tgd_result.prediction_error)
    fit.additional_slots["residVal"] = None if tgd_result.residuals is None else np.asarray(tgd_result.residuals, dtype=np.float64)
    fit.additional_slots["class_name"] = "gamlssVGD"

    return GAMLSSVGDResult(
        family=str(getattr(fit.family, "name", fit.family)),
        model=fit,
        validation_global_deviance=float(tgd_result.test_global_deviance),
        validation_increment=np.asarray(tgd_result.deviance_increment, dtype=np.float64),
        prediction_error=float(tgd_result.prediction_error),
        validation_residuals=None if tgd_result.residuals is None else np.asarray(tgd_result.residuals, dtype=np.float64),
        validation_size=int(tgd_result.validation_size),
    )


def gamlss_vgd(
    formula: str,
    data: Mapping[str, Any] | None = None,
    family: FamilyDefinition | str | None = None,
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    rand: Sequence[int] | np.ndarray | None = None,
    newdata: Mapping[str, Any] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    method: str = "RS",
) -> GAMLSSVGDResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::gamlssVGD`."""

    return gamlss_vgd_data(
        formula=formula,
        data=data,
        family=family,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        rand=rand,
        newdata=newdata,
        weights=weights,
        method=method,
    )


def vgd(*objects: Any) -> float | VGDComparisonResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::VGD`."""

    if not objects:
        raise ValueError("at least one object is required")

    def _value(object: Any) -> float:
        model = _unwrap_validation_model(object)
        if "VGD" not in model.additional_slots:
            raise ValueError("this is not a gamlssVGD-style object")
        return float(model.additional_slots["VGD"])

    if len(objects) == 1:
        return _value(objects[0])

    rows = tuple(
        sorted(
            (
                VGDComparisonRow(
                    family=str(getattr(_unwrap_validation_model(model).family, "name", _unwrap_validation_model(model).family)),
                    pred_gd=_value(model),
                )
                for model in objects
            ),
            key=lambda row: (row.pred_gd, row.family),
        )
    )
    return VGDComparisonResult(rows=rows)


def gamlss_cv_data(
    formula: str,
    data: Mapping[str, Any] | None = None,
    family: FamilyDefinition | str | None = None,
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    k_fold: int = 10,
    set_seed: int = 123,
    rand: Sequence[int] | np.ndarray | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    method: str = "RS",
) -> GAMLSSCVResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::gamlssCV`."""

    from .fitting import gamlss

    if data is None:
        raise ValueError("data is required")
    if int(k_fold) < 2:
        raise ValueError("k_fold must be at least 2")

    normalized_formula = _normalize_formula_text(formula)
    response_name = _response_name_from_formula(normalized_formula)
    response = np.asarray(data[response_name], dtype=np.float64).ravel()
    n_obs = int(response.size)
    if n_obs == 0:
        raise ValueError("data must contain at least one row")

    if rand is None:
        rng = np.random.default_rng(int(set_seed))
        fold_ids = np.tile(np.arange(1, int(k_fold) + 1, dtype=np.int64), int(np.ceil(n_obs / int(k_fold))))[:n_obs]
        rng.shuffle(fold_ids)
    else:
        fold_ids = np.asarray(rand)
    if fold_ids.shape[0] != n_obs:
        raise ValueError("rand must have the same length as the stored response")

    unique_folds = np.asarray(sorted(np.unique(fold_ids)))
    if unique_folds.size < 2:
        raise ValueError("rand must contain at least two unique folds")

    all_cv = np.zeros(unique_folds.size, dtype=np.float64)
    resid_cv = np.zeros(n_obs, dtype=np.float64)
    weights_array = None if weights is None else np.asarray(weights, dtype=np.float64)

    for index, fold_value in enumerate(unique_folds):
        split = np.where(fold_ids == fold_value, 2, 1)
        fold_result = gamlss_vgd_data(
            formula=normalized_formula,
            data=data,
            family=family,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            rand=split,
            weights=weights_array,
            method=method,
        )
        validation_mask = split == 2
        all_cv[index] = float(fold_result.validation_global_deviance)
        if fold_result.validation_residuals is not None:
            resid_cv[validation_mask] = np.asarray(fold_result.validation_residuals, dtype=np.float64)

    fit = gamlss(
        formula=normalized_formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        family=family,
        data=data,
        weights=weights_array,
        method=method,
    )
    fit.additional_slots["CV"] = float(np.sum(all_cv))
    fit.additional_slots["allCV"] = np.asarray(all_cv, dtype=np.float64)
    fit.additional_slots["residCV"] = np.asarray(resid_cv, dtype=np.float64)
    fit.additional_slots["folds"] = np.asarray(fold_ids)
    fit.additional_slots["K.fold"] = int(unique_folds.size)
    fit.additional_slots["class_name"] = "gamlssCV"

    return GAMLSSCVResult(
        family=str(getattr(fit.family, "name", fit.family)),
        model=fit,
        cv=float(np.sum(all_cv)),
        all_cv=np.asarray(all_cv, dtype=np.float64),
        resid_cv=np.asarray(resid_cv, dtype=np.float64),
        folds=np.asarray(fold_ids),
        k_fold=int(unique_folds.size),
    )


def gamlss_cv(
    formula: str,
    data: Mapping[str, Any] | None = None,
    family: FamilyDefinition | str | None = None,
    sigma_formula: str = "~1",
    parameter_formulas: Mapping[str, str] | None = None,
    k_fold: int = 10,
    set_seed: int = 123,
    rand: Sequence[int] | np.ndarray | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    method: str = "RS",
) -> GAMLSSCVResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::gamlssCV`."""

    return gamlss_cv_data(
        formula=formula,
        data=data,
        family=family,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        k_fold=k_fold,
        set_seed=set_seed,
        rand=rand,
        weights=weights,
        method=method,
    )


def cv(*objects: Any) -> float | CVComparisonResult:
    """R reference: `gamlss/R/gamlssVGD_23_12_21.R::CV`."""

    if not objects:
        raise ValueError("at least one object is required")

    def _value(object: Any) -> float:
        model = _unwrap_validation_model(object)
        if "CV" not in model.additional_slots:
            raise ValueError("this is not a gamlssCV-style object")
        return float(model.additional_slots["CV"])

    if len(objects) == 1:
        return _value(objects[0])

    rows = tuple(
        sorted(
            (
                CVComparisonRow(
                    family=str(getattr(_unwrap_validation_model(model).family, "name", _unwrap_validation_model(model).family)),
                    cv=_value(model),
                )
                for model in objects
            ),
            key=lambda row: (row.cv, row.family),
        )
    )
    return CVComparisonResult(rows=rows)

gamlssVGD = gamlss_vgd
VGD = vgd
getTGD = get_tgd
gamlssCV = gamlss_cv
CV = cv

__all__ = [
    "CVComparisonResult",
    "CVComparisonRow",
    "GAMLSSCVResult",
    "GAMLSSMLPredResult",
    "GAMLSSVGDResult",
    "TGDResult",
    "VGDComparisonResult",
    "VGDComparisonRow",
    "CV",
    "VGD",
    "cv",
    "gamlssCV",
    "gamlss_cv",
    "gamlss_cv_data",
    "gamlssVGD",
    "gamlss_ml_pred",
    "gamlss_ml_pred_data",
    "getTGD",
    "gamlss_vgd",
    "gamlss_vgd_data",
    "get_tgd",
    "get_tgd_data",
    "is_gamlss_cv",
    "is_gamlss_vgd",
    "vgd",
]
