"""R-aligned test global deviance selection interfaces.

R source reference:
- file: `gamlss/R/stepTGD.R`
- functions: `extractTGD`, `drop1TGD`, `add1TGD`, `stepTGD`,
  `drop1TGDP`, `add1TGDP`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .gamlssVGD_23_12_21 import get_tgd_data
from .model import GAMLSSModel
from .operations import formula, is_gamlss, terms


@dataclass(frozen=True)
class ExtractTGDResult:
    """Structured staged `extractTGD` payload."""

    edf: float
    tgd: float


@dataclass(frozen=True)
class TGDScopeRow:
    """One staged `drop1TGD`/`add1TGD` comparison row."""

    term: str
    df_fit: float
    tgd: float
    delta_df: float | None


@dataclass(frozen=True)
class TGDScopeResult:
    """Structured staged `drop1TGD`/`add1TGD` payload."""

    what: str
    direction: str
    rows: tuple[TGDScopeRow, ...]


@dataclass(frozen=True)
class StepTGDStep:
    """One staged `stepTGD` search step."""

    step: int
    formula: str
    tgd: float
    change: str
    df_fit: float
    deviance: float


@dataclass(frozen=True)
class StepTGDResult:
    """Structured staged `stepTGD` payload."""

    model: GAMLSSModel
    what: str
    direction: str
    steps: tuple[StepTGDStep, ...]


@dataclass(frozen=True)
class MultiParameterTGDScopeRow:
    """One staged multi-parameter `drop1TGD`/`add1TGD` row."""

    parameter: str
    term: str
    df_fit: float
    tgd: float
    delta_df: float | None


@dataclass(frozen=True)
class MultiParameterTGDScopeResult:
    """Structured staged multi-parameter `drop1TGD`/`add1TGD` payload."""

    direction: str
    rows: tuple[MultiParameterTGDScopeRow, ...]


@dataclass(frozen=True)
class StepTGDAllStep:
    """One staged multi-parameter `stepTGD` search step."""

    step: int
    parameter: str
    formula: str
    tgd: float
    change: str
    df_fit: float
    deviance: float


@dataclass(frozen=True)
class StepTGDAllResult:
    """Structured staged multi-parameter `stepTGD` payload."""

    model: GAMLSSModel
    parameters: tuple[str, ...]
    direction: str
    steps: tuple[StepTGDAllStep, ...]


def _require_gamlss_method(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _normalize_formula_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _rhs_terms(formula_text: str) -> list[str]:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        raise ValueError("formula must contain '~'")
    rhs = normalized.split("~", 1)[1].strip()
    if rhs in {"", "1"}:
        return []
    return [term.strip() for term in rhs.split("+") if term.strip() and term.strip() != "1"]


def _response_name_from_formula(formula_text: str) -> str:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        raise ValueError("formula must contain '~'")
    return normalized.split("~", 1)[0].strip()


def _compose_formula(response: str, terms_list: list[str]) -> str:
    rhs = " + ".join(terms_list) if terms_list else "1"
    return f"{response} ~ {rhs}"


def _formula_rhs_spec(formula_text: str) -> str:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        return normalized
    return f"~ {normalized.split('~', 1)[1].strip()}"


def _fit_with_updated_formula(
    object: GAMLSSModel,
    formula_updates: dict[str, str],
) -> GAMLSSModel:
    from .fitting import gamlss, gamlss_ml

    if object.call is None or "data" not in object.call:
        raise ValueError("object must contain stored call data")

    mu_formula = formula_updates.get("mu", _normalize_formula_text(object.formulas.get("mu", "")))
    sigma_formula = formula_updates.get("sigma", _normalize_formula_text(object.formulas.get("sigma", "~1")))
    family = object.family
    data = object.call["data"]
    weights = object.weights
    control = object.control
    method_name = str(object.additional_slots.get("method", "RS")).upper()
    parameter_formulas = {parameter: _normalize_formula_text(value) for parameter, value in object.formulas.items()}
    parameter_formulas.update(formula_updates)

    if method_name == "ML":
        return gamlss_ml(
            formula=mu_formula,
            sigma_formula=_formula_rhs_spec(sigma_formula),
            parameter_formulas={parameter: _formula_rhs_spec(value) for parameter, value in parameter_formulas.items()},
            family=family,
            data=data,
            weights=weights,
        )
    return gamlss(
        formula=mu_formula,
        sigma_formula=_formula_rhs_spec(sigma_formula),
        parameter_formulas={parameter: _formula_rhs_spec(value) for parameter, value in parameter_formulas.items()},
        family=family,
        data=data,
        weights=weights,
        method=method_name,
        control=None if not isinstance(control, dict) else None,
    )


def _scope_terms(object: GAMLSSModel, what: str, scope: Any | None) -> list[str]:
    available_terms = list(terms(object, what=what).get("term_labels", []))
    if scope is None:
        return available_terms
    if isinstance(scope, str):
        normalized = scope.strip()
        if normalized.startswith("~"):
            candidate_terms = _rhs_terms(f"response {normalized}")
        else:
            candidate_terms = [term.strip() for term in normalized.split("+") if term.strip()]
    else:
        candidate_terms = [str(term).strip() for term in scope if str(term).strip()]
    return [term for term in candidate_terms if term in available_terms or scope is not None]


def _updated_formulas_for_parameter(
    object: GAMLSSModel,
    what: str,
    updated_formula: str,
) -> dict[str, str]:
    formulas = {parameter: _normalize_formula_text(value) for parameter, value in object.formulas.items()}
    formulas[what] = updated_formula
    return formulas


def _parse_scope_spec(scope: Any | None) -> tuple[list[str], list[str]]:
    if scope is None:
        return [], []
    if isinstance(scope, dict):
        lower_raw = scope.get("lower")
        upper_raw = scope.get("upper")
        lower = _rhs_terms(f"response {lower_raw}") if isinstance(lower_raw, str) and str(lower_raw).strip().startswith("~") else (
            [str(term).strip() for term in lower_raw if str(term).strip()] if lower_raw is not None else []
        )
        upper = _rhs_terms(f"response {upper_raw}") if isinstance(upper_raw, str) and str(upper_raw).strip().startswith("~") else (
            [str(term).strip() for term in upper_raw if str(term).strip()] if upper_raw is not None else []
        )
        return lower, upper
    if isinstance(scope, str) and scope.strip().startswith("~"):
        parsed = _rhs_terms(f"response {scope}")
        return [], parsed
    parsed = [str(term).strip() for term in scope if str(term).strip()]
    return [], parsed


def extract_tgd_data(
    fit: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
) -> ExtractTGDResult:
    """R reference: `gamlss/R/stepTGD.R::extractTGD`."""

    _require_gamlss_method(fit)
    tgd_result = get_tgd_data(fit, newdata=newdata)
    return ExtractTGDResult(
        edf=float(fit.df_fit),
        tgd=float(tgd_result.test_global_deviance),
    )


def extract_tgd(
    fit: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
) -> ExtractTGDResult:
    """R reference: `gamlss/R/stepTGD.R::extractTGD`."""

    return extract_tgd_data(fit=fit, newdata=newdata)


def drop1_tgd(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: Any | None = None,
    what: str = "mu",
    sorted: bool = False,
) -> TGDScopeResult:
    """R reference: `gamlss/R/stepTGD.R::drop1TGD`."""

    _require_gamlss_method(object)
    selected = what.strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")

    base_formula = _normalize_formula_text(formula(object, what=selected))
    response = _response_name_from_formula(base_formula)
    base_terms = _rhs_terms(base_formula)
    candidate_terms = _scope_terms(object, selected, scope)
    if not candidate_terms:
        raise ValueError("no terms in scope")

    baseline = extract_tgd_data(object, newdata=newdata)
    rows = [
        TGDScopeRow(term="<none>", df_fit=float(object.df_fit), tgd=float(baseline.tgd), delta_df=None)
    ]
    for term_name in candidate_terms:
        if term_name not in base_terms:
            continue
        new_terms = [term for term in base_terms if term != term_name]
        updated_formula = _compose_formula(response, new_terms)
        fitted_model = _fit_with_updated_formula(object, _updated_formulas_for_parameter(object, selected, updated_formula))
        result = extract_tgd_data(fitted_model, newdata=newdata)
        rows.append(
            TGDScopeRow(
                term=f"- {term_name}",
                df_fit=float(fitted_model.df_fit),
                tgd=float(result.tgd),
                delta_df=float(object.df_fit - fitted_model.df_fit),
            )
        )

    if sorted:
        baseline_row = rows[0]
        other_rows = list(__builtins__["sorted"](rows[1:], key=lambda row: (row.tgd, row.term)))
        rows = [baseline_row, *other_rows]

    return TGDScopeResult(what=selected, direction="drop", rows=tuple(rows))


def add1_tgd(
    object: GAMLSSModel,
    scope: Any,
    newdata: Mapping[str, Any] | None = None,
    what: str = "mu",
    sorted: bool = False,
) -> TGDScopeResult:
    """R reference: `gamlss/R/stepTGD.R::add1TGD`."""

    _require_gamlss_method(object)
    selected = what.strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")

    base_formula = _normalize_formula_text(formula(object, what=selected))
    response = _response_name_from_formula(base_formula)
    base_terms = _rhs_terms(base_formula)
    raw_scope = _scope_terms(object, selected, scope)
    candidate_terms = [term for term in raw_scope if term not in base_terms]
    if not candidate_terms:
        if isinstance(scope, str) and scope.strip().startswith("~"):
            candidate_terms = [term for term in _rhs_terms(f"response {scope}") if term not in base_terms]
        elif not raw_scope:
            candidate_terms = [str(term).strip() for term in scope if str(term).strip()]
        if not candidate_terms:
            raise ValueError("no terms in scope for adding to object")

    baseline = extract_tgd_data(object, newdata=newdata)
    rows = [
        TGDScopeRow(term="<none>", df_fit=float(object.df_fit), tgd=float(baseline.tgd), delta_df=None)
    ]
    for term_name in candidate_terms:
        if term_name in base_terms:
            continue
        updated_formula = _compose_formula(response, [*base_terms, term_name])
        fitted_model = _fit_with_updated_formula(object, _updated_formulas_for_parameter(object, selected, updated_formula))
        result = extract_tgd_data(fitted_model, newdata=newdata)
        rows.append(
            TGDScopeRow(
                term=f"+ {term_name}",
                df_fit=float(fitted_model.df_fit),
                tgd=float(result.tgd),
                delta_df=float(fitted_model.df_fit - object.df_fit),
            )
        )

    if sorted:
        baseline_row = rows[0]
        other_rows = list(__builtins__["sorted"](rows[1:], key=lambda row: (row.tgd, row.term)))
        rows = [baseline_row, *other_rows]

    return TGDScopeResult(what=selected, direction="add", rows=tuple(rows))


def step_tgd(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: Any | None = None,
    what: str = "mu",
    direction: str = "both",
    steps: int = 100,
) -> StepTGDResult:
    """R reference: `gamlss/R/stepTGD.R::stepTGD`."""

    _require_gamlss_method(object)
    selected = what.strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")

    direction_name = direction.strip().lower()
    if direction_name not in {"forward", "backward", "both"}:
        raise ValueError("direction must be one of 'forward', 'backward', or 'both'")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    current = object
    current_formula = _normalize_formula_text(formula(current, what=selected))
    current_tgd = extract_tgd_data(current, newdata=newdata).tgd
    lower_scope, upper_scope = _parse_scope_spec(scope)

    path: list[StepTGDStep] = [
        StepTGDStep(step=0, formula=current_formula, tgd=float(current_tgd), change="", df_fit=float(current.df_fit), deviance=float(current.g_dev))
    ]

    backward = direction_name in {"backward", "both"}
    forward = direction_name in {"forward", "both"}
    if scope is None and direction_name != "forward":
        forward = False

    for step_index in range(1, steps + 1):
        current_formula = _normalize_formula_text(formula(current, what=selected))
        current_terms = _rhs_terms(current_formula)
        current_tgd = extract_tgd_data(current, newdata=newdata).tgd

        candidate_drop_terms = [term for term in current_terms if term not in lower_scope] if backward else []
        candidate_add_terms = [term for term in upper_scope if term not in current_terms] if forward else []

        best_change = ""
        best_tgd = float(current_tgd)
        best_formula = current_formula
        best_model: GAMLSSModel | None = None

        for term_name in candidate_drop_terms:
            new_terms = [term for term in current_terms if term != term_name]
            updated_formula = _compose_formula(_response_name_from_formula(current_formula), new_terms)
            fitted_model = _fit_with_updated_formula(current, _updated_formulas_for_parameter(current, selected, updated_formula))
            result = extract_tgd_data(fitted_model, newdata=newdata)
            if result.tgd < best_tgd - 1e-7:
                best_tgd = float(result.tgd)
                best_change = f"- {term_name}"
                best_formula = updated_formula
                best_model = fitted_model

        for term_name in candidate_add_terms:
            updated_formula = _compose_formula(_response_name_from_formula(current_formula), [*current_terms, term_name])
            fitted_model = _fit_with_updated_formula(current, _updated_formulas_for_parameter(current, selected, updated_formula))
            result = extract_tgd_data(fitted_model, newdata=newdata)
            if result.tgd < best_tgd - 1e-7:
                best_tgd = float(result.tgd)
                best_change = f"+ {term_name}"
                best_formula = updated_formula
                best_model = fitted_model

        if best_model is None:
            break

        current = best_model
        path.append(
            StepTGDStep(
                step=step_index,
                formula=best_formula,
                tgd=float(best_tgd),
                change=best_change,
                df_fit=float(current.df_fit),
                deviance=float(current.g_dev),
            )
        )

    return StepTGDResult(model=current, what=selected, direction=direction_name, steps=tuple(path))


def drop1_tgd_all(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
) -> MultiParameterTGDScopeResult:
    """Staged multi-parameter aggregation of `drop1_tgd()`."""

    _require_gamlss_method(object)
    requested_parameters = [parameter for parameter in object.par] if parameters is None else [str(parameter).strip().lower() for parameter in parameters]
    if not requested_parameters:
        raise ValueError("at least one parameter is required")

    baseline = extract_tgd_data(object, newdata=newdata)
    rows: list[MultiParameterTGDScopeRow] = [
        MultiParameterTGDScopeRow(parameter="", term="<none>", df_fit=float(object.df_fit), tgd=float(baseline.tgd), delta_df=None)
    ]
    for parameter in requested_parameters:
        parameter_scope = None if scope is None else scope.get(parameter)
        if isinstance(parameter_scope, dict):
            parameter_scope = parameter_scope.get("lower")
        try:
            result = drop1_tgd(object, newdata=newdata, scope=parameter_scope, what=parameter)
        except ValueError as exc:
            if "no terms in scope" in str(exc):
                continue
            raise
        for row in result.rows[1:]:
            rows.append(
                MultiParameterTGDScopeRow(
                    parameter=parameter,
                    term=row.term,
                    df_fit=float(row.df_fit),
                    tgd=float(row.tgd),
                    delta_df=None if row.delta_df is None else float(row.delta_df),
                )
            )
    return MultiParameterTGDScopeResult(direction="drop", rows=tuple(rows))


def add1_tgd_all(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
) -> MultiParameterTGDScopeResult:
    """Staged multi-parameter aggregation of `add1_tgd()`."""

    _require_gamlss_method(object)
    requested_parameters = [parameter for parameter in object.par] if parameters is None else [str(parameter).strip().lower() for parameter in parameters]
    if not requested_parameters:
        raise ValueError("at least one parameter is required")

    baseline = extract_tgd_data(object, newdata=newdata)
    rows: list[MultiParameterTGDScopeRow] = [
        MultiParameterTGDScopeRow(parameter="", term="<none>", df_fit=float(object.df_fit), tgd=float(baseline.tgd), delta_df=None)
    ]
    for parameter in requested_parameters:
        parameter_scope = None if scope is None else scope.get(parameter)
        if isinstance(parameter_scope, dict):
            parameter_scope = parameter_scope.get("upper")
        if parameter_scope is None:
            continue
        try:
            result = add1_tgd(object, newdata=newdata, scope=parameter_scope, what=parameter)
        except ValueError as exc:
            if "no terms in scope" in str(exc):
                continue
            raise
        for row in result.rows[1:]:
            rows.append(
                MultiParameterTGDScopeRow(
                    parameter=parameter,
                    term=row.term,
                    df_fit=float(row.df_fit),
                    tgd=float(row.tgd),
                    delta_df=None if row.delta_df is None else float(row.delta_df),
                )
            )
    return MultiParameterTGDScopeResult(direction="add", rows=tuple(rows))


def drop1_tgdp(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
) -> MultiParameterTGDScopeResult:
    """R reference: `gamlss/R/stepTGD.R::drop1TGDP`."""

    return drop1_tgd_all(object=object, newdata=newdata, scope=scope, parameters=parameters)


def add1_tgdp(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
) -> MultiParameterTGDScopeResult:
    """R reference: `gamlss/R/stepTGD.R::add1TGDP`."""

    return add1_tgd_all(object=object, newdata=newdata, scope=scope, parameters=parameters)


def step_tgd_all(
    object: GAMLSSModel,
    newdata: Mapping[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
    direction: str = "both",
    steps: int = 100,
) -> StepTGDAllResult:
    """Staged multi-parameter `stepTGD` search."""

    _require_gamlss_method(object)
    direction_name = direction.strip().lower()
    if direction_name not in {"forward", "backward", "both"}:
        raise ValueError("direction must be one of 'forward', 'backward', or 'both'")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    if parameters is None:
        requested_parameters = [parameter for parameter in object.par]
    else:
        requested_parameters = [str(parameter).strip().lower() for parameter in parameters]
    if not requested_parameters:
        raise ValueError("at least one parameter is required")
    missing = [parameter for parameter in requested_parameters if parameter not in object.par]
    if missing:
        raise ValueError(f"parameters not present in object: {missing}")

    current = object
    current_tgd = float(extract_tgd_data(current, newdata=newdata).tgd)
    path: list[StepTGDAllStep] = [
        StepTGDAllStep(
            step=0,
            parameter="",
            formula=" | ".join(f"{parameter}: {_normalize_formula_text(formula(current, what=parameter))}" for parameter in requested_parameters),
            tgd=current_tgd,
            change="",
            df_fit=float(current.df_fit),
            deviance=float(current.g_dev),
        )
    ]

    parameter_scopes = {parameter: (scope or {}).get(parameter) for parameter in requested_parameters}
    current.additional_slots["drop1_tgd_all"] = drop1_tgd_all(current, newdata=newdata, scope=parameter_scopes, parameters=tuple(requested_parameters))
    current.additional_slots["add1_tgd_all"] = add1_tgd_all(current, newdata=newdata, scope=parameter_scopes, parameters=tuple(requested_parameters))

    for step_index in range(1, steps + 1):
        best_parameter = ""
        best_model = current
        best_change = ""
        best_formula = ""
        best_tgd = current_tgd

        for parameter in requested_parameters:
            result = step_tgd(current, newdata=newdata, scope=parameter_scopes.get(parameter), what=parameter, direction=direction_name, steps=1)
            if len(result.steps) <= 1:
                continue
            candidate_model = result.model
            candidate_step = result.steps[-1]
            candidate_tgd = float(candidate_step.tgd)
            if candidate_tgd + 1e-7 < best_tgd:
                best_parameter = parameter
                best_model = candidate_model
                best_change = candidate_step.change
                best_formula = candidate_step.formula
                best_tgd = candidate_tgd

        if not best_parameter:
            break

        current = best_model
        current_tgd = best_tgd
        current.additional_slots["step_tgd_all_path"] = tuple(path)
        current.additional_slots["drop1_tgd_all"] = drop1_tgd_all(current, newdata=newdata, scope=parameter_scopes, parameters=tuple(requested_parameters))
        current.additional_slots["add1_tgd_all"] = add1_tgd_all(current, newdata=newdata, scope=parameter_scopes, parameters=tuple(requested_parameters))
        path.append(
            StepTGDAllStep(
                step=step_index,
                parameter=best_parameter,
                formula=best_formula,
                tgd=current_tgd,
                change=best_change,
                df_fit=float(current.df_fit),
                deviance=float(current.g_dev),
            )
        )

    current.additional_slots["step_tgd_all_path"] = tuple(path)
    return StepTGDAllResult(model=current, parameters=tuple(requested_parameters), direction=direction_name, steps=tuple(path))

extractTGD = extract_tgd
drop1TGD = drop1_tgd
add1TGD = add1_tgd
stepTGD = step_tgd
drop1TGDP = drop1_tgdp
add1TGDP = add1_tgdp

__all__ = [
    "ExtractTGDResult",
    "MultiParameterTGDScopeResult",
    "MultiParameterTGDScopeRow",
    "StepTGDAllResult",
    "StepTGDAllStep",
    "StepTGDResult",
    "StepTGDStep",
    "TGDScopeResult",
    "TGDScopeRow",
    "add1TGD",
    "add1TGDP",
    "add1_tgd",
    "add1_tgd_all",
    "add1_tgdp",
    "drop1TGD",
    "drop1TGDP",
    "drop1_tgd",
    "drop1_tgd_all",
    "drop1_tgdp",
    "extractTGD",
    "extract_tgd",
    "extract_tgd_data",
    "stepTGD",
    "step_tgd",
    "step_tgd_all",
]
