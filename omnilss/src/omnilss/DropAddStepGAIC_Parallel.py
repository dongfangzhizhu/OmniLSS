"""R-aligned drop/add/step GAIC interfaces.

R source reference:
- file: `gamlss/R/DropAddStepGAIC-Parallel.R`
- functions: `dropterm.gamlss`, `addterm.gamlss`, `stepGAIC`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import GAMLSSModel
from .operations import formula, is_gamlss, terms
from .stepGAIC_03_10_13 import ExtractAICResult, StepGAICResult, StepGAICStep, extract_aic


@dataclass(frozen=True)
class ScopeSelectionRow:
    """One row in a staged add/drop table."""

    term: str
    df_fit: float
    criterion: float
    delta_df: float | None
    delta_criterion: float | None


@dataclass(frozen=True)
class ScopeSelectionResult:
    """Structured add/drop-term model selection table."""

    what: str
    direction: str
    k: float
    rows: tuple[ScopeSelectionRow, ...]


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


def dropterm(
    object: GAMLSSModel,
    scope: Any | None = None,
    what: str = "mu",
    k: float = 2.0,
) -> ScopeSelectionResult:
    """R reference: `gamlss/R/DropAddStepGAIC-Parallel.R::dropterm.gamlss`."""

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

    base_aic = extract_aic(object, k=k).aic
    rows = [ScopeSelectionRow(term="<none>", df_fit=float(object.df_fit), criterion=float(base_aic), delta_df=None, delta_criterion=None)]
    for term_name in candidate_terms:
        if term_name not in base_terms:
            continue
        new_terms = [term for term in base_terms if term != term_name]
        updated_formula = _compose_formula(response, new_terms)
        fitted_model = _fit_with_updated_formula(object, _updated_formulas_for_parameter(object, selected, updated_formula))
        result = extract_aic(fitted_model, k=k)
        rows.append(
            ScopeSelectionRow(
                term=f"- {term_name}",
                df_fit=float(fitted_model.df_fit),
                criterion=float(result.aic),
                delta_df=float(fitted_model.df_fit - object.df_fit),
                delta_criterion=float(result.aic - base_aic),
            )
        )
    return ScopeSelectionResult(what=selected, direction="drop", k=float(k), rows=tuple(rows))


def addterm(
    object: GAMLSSModel,
    scope: Any,
    what: str = "mu",
    k: float = 2.0,
) -> ScopeSelectionResult:
    """R reference: `gamlss/R/DropAddStepGAIC-Parallel.R::addterm.gamlss`."""

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

    base_aic = extract_aic(object, k=k).aic
    rows = [ScopeSelectionRow(term="<none>", df_fit=float(object.df_fit), criterion=float(base_aic), delta_df=None, delta_criterion=None)]
    for term_name in candidate_terms:
        if term_name in base_terms:
            continue
        updated_formula = _compose_formula(response, [*base_terms, term_name])
        fitted_model = _fit_with_updated_formula(object, _updated_formulas_for_parameter(object, selected, updated_formula))
        result = extract_aic(fitted_model, k=k)
        rows.append(
            ScopeSelectionRow(
                term=f"+ {term_name}",
                df_fit=float(fitted_model.df_fit),
                criterion=float(result.aic),
                delta_df=float(fitted_model.df_fit - object.df_fit),
                delta_criterion=float(result.aic - base_aic),
            )
        )
    return ScopeSelectionResult(what=selected, direction="add", k=float(k), rows=tuple(rows))


def step_gaic(
    object: GAMLSSModel,
    scope: Any | None = None,
    what: str = "mu",
    direction: str = "both",
    k: float = 2.0,
    steps: int = 100,
) -> StepGAICResult:
    """R reference: `gamlss/R/DropAddStepGAIC-Parallel.R::stepGAIC`."""

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
    current_aic = extract_aic(current, k=k).aic
    lower_scope, upper_scope = _parse_scope_spec(scope)

    path: list[StepGAICStep] = [
        StepGAICStep(step=0, formula=current_formula, criterion=float(current_aic), change="", df_fit=float(current.df_fit), deviance=float(current.g_dev))
    ]

    backward = direction_name in {"backward", "both"}
    forward = direction_name in {"forward", "both"}
    if scope is None and direction_name != "forward":
        forward = False

    for step_index in range(1, steps + 1):
        current_formula = _normalize_formula_text(formula(current, what=selected))
        current_terms = _rhs_terms(current_formula)
        current_aic = extract_aic(current, k=k).aic

        candidate_drop_terms = [term for term in current_terms if term not in lower_scope] if backward else []
        candidate_add_terms = [term for term in upper_scope if term not in current_terms] if forward else []

        best_change = ""
        best_aic = float(current_aic)
        best_model = current

        if candidate_drop_terms:
            drop_result = dropterm(current, scope=candidate_drop_terms, what=selected, k=k)
            for row in drop_result.rows[1:]:
                if row.criterion + 1e-7 < best_aic:
                    term_name = row.term.removeprefix("- ").strip()
                    updated_terms = [term for term in current_terms if term != term_name]
                    updated_formula = _compose_formula(_response_name_from_formula(current_formula), updated_terms)
                    best_model = _fit_with_updated_formula(current, _updated_formulas_for_parameter(current, selected, updated_formula))
                    best_aic = float(row.criterion)
                    best_change = row.term

        if candidate_add_terms:
            add_result = addterm(current, scope=candidate_add_terms, what=selected, k=k)
            for row in add_result.rows[1:]:
                if row.criterion + 1e-7 < best_aic:
                    term_name = row.term.removeprefix("+ ").strip()
                    updated_terms = [*current_terms, term_name]
                    updated_formula = _compose_formula(_response_name_from_formula(current_formula), updated_terms)
                    best_model = _fit_with_updated_formula(current, _updated_formulas_for_parameter(current, selected, updated_formula))
                    best_aic = float(row.criterion)
                    best_change = row.term

        if not best_change:
            break

        current = best_model
        path.append(
            StepGAICStep(
                step=step_index,
                formula=_normalize_formula_text(formula(current, what=selected)),
                criterion=float(best_aic),
                change=best_change,
                df_fit=float(current.df_fit),
                deviance=float(current.g_dev),
            )
        )

    current.additional_slots["step_gaic_path"] = tuple(path)
    return StepGAICResult(model=current, what=selected, direction=direction_name, k=float(k), steps=tuple(path))

stepGAIC = step_gaic

__all__ = [
    "ScopeSelectionResult",
    "ScopeSelectionRow",
    "StepGAICResult",
    "StepGAICStep",
    "addterm",
    "dropterm",
    "stepGAIC",
    "step_gaic",
]
