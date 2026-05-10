"""R-aligned parallel-style multi-parameter GAIC interfaces.

R source reference:
- file: `gamlss/R/stepGAICAll-B-Parallel.R`
- functions: `droptermAllP`, `addtermAllP`, `stepGAICAll.B`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import GAMLSSModel


@dataclass(frozen=True)
class StepGAICAllStep:
    """One staged step in a multi-parameter `stepGAICAll` search path."""

    step: int
    parameter: str
    formula: str
    criterion: float
    change: str
    df_fit: float
    deviance: float


@dataclass(frozen=True)
class StepGAICAllResult:
    """Structured staged `stepGAICAll` result."""

    model: GAMLSSModel
    parameters: tuple[str, ...]
    direction: str
    k: float
    steps: tuple[StepGAICAllStep, ...]


@dataclass(frozen=True)
class MultiParameterScopeRow:
    """One row in a staged multi-parameter add/drop table."""

    parameter: str
    term: str
    df_fit: float
    criterion: float
    delta_df: float | None
    delta_criterion: float | None


@dataclass(frozen=True)
class MultiParameterScopeResult:
    """Structured multi-parameter add/drop model selection table."""

    direction: str
    k: float
    rows: tuple[MultiParameterScopeRow, ...]


def dropterm_all(
    object: GAMLSSModel,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
    k: float = 2.0,
) -> MultiParameterScopeResult:
    """R reference: `gamlss/R/stepGAICAll-B-Parallel.R::droptermAllP`.

    Staged behavior:
    - Aggregates single-parameter `dropterm()` results across supported parameters.
    - Currently supports `mu` and `sigma`.
    """

    from .methods import _require_gamlss_method
    from .stepGAIC_03_10_13 import extract_aic
    from .DropAddStepGAIC_Parallel import dropterm

    _require_gamlss_method(object)
    requested_parameters = (
        [parameter for parameter in object.par]
        if parameters is None
        else [str(parameter).strip().lower() for parameter in parameters]
    )
    if not requested_parameters:
        raise ValueError("at least one parameter is required")

    base_aic = float(extract_aic(object, k=k).aic)
    rows: list[MultiParameterScopeRow] = [
        MultiParameterScopeRow(
            parameter="",
            term="<none>",
            df_fit=float(object.df_fit),
            criterion=base_aic,
            delta_df=None,
            delta_criterion=None,
        )
    ]

    for parameter in requested_parameters:
        parameter_scope = None if scope is None else scope.get(parameter)
        if isinstance(parameter_scope, dict):
            parameter_scope = parameter_scope.get("lower")
        try:
            result = dropterm(object, scope=parameter_scope, what=parameter, k=k)
        except ValueError as exc:
            if "no terms in scope" in str(exc):
                continue
            raise
        for row in result.rows[1:]:
            rows.append(
                MultiParameterScopeRow(
                    parameter=parameter,
                    term=row.term,
                    df_fit=float(row.df_fit),
                    criterion=float(row.criterion),
                    delta_df=None if row.delta_df is None else float(row.delta_df),
                    delta_criterion=None if row.delta_criterion is None else float(row.delta_criterion),
                )
            )

    return MultiParameterScopeResult(direction="drop", k=float(k), rows=tuple(rows))


def addterm_all(
    object: GAMLSSModel,
    scope: dict[str, Any],
    parameters: tuple[str, ...] | list[str] | None = None,
    k: float = 2.0,
) -> MultiParameterScopeResult:
    """R reference: `gamlss/R/stepGAICAll-B-Parallel.R::addtermAllP`.

    Staged behavior:
    - Aggregates single-parameter `addterm()` results across supported parameters.
    - Currently supports `mu` and `sigma`.
    """

    from .methods import _require_gamlss_method
    from .stepGAIC_03_10_13 import extract_aic
    from .DropAddStepGAIC_Parallel import addterm

    _require_gamlss_method(object)
    requested_parameters = (
        [parameter for parameter in object.par]
        if parameters is None
        else [str(parameter).strip().lower() for parameter in parameters]
    )
    if not requested_parameters:
        raise ValueError("at least one parameter is required")

    base_aic = float(extract_aic(object, k=k).aic)
    rows: list[MultiParameterScopeRow] = [
        MultiParameterScopeRow(
            parameter="",
            term="<none>",
            df_fit=float(object.df_fit),
            criterion=base_aic,
            delta_df=None,
            delta_criterion=None,
        )
    ]

    for parameter in requested_parameters:
        parameter_scope = None if scope is None else scope.get(parameter)
        if isinstance(parameter_scope, dict):
            parameter_scope = parameter_scope.get("upper")
        if parameter_scope is None:
            continue
        result = addterm(object, scope=parameter_scope, what=parameter, k=k)
        for row in result.rows[1:]:
            rows.append(
                MultiParameterScopeRow(
                    parameter=parameter,
                    term=row.term,
                    df_fit=float(row.df_fit),
                    criterion=float(row.criterion),
                    delta_df=None if row.delta_df is None else float(row.delta_df),
                    delta_criterion=None if row.delta_criterion is None else float(row.delta_criterion),
                )
            )

    return MultiParameterScopeResult(direction="add", k=float(k), rows=tuple(rows))


def step_gaic_all(
    object: GAMLSSModel,
    scope: dict[str, Any] | None = None,
    parameters: tuple[str, ...] | list[str] | None = None,
    direction: str = "both",
    k: float = 2.0,
    steps: int = 100,
) -> StepGAICAllResult:
    """R references:
    - `gamlss/R/stepGAICAll-A-parallel.R::stepGAICAll.A`
    - `gamlss/R/stepGAICAll-B-Parallel.R::stepGAICAll.B`

    Staged behavior:
    - Rotates through supported parameters and applies `step_gaic` one move at a time.
    - Supports parameters present in the fitted object.
    """

    from .methods import _require_gamlss_method, _normalize_formula_text
    from .stepGAIC_03_10_13 import extract_aic
    from .DropAddStepGAIC_Parallel import step_gaic
    from .operations import formula

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
    current_aic = float(extract_aic(current, k=k).aic)
    path: list[StepGAICAllStep] = [
        StepGAICAllStep(
            step=0,
            parameter="",
            formula=" | ".join(
                f"{parameter}: {_normalize_formula_text(formula(current, what=parameter))}" for parameter in requested_parameters
            ),
            criterion=current_aic,
            change="",
            df_fit=float(current.df_fit),
            deviance=float(current.g_dev),
        )
    ]

    parameter_scopes = {parameter: (scope or {}).get(parameter) for parameter in requested_parameters}
    current.additional_slots["dropterm_all"] = dropterm_all(
        current,
        scope=parameter_scopes,
        parameters=tuple(requested_parameters),
        k=k,
    )
    current.additional_slots["addterm_all"] = addterm_all(
        current,
        scope=parameter_scopes,
        parameters=tuple(requested_parameters),
        k=k,
    )

    for step_index in range(1, steps + 1):
        best_parameter = ""
        best_model = current
        best_change = ""
        best_formula = ""
        best_aic = current_aic

        for parameter in requested_parameters:
            result = step_gaic(
                current,
                scope=parameter_scopes.get(parameter),
                what=parameter,
                direction=direction_name,
                k=k,
                steps=1,
            )
            if len(result.steps) <= 1:
                continue
            candidate_model = result.model
            candidate_step = result.steps[-1]
            candidate_aic = float(candidate_step.criterion)
            if candidate_aic + 1e-7 < best_aic:
                best_parameter = parameter
                best_model = candidate_model
                best_change = candidate_step.change
                best_formula = candidate_step.formula
                best_aic = candidate_aic

        if not best_parameter:
            break

        current = best_model
        current_aic = best_aic
        current.additional_slots["step_gaic_all_path"] = tuple(path)
        current.additional_slots["dropterm_all"] = dropterm_all(
            current,
            scope=parameter_scopes,
            parameters=tuple(requested_parameters),
            k=k,
        )
        current.additional_slots["addterm_all"] = addterm_all(
            current,
            scope=parameter_scopes,
            parameters=tuple(requested_parameters),
            k=k,
        )
        path.append(
            StepGAICAllStep(
                step=step_index,
                parameter=best_parameter,
                formula=best_formula,
                criterion=current_aic,
                change=best_change,
                df_fit=float(current.df_fit),
                deviance=float(current.g_dev),
            )
        )

    current.additional_slots["step_gaic_all_path"] = tuple(path)
    return StepGAICAllResult(
        model=current,
        parameters=tuple(requested_parameters),
        direction=direction_name,
        k=float(k),
        steps=tuple(path),
    )


droptermAllP = dropterm_all
addtermAllP = addterm_all
stepGAICAll_B = step_gaic_all

__all__ = [
    "MultiParameterScopeResult",
    "MultiParameterScopeRow",
    "StepGAICAllResult",
    "StepGAICAllStep",
    "addtermAllP",
    "addterm_all",
    "droptermAllP",
    "dropterm_all",
    "step_gaic_all",
    "stepGAICAll_B",
]
