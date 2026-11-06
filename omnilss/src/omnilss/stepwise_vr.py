"""R stepGAIC.VR: Venables-Ripley stepwise selection for gamlss models.

R source: R/gamlss/R/stepGAIC-03-10-13..R::stepGAIC.VR.

The R function is MASS::stepAIC adapted to gamlss: at every step it builds one
table holding the drop candidates ("- term") and the add candidates ("+ term"),
sorts it by AIC, and moves to the best row unless the current model is already
best.  The selected path is returned as an analysis-of-deviance table with the
columns Step / Df / Deviance / Resid. Df / Resid. Dev / AIC.

Compared with live R on a NO model: the anova path (step labels, Df, deviance,
residual df, AIC) and the final formula agree to the fitting tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .DropAddStepGAIC_Parallel import (
    _compose_formula,
    _fit_with_updated_formula,
    _normalize_formula_text,
    _parse_scope_spec,
    _require_gamlss_method,
    _response_name_from_formula,
    _rhs_terms,
    _updated_formulas_for_parameter,
    addterm,
    dropterm,
)
from .model import GAMLSSModel
from .stepGAIC_03_10_13 import extract_aic


@dataclass(frozen=True)
class StepwisePathRow:
    """One row of the analysis-of-deviance path table."""

    step: str
    df: float
    deviance: float
    resid_df: float
    resid_dev: float
    aic: float


@dataclass(frozen=True)
class StepwisePathResult:
    """R's stepGAIC.VR() return value: the final model plus the path table."""

    rows: tuple[StepwisePathRow, ...]
    heading: tuple[str, ...]
    model: GAMLSSModel
    what: str
    term_labels: tuple[str, ...] = field(default_factory=tuple)

    def to_frame(self) -> pd.DataFrame:
        """The R anova table, with the same column names and row order."""

        frame = pd.DataFrame(
            [
                {
                    "Step": row.step,
                    "Df": row.df,
                    "Deviance": row.deviance,
                    "Resid. Df": row.resid_df,
                    "Resid. Dev": row.resid_dev,
                    "AIC": row.aic,
                }
                for row in self.rows
            ]
        )
        frame.attrs["heading"] = list(self.heading)
        return frame


def stepGAIC_VR(
    object: GAMLSSModel,
    scope: Any | None = None,
    direction: str = "both",
    what: str = "mu",
    k: float = 2.0,
    steps: int = 1000,
    trace: bool = False,
) -> StepwisePathResult:
    """R stepGAIC.VR: stepwise selection by AIC over whole terms."""

    _require_gamlss_method(object)
    selected = str(what).strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")

    direction_name = str(direction).strip().lower()
    if direction_name not in {"both", "backward", "forward"}:
        raise ValueError("direction must be one of 'both', 'backward', or 'forward'")
    backward = direction_name in {"both", "backward"}
    forward = direction_name in {"both", "forward"}
    lower, upper = _parse_scope_spec(scope)
    if scope is None:
        # R disables the forward direction when scope is missing.
        forward = False

    initial_formula = _normalize_formula_text(object.formulas.get(selected, ""))
    response = _response_name_from_formula(initial_formula)
    # R resolves "." against the model formula when it evaluates update.formula(),
    # so the scope keeps the current terms plus whatever was listed.
    initial_terms = _rhs_terms(initial_formula)

    def expand(terms: list[str]) -> list[str]:
        resolved: list[str] = []
        for term in terms:
            for name in initial_terms if term == "." else [term]:
                if name not in resolved:
                    resolved.append(name)
        return resolved

    lower = expand(lower)
    upper = expand(upper)
    current = object
    n = int(object.n)

    def aic_of(model: GAMLSSModel) -> float:
        return float(extract_aic(model, k=k).aic)

    base_aic = aic_of(current)
    rows: list[StepwisePathRow] = [
        StepwisePathRow(
            step="",
            df=float("nan"),
            deviance=float("nan"),
            resid_df=float(n) - float(current.df_fit),
            resid_dev=float(current.deviance),
            aic=base_aic,
        )
    ]

    while steps > 0:
        steps -= 1
        previous_aic = base_aic
        current_terms = _rhs_terms(_normalize_formula_text(current.formulas.get(selected, "")))
        drop_candidates = [term for term in current_terms if term not in lower] if backward else []
        add_candidates = [term for term in upper if term not in current_terms] if forward else []

        candidates: list[tuple[str, float, str]] = []
        if drop_candidates:
            for row in dropterm(current, scope=drop_candidates, what=selected, k=k).rows[1:]:
                if row.criterion == row.criterion:
                    candidates.append((row.term, float(row.criterion), "drop"))
        if add_candidates:
            for row in addterm(current, scope=add_candidates, what=selected, k=k).rows[1:]:
                if row.criterion == row.criterion:
                    candidates.append((row.term, float(row.criterion), "add"))
        if not candidates:
            break

        candidates.sort(key=lambda item: item[1])
        label, best_aic, kind = candidates[0]
        if best_aic + 1e-7 >= previous_aic:
            break

        term_name = label[2:].strip()
        if kind == "drop":
            updated_terms = [term for term in current_terms if term != term_name]
        else:
            updated_terms = [*current_terms, term_name]
        updated_formula = _compose_formula(response, updated_terms)
        fitted_model = _fit_with_updated_formula(
            current, _updated_formulas_for_parameter(current, selected, updated_formula)
        )
        fitted_aic = aic_of(fitted_model)
        if fitted_aic >= previous_aic + 1e-7:
            break

        current = fitted_model
        base_aic = fitted_aic
        rows.append(
            StepwisePathRow(
                step=label,
                df=abs(float(n) - float(current.df_fit) - rows[-1].resid_df),
                deviance=abs(float(current.deviance) - rows[-1].resid_dev),
                resid_df=float(n) - float(current.df_fit),
                resid_dev=float(current.deviance),
                aic=fitted_aic,
            )
        )
        if trace:
            print(f"Step:  AIC={fitted_aic:.6g}  {updated_formula}")

    final_formula = _normalize_formula_text(current.formulas.get(selected, ""))
    heading = (
        "Stepwise Model Path \nAnalysis of Deviance Table",
        "\nInitial",
        selected,
        " Model:",
        initial_formula,
        "\nFinal",
        selected,
        " Model:",
        final_formula,
        "\n",
    )
    return StepwisePathResult(
        rows=tuple(rows),
        heading=heading,
        model=current,
        what=selected,
        term_labels=tuple(row.step for row in rows),
    )


# Pythonic alias; the R name is the primary export.
step_gaic_vr = stepGAIC_VR


__all__ = ["StepwisePathResult", "StepwisePathRow", "stepGAIC_VR", "step_gaic_vr"]