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
        current_terms = _rhs_terms(
            _normalize_formula_text(current.formulas.get(selected, ""))
        )
        drop_candidates = (
            [term for term in current_terms if term not in lower] if backward else []
        )
        add_candidates = (
            [term for term in upper if term not in current_terms] if forward else []
        )

        candidates: list[tuple[str, float, str]] = []
        if drop_candidates:
            for row in dropterm(
                current, scope=drop_candidates, what=selected, k=k
            ).rows[1:]:
                if row.criterion == row.criterion:
                    candidates.append((row.term, float(row.criterion), "drop"))
        if add_candidates:
            for row in addterm(current, scope=add_candidates, what=selected, k=k).rows[
                1:
            ]:
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


@dataclass(frozen=True)
class StepGAICCHRow:
    """One row of R ``stepGAIC.CH``'s analysis-of-deviance path table."""

    from_term: str
    to_term: str
    df: float
    deviance: float
    resid_df: float
    resid_dev: float
    aic: float


@dataclass(frozen=True)
class StepGAICCHResult:
    """R ``stepGAIC.CH`` return value: final model plus regimen path table."""

    rows: tuple[StepGAICCHRow, ...]
    heading: tuple[str, ...]
    model: GAMLSSModel
    what: str
    keep: tuple[Any, ...] = ()

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "From": row.from_term,
                    "To": row.to_term,
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


def _scope_char(item: Any) -> list[str]:
    """Mirror R ``stepGAIC.CH``'s ``scope.char`` helper."""
    if isinstance(item, (list, tuple)):
        return [str(tok).strip().replace(" ", "") for tok in item if str(tok).strip()]
    text = str(item).strip()
    if "~" in text:
        text = text.split("~", 1)[1]
    return [tok.strip().replace(" ", "") for tok in text.split("+") if tok.strip()]


def _normalize_term_key(term: str) -> str:
    return str(term).strip().replace(" ", "")


def _untangle_scope(
    term_labels: list[str],
    regimens: list[list[str]],
) -> tuple[list[str], list[int]]:
    """Mirror R ``untangle.scope``: locate each regimen's initial level (0-based)."""
    remaining = [_normalize_term_key(t) for t in term_labels]
    select: list[int] = []
    for idx, reg in enumerate(regimens):
        reg_norm = [_normalize_term_key(t) for t in reg]
        hits = [pos for pos, rterm in enumerate(reg_norm) if rterm in remaining]
        if len(hits) > 1:
            raise ValueError(
                f"The elements of regimen {idx + 1} appear more than once in the initial model"
            )
        if len(hits) == 1:
            pos = hits[0]
            select.append(pos)
            remaining.remove(reg_norm[pos])
        else:
            if "1" not in reg_norm:
                raise ValueError(
                    f"regimen {idx + 1} does not appear in the initial model"
                )
            select.append(reg_norm.index("1"))
    return remaining, select


def _compose_ch_formula(
    response: str,
    base_terms: list[str],
    chosen_terms: list[str],
    *,
    include_response: bool,
) -> str:
    active: list[str] = []
    for term in [*base_terms, *chosen_terms]:
        cleaned = str(term).strip()
        if cleaned and cleaned != "1" and cleaned not in active:
            active.append(cleaned)
    rhs = " + ".join(active) if active else "1"
    return f"{response} ~ {rhs}" if include_response else f"~ {rhs}"


def stepGAIC_CH(
    object: GAMLSSModel,
    scope: Any | None = None,
    direction: str = "both",
    trace: bool = False,
    keep: Any | None = None,
    steps: int = 1000,
    what: str = "mu",
    parameter: str | None = None,
    k: float = 2.0,
) -> StepGAICCHResult:
    """R reference: ``gamlss/R/stepGAIC-03-10-13..R::stepGAIC.CH``.

    Chambers-Hastie (``step.gam``) regimen-grid stepwise model selection for a
    single GAMLSS distribution parameter.
    """
    from .gamlss_scope import gamlss_scope

    _require_gamlss_method(object)
    selected = str(parameter if parameter is not None else what).strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")

    direction_name = str(direction).strip().lower()
    if direction_name not in {"both", "backward", "forward"}:
        raise ValueError("direction must be one of 'both', 'backward', or 'forward'")
    backward = direction_name in {"both", "backward"}
    forward = direction_name in {"both", "forward"}

    if scope is None:
        call_data = None if object.call is None else object.call.get("data")
        if call_data is None:
            raise ValueError("stepGAIC_CH requires scope or object.call['data']")
        mu_fml = _normalize_formula_text(object.formulas.get("mu", ""))
        resp_col = _response_name_from_formula(mu_fml)
        scope = gamlss_scope(call_data, response=resp_col, form=True)

    raw_regimens = list(scope.values()) if isinstance(scope, dict) else list(scope)
    if not raw_regimens:
        raise ValueError("scope must contain at least one regimen")
    regimens = [_scope_char(reg) for reg in raw_regimens]

    initial_formula = _normalize_formula_text(object.formulas.get(selected, ""))
    mu_formula = _normalize_formula_text(object.formulas.get("mu", ""))
    response = _response_name_from_formula(mu_formula)
    initial_terms = _rhs_terms(initial_formula)
    base_terms, items = _untangle_scope(initial_terms, regimens)

    n = int(object.n)
    n_items = len(items)
    visited: set[tuple[int, ...]] = {tuple(items)}
    form_vector = [regimens[i][items[i]] for i in range(n_items)]

    fit = object
    b_aic = float(extract_aic(fit, k=k).aic)
    kept: list[Any] = []
    if callable(keep):
        kept.append(keep(fit, b_aic))

    rows: list[StepGAICCHRow] = [
        StepGAICCHRow(
            from_term="",
            to_term="",
            df=float("nan"),
            deviance=float("nan"),
            resid_df=float(n) - float(fit.df_fit),
            resid_dev=float(fit.deviance),
            aic=b_aic,
        )
    ]
    if trace:
        print(f"Distribution parameter: {selected}")
        print(f"Start: {initial_formula}; AIC={b_aic:.4f}")

    prev_aic = b_aic + 1.0
    remaining_steps = int(steps)

    while b_aic < prev_aic and remaining_steps > 0:
        remaining_steps -= 1
        prev_aic = b_aic
        best_items = list(items)
        best_fit = fit
        best_vector = list(form_vector)
        best_from = ""
        best_to = ""

        for i in range(n_items):
            candidates: list[int] = []
            if backward and items[i] - 1 >= 0:
                candidates.append(items[i] - 1)
            if forward and items[i] + 1 < len(regimens[i]):
                candidates.append(items[i] + 1)

            for next_lvl in candidates:
                trial = list(items)
                trial[i] = next_lvl
                trial_key = tuple(trial)
                if trial_key in visited:
                    continue
                visited.add(trial_key)
                t_vector = list(form_vector)
                t_vector[i] = regimens[i][next_lvl]
                cand_formula = _compose_ch_formula(
                    response,
                    base_terms,
                    t_vector,
                    include_response=(selected == "mu"),
                )
                updated_map = _updated_formulas_for_parameter(
                    object, selected, cand_formula
                )
                t_fit = _fit_with_updated_formula(object, updated_map)
                t_aic = float(extract_aic(t_fit, k=k).aic)
                if callable(keep):
                    kept.append(keep(t_fit, t_aic))
                if trace:
                    print(f"Trial: {cand_formula}; AIC={t_aic:.4f}")
                if t_aic < b_aic - 1e-12:
                    b_aic = t_aic
                    best_items = trial
                    best_fit = t_fit
                    best_vector = t_vector
                    best_from = form_vector[i]
                    best_to = t_vector[i]

        if b_aic >= prev_aic:
            break

        items = best_items
        fit = best_fit
        form_vector = best_vector
        cur_resid_df = float(n) - float(fit.df_fit)
        cur_resid_dev = float(fit.deviance)
        rows.append(
            StepGAICCHRow(
                from_term="" if best_from == "1" else best_from,
                to_term="" if best_to == "1" else best_to,
                df=cur_resid_df - rows[-1].resid_df,
                deviance=cur_resid_dev - rows[-1].resid_dev,
                resid_df=cur_resid_df,
                resid_dev=cur_resid_dev,
                aic=b_aic,
            )
        )
        if trace:
            print(f"Step : {fit.formulas.get(selected, '')}; AIC={b_aic:.4f}")

    final_formula = _normalize_formula_text(fit.formulas.get(selected, ""))
    heading = (
        "Stepwise Model Path \nAnalysis of Deviance Table",
        "\nInitial Model:",
        initial_formula,
        "\nFinal Model:",
        final_formula,
        "\n",
    )
    return StepGAICCHResult(
        rows=tuple(rows),
        heading=heading,
        model=fit,
        what=selected,
        keep=tuple(kept),
    )


step_gaic_ch = stepGAIC_CH


__all__ = [
    "StepGAICCHResult",
    "StepGAICCHRow",
    "StepwisePathResult",
    "StepwisePathRow",
    "stepGAIC_CH",
    "stepGAIC_VR",
    "step_gaic_ch",
    "step_gaic_vr",
]
