"""R gamlss drop1All() / add1All(): single-term deletion/addition tables.

R source: R/gamlss/R/stepGAICAll-B-Parallel.R.

Both functions return an "anova"-class data.frame whose first row is <none>
(the current model) and whose remaining rows are the candidate terms.  The
columns are Df (change in effective df), AIC, and -- with test="Chisq" -- LRT
and Pr(Chi).  The refitting is delegated to the existing R-aligned dropterm /
addterm implementations, so this module is the R table-format adapter.

R refits with what="All"; the engine's dropterm/addterm refits the mu formula
(keeping the other parameter formulas), which is equivalent for the usual
models where only mu carries the terms being considered.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .DropAddStepGAIC_Parallel import addterm, dropterm
from .operations import formula

_TESTS = ("Chisq", "none")


def _check_test(test: str) -> str:
    if test not in _TESTS:
        raise ValueError(f"'arg' should be one of {list(_TESTS)}, got {test!r}")
    return test


def _table(
    result: Any,
    heading: str,
    model_formula: str,
    k: float,
    test: str,
    sorted_: bool,
    prefix: str,
    direction: str,
) -> pd.DataFrame:
    rows = result.rows
    base_df = float(rows[0].df_fit)
    base_aic = float(rows[0].criterion)

    records: list[dict[str, Any]] = []
    for row in rows:
        if row.term == "<none>":
            records.append(
                {
                    "term": "<none>",
                    "Df": np.nan,
                    "AIC": base_aic,
                    "LRT": np.nan,
                    "Pr(Chi)": np.nan,
                    "delta_criterion": 0.0,
                    "delta_df": 0.0,
                }
            )
        else:
            term = (
                row.term[len(prefix) :].strip()
                if row.term.startswith(prefix)
                else row.term
            )
            delta_df = 0.0 if row.delta_df is None else float(row.delta_df)
            delta_criterion = (
                0.0 if row.delta_criterion is None else float(row.delta_criterion)
            )
            df_change = base_df - float(row.df_fit)
            lrt = delta_criterion - k * delta_df
            if direction == "add":
                # R's add1All reverses both the df difference and the LRT sign.
                df_change = -df_change
                lrt = -lrt
            p_value = (
                float(chi2.sf(lrt, df_change))
                if np.isfinite(lrt) and df_change > 0
                else np.nan
            )
            records.append(
                {
                    "term": term,
                    "Df": df_change,
                    "AIC": float(row.criterion),
                    "LRT": lrt,
                    "Pr(Chi)": p_value,
                    "delta_criterion": delta_criterion,
                    "delta_df": delta_df,
                }
            )

    table = pd.DataFrame.from_records(records)
    if test == "Chisq":
        table.loc[0, "LRT"] = np.nan
        table.loc[0, "Pr(Chi)"] = np.nan
        columns = ["term", "Df", "AIC", "LRT", "Pr(Chi)"]
    else:
        columns = ["term", "Df", "AIC"]
    table = table[columns]
    if sorted_:
        table = table.sort_values("AIC").reset_index(drop=True)
    table = table.set_index("term")
    table.index.name = None
    table.attrs["heading"] = [heading, "\nModel:", model_formula]
    return table


def drop1All(
    object: Any,
    scope: Any | None = None,
    test: str = "Chisq",
    k: float = 2.0,
    sorted: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """R drop1All: single-term deletions with AIC (and optional LRT) table."""

    test = _check_test(test)
    result = dropterm(object, scope=scope, what="mu", k=k)
    return _table(
        result,
        "Single term deletions",
        str(formula(object, what="mu")),
        float(k),
        test,
        bool(sorted),
        "- ",
        "drop",
    )


def _scope_terms(scope: Any) -> list[str]:
    if isinstance(scope, str):
        text = scope.strip()
        if text.startswith("~"):
            text = text[1:].strip()
        return [
            term.strip()
            for term in text.split("+")
            if term.strip() and term.strip() != "1"
        ]
    return [str(term).strip() for term in scope if str(term).strip()]


def add1All(
    object: Any,
    scope: Any,
    test: str = "Chisq",
    k: float = 2.0,
    sorted: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """R add1All: single-term additions with AIC (and optional LRT) table.

    R refits with what="All", i.e. the candidate term is added to *every*
    distribution parameter formula (mu and sigma for a two-parameter family),
    which is why the df change can exceed one.  The engine's addterm only
    touches mu, so the refits are performed here with all formulas updated.
    """

    from types import SimpleNamespace

    from .DropAddStepGAIC_Parallel import (
        _compose_formula,
        _fit_with_updated_formula,
        _normalize_formula_text,
        _response_name_from_formula,
        _rhs_terms,
    )
    from .stepGAIC_03_10_13 import extract_aic

    if scope is None:
        raise ValueError("no terms in scope")
    test = _check_test(test)
    candidates = _scope_terms(scope)
    if not candidates:
        raise ValueError("no terms in scope for adding to object")

    mu_formula = _normalize_formula_text(formula(object, what="mu"))
    response = _response_name_from_formula(mu_formula)
    base_terms = _rhs_terms(mu_formula)
    candidates = [term for term in candidates if term not in base_terms]
    if not candidates:
        raise ValueError("no terms in scope for adding to object")

    base_aic = float(extract_aic(object, k=k).aic)
    base_df = float(object.df_fit)
    rows = [
        SimpleNamespace(
            term="<none>",
            df_fit=base_df,
            criterion=base_aic,
            delta_df=0.0,
            delta_criterion=0.0,
        )
    ]
    for term in candidates:
        updates: dict[str, str] = {}
        for parameter, raw in object.formulas.items():
            text = _normalize_formula_text(raw)
            if parameter == "mu":
                updates[parameter] = _compose_formula(response, [*base_terms, term])
            else:
                others = [item for item in _rhs_terms(text) if item != "1"]
                updates[parameter] = "~ " + " + ".join(["1", *others, term])
        fitted = _fit_with_updated_formula(object, updates)
        aic = float(extract_aic(fitted, k=k).aic)
        rows.append(
            SimpleNamespace(
                term=f"+ {term}",
                df_fit=float(fitted.df_fit),
                criterion=aic,
                delta_df=float(fitted.df_fit) - base_df,
                delta_criterion=aic - base_aic,
            )
        )

    result = SimpleNamespace(rows=tuple(rows))
    return _table(
        result,
        "Single term additions for",
        str(formula(object, what="mu")),
        float(k),
        test,
        bool(sorted),
        "+ ",
        "add",
    )


__all__ = ["add1All", "drop1All"]
