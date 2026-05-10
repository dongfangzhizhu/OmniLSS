"""R-aligned pseudo R-squared interfaces.

R source reference:
- file: `gamlss/R/extra.R`
- functions: `Rsq.gamlss`
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class RSquaredResult:
    """Structured pseudo-R-squared summary for staged GAMLSS fits."""

    cox_snell: float
    cragg_uhler: float


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def rsq(
    object: GAMLSSModel,
    type: str = "both",
) -> float | RSquaredResult:
    """R reference: `gamlss/R/extra.R::Rsq.gamlss`.

    Staged behavior:
    - Builds an intercept-only null model using stored call data.
    - Returns Cox-Snell, Cragg-Uhler, or both.
    """

    _require_gamlss(object)
    requested = str(type).strip().lower()
    valid = {"cox snell", "cragg uhler", "both"}
    if requested not in valid:
        raise ValueError("type must be one of 'Cox Snell', 'Cragg Uhler', or 'both'")

    if object.call is None or "data" not in object.call:
        raise ValueError("Rsq requires stored call data to build a null model")

    response_name = object.terms.get("mu", {}).get("response")
    if response_name is None:
        formula_mu = object.formulas.get("mu")
        if formula_mu is None or "~" not in str(formula_mu):
            raise ValueError("Rsq requires a stored mu formula with response")
        response_name = str(formula_mu).split("~", 1)[0].strip()

    family = object.family
    call_data = object.call["data"]

    from .fitting import gamlss_ml
    from .logLik import log_likelihood

    null_model = gamlss_ml(
        formula=f"{response_name} ~ 1",
        family=family,
        data=call_data,
        sigma_formula="~1",
    )
    ll_null = log_likelihood(null_model).value
    ll_fit = log_likelihood(object).value
    n = max(int(object.additional_slots.get("noObs", object.n)), 1)

    cox_snell = 1.0 - math.exp((2.0 / n) * (ll_null - ll_fit))
    denom = 1.0 - math.exp((2.0 / n) * ll_null)
    if abs(denom) <= np.finfo(np.float64).eps:
        cragg_uhler = float("nan")
    else:
        cragg_uhler = cox_snell / denom

    result = RSquaredResult(
        cox_snell=float(cox_snell),
        cragg_uhler=float(cragg_uhler),
    )
    if requested == "cox snell":
        return result.cox_snell
    if requested == "cragg uhler":
        return result.cragg_uhler
    return result


# R-style exact-name alias
Rsq = rsq

__all__ = [
    "RSquaredResult",
    "Rsq",
    "rsq",
]
