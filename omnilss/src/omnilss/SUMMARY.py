"""R-aligned summary interfaces.

R source reference:
- file: `gamlss/R/SUMMARY.R`
- functions: `summary.gamlss`
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from .model import GAMLSSModel
from .operations import coef, gaic
from .vcov_gamlss import vcov


def _require_gamlss_method(object: GAMLSSModel) -> None:
    """Validate that the object is a GAMLSSModel."""
    if not hasattr(object, "g_dev"):
        raise TypeError("object must be a GAMLSSModel")


def _normal_two_sided_pvalue(z_value: np.ndarray) -> np.ndarray:
    """Compute two-sided p-values from z-statistics."""
    distribution = NormalDist()
    flattened = np.asarray(np.abs(z_value), dtype=np.float64).ravel()
    pvals = np.array(
        [2.0 * (1.0 - distribution.cdf(float(value))) for value in flattened],
        dtype=np.float64,
    )
    return pvals.reshape(np.asarray(z_value).shape)


@dataclass(frozen=True)
class SummaryResult:
    """Compact summary payload for a fitted staged GAMLSS model."""

    family: str
    method: str
    nobs: int
    df_fit: float
    residual_df: float
    iteration: int
    global_deviance: float
    aic: float
    sbc: float
    converged: bool
    cycles: int
    coefficients: dict[str, dict[str, Any]]


def summary(
    object: GAMLSSModel,
    type: str = "vcov",
) -> SummaryResult:
    """R reference: `gamlss/R/SUMMARY.R::summary.gamlss`.

    The staged port returns structured data instead of printing directly.
    """

    _require_gamlss_method(object)
    cov_all = vcov(object, type="all") if type == "vcov" else None
    family_name = getattr(object.family, "name", str(object.family))
    method_name = str(object.additional_slots.get("method", "RS"))
    nobs = int(object.additional_slots.get("noObs", object.n))
    residual_df = float(object.additional_slots.get("df.residual", nobs - object.df_fit))
    aic = float(object.additional_slots.get("aic", gaic(object, k=2.0)))
    sbc = float(object.additional_slots.get("sbc", gaic(object, k=float(np.log(max(nobs, 1))))))

    coefficients: dict[str, dict[str, Any]] = {}
    cursor = 0
    links = getattr(object.family, "links", None) or {}
    for parameter in object.par:
        estimate = np.asarray(coef(object, parameter), dtype=np.float64).ravel()
        count = estimate.size
        parameter_se = None
        if cov_all is not None and count:
            parameter_se = cov_all["se"][cursor : cursor + count]
        coefficients[parameter] = {
            "estimate": estimate,
            "std_error": parameter_se,
            "t_value": None if parameter_se is None else np.divide(
                estimate,
                parameter_se,
                out=np.full_like(estimate, np.nan, dtype=np.float64),
                where=np.asarray(parameter_se) != 0,
            ),
            "p_value": None
            if parameter_se is None
            else _normal_two_sided_pvalue(
                np.divide(
                    estimate,
                    parameter_se,
                    out=np.full_like(estimate, np.nan, dtype=np.float64),
                    where=np.asarray(parameter_se) != 0,
                )
            ),
            "link": links.get(parameter),
        }
        cursor += count

    return SummaryResult(
        family=family_name,
        method=method_name,
        nobs=nobs,
        df_fit=float(object.df_fit),
        residual_df=residual_df,
        iteration=int(object.iter),
        global_deviance=float(object.additional_slots.get("G.deviance", object.g_dev)),
        aic=aic,
        sbc=sbc,
        converged=bool(object.additional_slots.get("converged", True)),
        cycles=int(object.additional_slots.get("cycles", object.iter)),
        coefficients=coefficients,
    )


__all__ = [
    "SummaryResult",
    "summary",
]