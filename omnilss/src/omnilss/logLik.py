"""R-aligned log-likelihood interfaces.

R source reference:
- file: `gamlss/R/logLik.R`
- functions: `logLik.gamlss`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import GAMLSSModel


def _require_gamlss_method(object: GAMLSSModel) -> None:
    """Validate that the object is a GAMLSSModel."""
    if not hasattr(object, "g_dev"):
        raise TypeError("object must be a GAMLSSModel")


@dataclass(frozen=True)
class LogLikResult:
    """Python representation of the R `logLik` return value."""

    value: float
    nall: int
    nobs: int
    df: float


def log_likelihood(object: GAMLSSModel) -> LogLikResult:
    """R reference: `gamlss/R/logLik.R::logLik.gamlss`."""

    _require_gamlss_method(object)
    nobs = int(object.additional_slots.get("noObs", object.n))
    return LogLikResult(
        value=-float(object.additional_slots.get("G.deviance", object.g_dev)) / 2.0,
        nall=int(object.n),
        nobs=nobs,
        df=float(object.df_fit),
    )


# R-style alias
logLik = log_likelihood


__all__ = [
    "LogLikResult",
    "logLik",
    "log_likelihood",
]