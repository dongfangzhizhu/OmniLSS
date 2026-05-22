"""R-aligned tail diagnostics helpers.

R reference: `gamlss/R/FitTail.R::{loglogSurv, logSurv}`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .ECDF import ECDF


@dataclass(frozen=True)
class TailCurveResult:
    x: np.ndarray
    curve: np.ndarray


def logSurv(y) -> TailCurveResult:
    """Return log survival curve: log(1 - F_n(x))."""
    ecdf = ECDF(y)
    eps = np.finfo(np.float64).eps
    surv = np.clip(1.0 - ecdf.p, eps, 1.0)
    return TailCurveResult(x=ecdf.x, curve=np.log(surv))


def loglogSurv(y) -> TailCurveResult:
    """Return log(-log(survival)) curve used in tail diagnostics."""
    ls = logSurv(y)
    return TailCurveResult(x=ls.x, curve=np.log(-ls.curve))


__all__ = ["logSurv", "loglogSurv", "TailCurveResult"]
