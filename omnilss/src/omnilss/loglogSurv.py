"""R-aligned log-survival helpers with FitTail-compatible aliases.

R reference: `gamlss/R/FitTail.R::{loglogSurv, logSurv, logSurv0}`.
"""

from __future__ import annotations

from .FitTail import TailCurveResult, logSurv, loglogSurv


def logSurv0(y) -> TailCurveResult:
    """Compatibility alias returning `logSurv(y)` payload."""
    return logSurv(y)


__all__ = ["logSurv", "loglogSurv", "logSurv0", "TailCurveResult"]
