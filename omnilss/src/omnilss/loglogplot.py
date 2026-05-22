"""R-aligned log-log survival transform helper.

R reference: `gamlss/R/loglogplot.R::loglogplot`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .ECDF import ECDF


@dataclass(frozen=True)
class LogLogPlotResult:
    x: np.ndarray
    log_x: np.ndarray
    log_log_survival: np.ndarray


def loglogplot(y) -> LogLogPlotResult:
    """Compute log(x) vs log(-log(S(x))) diagnostic coordinates."""
    ecdf = ECDF(y)
    # clip probabilities to avoid log(0)
    eps = np.finfo(np.float64).eps
    surv = np.clip(1.0 - ecdf.p, eps, 1.0 - eps)
    x = np.clip(ecdf.x, eps, None)
    return LogLogPlotResult(
        x=ecdf.x,
        log_x=np.log(x),
        log_log_survival=np.log(-np.log(surv)),
    )


__all__ = ["loglogplot", "LogLogPlotResult"]
