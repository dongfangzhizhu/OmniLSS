"""R-aligned momentSK consistency checker.

R reference: `gamlss.dist/R/momentSK.R::checkMomentSK`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .momentSK import momentSK


@dataclass(frozen=True)
class CheckMomentSKResult:
    mean_abs_error: float
    variance_abs_error: float
    skewness_abs_error: float
    kurtosis_abs_error: float


def checkMomentSK(
    y, target_mean, target_variance, target_skewness, target_kurtosis
) -> CheckMomentSKResult:
    """Compare empirical moments against target moments using absolute errors."""
    m = momentSK(y)
    return CheckMomentSKResult(
        mean_abs_error=float(abs(m.mean - float(target_mean))),
        variance_abs_error=float(abs(m.variance - float(target_variance))),
        skewness_abs_error=float(abs(m.skewness - float(target_skewness))),
        kurtosis_abs_error=float(abs(m.kurtosis - float(target_kurtosis))),
    )


__all__ = ["checkMomentSK", "CheckMomentSKResult"]
