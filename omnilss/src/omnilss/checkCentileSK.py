"""R-aligned centileSK consistency checker.

R reference: `gamlss.dist/R/momentSK.R::checkCentileSK`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .centileSK import centileSK


@dataclass(frozen=True)
class CheckCentileSKResult:
    max_abs_error: float
    mean_abs_error: float


def checkCentileSK(y, centiles, target_values) -> CheckCentileSKResult:
    """Compare empirical centiles against target centile values."""
    got = centileSK(y, centiles=centiles).values
    tgt = np.asarray(target_values, dtype=np.float64).ravel()
    if got.shape != tgt.shape:
        raise ValueError("target_values shape mismatch")
    err = np.abs(got - tgt)
    return CheckCentileSKResult(
        max_abs_error=float(np.max(err)), mean_abs_error=float(np.mean(err))
    )


__all__ = ["checkCentileSK", "CheckCentileSKResult"]
