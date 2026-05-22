"""R-aligned hazard helper utilities.

R reference: `gamlss.dist/R/hazardFun.R::{hazardFun, gen.hazard}`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HazardResult:
    x: np.ndarray
    hazard: np.ndarray


def hazardFun(pdf, cdf, x) -> HazardResult:
    """Compute hazard h(x) = f(x) / (1 - F(x)) from aligned arrays."""
    f = np.asarray(pdf, dtype=np.float64).ravel()
    F = np.asarray(cdf, dtype=np.float64).ravel()
    xv = np.asarray(x, dtype=np.float64).ravel()
    if f.shape != F.shape or f.shape != xv.shape:
        raise ValueError("pdf, cdf and x must have the same shape")
    eps = np.finfo(np.float64).eps
    surv = np.maximum(1.0 - F, eps)
    return HazardResult(x=xv, hazard=f / surv)


def gen_hazard(pdf, cdf, x) -> HazardResult:
    """Compatibility alias for generic hazard computation."""
    return hazardFun(pdf=pdf, cdf=cdf, x=x)


__all__ = ["hazardFun", "gen_hazard", "HazardResult"]
