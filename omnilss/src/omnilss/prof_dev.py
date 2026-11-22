"""R-aligned profile deviance helper.

R reference: `gamlss/R/PROFILE_new_19-08-12.R::prof.dev`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class ProfDevResult:
    grid: np.ndarray
    deviance: np.ndarray
    weighted_deviance: np.ndarray
    minimum_index: int
    rank_order: np.ndarray


def prof_dev(grid, deviance, weights=None) -> ProfDevResult:
    """Build a structured profile-deviance payload from evaluated points."""
    g = np.asarray(grid, dtype=np.float64).ravel()
    d = np.asarray(deviance, dtype=np.float64).ravel()
    if g.size == 0 or d.size == 0:
        raise ValueError("grid and deviance must be non-empty")
    if g.shape != d.shape:
        raise ValueError("grid and deviance must have the same shape")
    if not np.isfinite(g).all() or not np.isfinite(d).all():
        raise ValueError("grid and deviance must be finite")

    if weights is None:
        w = np.ones_like(d, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != d.shape:
            raise ValueError("weights must have the same shape as deviance")
        if np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("weights must be finite and non-negative")

    wd = w * d
    order = np.argsort(wd)
    i = int(order[0])
    return ProfDevResult(
        grid=g,
        deviance=d,
        weighted_deviance=wd,
        minimum_index=i,
        rank_order=np.asarray(order, dtype=np.int64),
    )


def prof_term(
    object: GAMLSSModel, term_values, deviance, weights=None
) -> ProfDevResult:
    """R-style wrapper that validates model identity before returning profile data."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    return prof_dev(term_values, deviance, weights=weights)


__all__ = ["prof_dev", "prof_term", "ProfDevResult"]
