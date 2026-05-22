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
    minimum_index: int


def prof_dev(grid, deviance) -> ProfDevResult:
    """Build a structured profile-deviance payload from evaluated points."""
    g = np.asarray(grid, dtype=np.float64).ravel()
    d = np.asarray(deviance, dtype=np.float64).ravel()
    if g.size == 0 or d.size == 0:
        raise ValueError("grid and deviance must be non-empty")
    if g.shape != d.shape:
        raise ValueError("grid and deviance must have the same shape")
    i = int(np.argmin(d))
    return ProfDevResult(grid=g, deviance=d, minimum_index=i)


def prof_term(object: GAMLSSModel, term_values, deviance) -> ProfDevResult:
    """R-style wrapper that validates model identity before returning profile data."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    return prof_dev(term_values, deviance)


__all__ = ["prof_dev", "prof_term", "ProfDevResult"]
