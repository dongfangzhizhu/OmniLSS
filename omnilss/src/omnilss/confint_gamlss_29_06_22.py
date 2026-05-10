"""R-aligned confidence interval interfaces.

R source reference:
- file: `gamlss/R/confint-gamlss-29-06-22.R`
- functions: `confint.gamlss`
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from .model import GAMLSSModel


def confint(
    object: GAMLSSModel,
    level: float = 0.95,
    what: str = "all",
) -> dict[str, np.ndarray] | np.ndarray:
    """R reference: `gamlss/R/confint-gamlss-29-06-22.R::confint.gamlss`."""

    from .methods import _require_gamlss_method
    from .operations import coef
    from .vcov_gamlss import vcov

    _require_gamlss_method(object)
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    cov = vcov(object, type="vcov")
    flat_coef = vcov(object, type="coef")
    se = np.sqrt(np.maximum(np.diag(cov), 0.0)) if cov.size else np.array([], dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    z_values = np.array(
        [NormalDist().inv_cdf(alpha), NormalDist().inv_cdf(1.0 - alpha)],
        dtype=np.float64,
    )
    full_ci = flat_coef[:, None] + se[:, None] * z_values[None, :]

    grouped: dict[str, np.ndarray] = {}
    cursor = 0
    for parameter in object.par:
        estimates = np.asarray(coef(object, parameter), dtype=np.float64).ravel()
        grouped[parameter] = full_ci[cursor : cursor + estimates.size]
        cursor += estimates.size

    if what == "all":
        return grouped
    if what not in grouped:
        raise ValueError(f"{what} is not a parameter in the object")
    return grouped[what]


__all__ = [
    "confint",
]
