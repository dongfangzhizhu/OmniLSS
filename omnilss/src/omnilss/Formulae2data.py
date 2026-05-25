"""R-aligned formula-to-data helper.

R reference: `gamlss/R/Formulae2data.R::Formulae2data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .fitting import _build_design_matrix_with_smooths


@dataclass(frozen=True)
class Formulae2DataResult:
    """Structured output from :func:`Formulae2data`."""

    response_name: str
    response: np.ndarray
    weights: np.ndarray
    design_matrix: np.ndarray
    column_labels: tuple[str, ...]
    has_smooth_terms: bool


def Formulae2data(formula: str, data: Mapping[str, Any], weights_key: str = "weights") -> Formulae2DataResult:
    """Build response and model matrix from a GAMLSS-style formula."""

    response_name, design, labels, smooth_info = _build_design_matrix_with_smooths(formula, data)
    y = np.asarray(data[response_name], dtype=np.float64)
    if weights_key in data:
        w = np.asarray(data[weights_key], dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError("weights must have same shape as response")
        if np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("weights must be finite and non-negative")
    else:
        w = np.ones_like(y, dtype=np.float64)

    has_smooth_terms = smooth_info is not None
    if has_smooth_terms:
        labels = tuple(getattr(smooth_info, "terms", ()))
    return Formulae2DataResult(
        response_name=response_name,
        response=y,
        weights=w,
        design_matrix=np.asarray(design, dtype=np.float64),
        column_labels=tuple(labels),
        has_smooth_terms=has_smooth_terms,
    )


__all__ = ["Formulae2data", "Formulae2DataResult"]
