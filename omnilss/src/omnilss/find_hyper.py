"""R-aligned hyper-parameter search summary helper.

R reference: `gamlss/R/findhyper-23-05-13.R::find.hyper`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FindHyperResult:
    values: np.ndarray
    scores: np.ndarray
    best_value: float
    best_score: float
    best_index: int
    direction: str
    rank_order: np.ndarray


def find_hyper(values: Iterable[float], scores: Iterable[float], direction: str = "min") -> FindHyperResult:
    """Return best hyper-parameter candidate from scored grid search results."""
    vals = np.asarray(list(values), dtype=np.float64)
    scr = np.asarray(list(scores), dtype=np.float64)
    if vals.size == 0 or scr.size == 0:
        raise ValueError("values and scores must be non-empty")
    if vals.shape != scr.shape:
        raise ValueError("values and scores must have the same shape")
    if direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")
    if not np.isfinite(vals).all() or not np.isfinite(scr).all():
        raise ValueError("values and scores must be finite")

    order = np.argsort(scr)
    if direction == "max":
        order = order[::-1]
    idx = int(order[0])
    return FindHyperResult(
        values=vals,
        scores=scr,
        best_value=float(vals[idx]),
        best_score=float(scr[idx]),
        best_index=idx,
        direction=direction,
        rank_order=np.asarray(order, dtype=np.int64),
    )


__all__ = ["find_hyper", "FindHyperResult"]
