"""R-aligned histogram smoothing payload helpers.

R reference: `gamlss/R/histSmo.R::{histSmo, histSmoO, histSmoC, histSmoP}`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HistSmoResult:
    bin_edges: np.ndarray
    counts: np.ndarray
    density: np.ndarray


def _hist_payload(y, bins: int = 10) -> HistSmoResult:
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("y must contain at least one value")
    counts, edges = np.histogram(arr, bins=bins)
    widths = np.diff(edges)
    total = max(arr.size, 1)
    density = counts / (total * widths)
    return HistSmoResult(
        bin_edges=edges, counts=counts.astype(np.float64), density=density
    )


def histSmo(y, bins: int = 10) -> HistSmoResult:
    return _hist_payload(y, bins=bins)


def histSmoO(y, bins: int = 10) -> HistSmoResult:
    return _hist_payload(y, bins=bins)


def histSmoC(y, bins: int = 10) -> HistSmoResult:
    return _hist_payload(y, bins=bins)


def histSmoP(y, bins: int = 10) -> HistSmoResult:
    return _hist_payload(y, bins=bins)


__all__ = ["histSmo", "histSmoO", "histSmoC", "histSmoP", "HistSmoResult"]
