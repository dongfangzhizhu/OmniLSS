"""Smoother plotting data helpers (`plotDF`, `plotLambda`).

R reference: `gamlss` smoother diagnostics (`plotDF`, `plotLambda`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .getSmo import getSmo
from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class SmootherPlotRow:
    parameter: str
    variable: str
    smoother: str
    value: float


def plotDF(object: GAMLSSModel, parameter: str = "mu") -> tuple[SmootherPlotRow, ...]:
    """Return per-smoother effective degrees-of-freedom rows for diagnostics."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    rows: list[SmootherPlotRow] = []
    for info in getSmo(object, parameter=parameter):
        if info.edf is None:
            continue
        rows.append(
            SmootherPlotRow(
                parameter=info.parameter,
                variable=info.variable,
                smoother=info.smoother,
                value=float(info.edf),
            )
        )
    return tuple(rows)


def plotLambda(
    object: GAMLSSModel, parameter: str = "mu"
) -> tuple[SmootherPlotRow, ...]:
    """Return per-smoother lambda rows for diagnostics."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    rows: list[SmootherPlotRow] = []
    for info in getSmo(object, parameter=parameter):
        if info.lambda_ is None:
            continue
        rows.append(
            SmootherPlotRow(
                parameter=info.parameter,
                variable=info.variable,
                smoother=info.smoother,
                value=float(info.lambda_),
            )
        )
    return tuple(rows)


__all__ = ["SmootherPlotRow", "plotDF", "plotLambda"]
