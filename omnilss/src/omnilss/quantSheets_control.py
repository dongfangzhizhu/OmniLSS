"""R-aligned quantSheets control helper.

R reference: `gamlss/R/quantSheets.R::quantSheets.control`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class QuantSheetsControl:
    n_intervals: int = 10
    degree: int = 3
    lambda_: float = 1.0
    max_iter: int = 50
    tol: float = 1e-6


def quantSheets_control(
    n_intervals: int = 10,
    degree: int = 3,
    lambda_: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float | int]:
    """Build control dictionary for quant-sheet fitting workflows."""
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    control = QuantSheetsControl(
        n_intervals=int(n_intervals),
        degree=int(degree),
        lambda_=float(lambda_),
        max_iter=int(max_iter),
        tol=float(tol),
    )
    return asdict(control)


__all__ = ["QuantSheetsControl", "quantSheets_control"]
