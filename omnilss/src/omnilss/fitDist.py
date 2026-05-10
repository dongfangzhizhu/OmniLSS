"""R-aligned distribution fitting interfaces.

R source reference:
- file: `gamlss/R/fitDist.R`
- functions: `fitDist`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import GAMLSSModel
from .operations import gaic


@dataclass(frozen=True)
class FitDistRow:
    """One staged `fitDist` ranking row."""

    family: str
    criterion: float
    df_fit: float
    global_deviance: float


@dataclass(frozen=True)
class FitDistResult:
    """Structured staged `fitDist` payload."""

    type: str
    k: float
    best_family: str
    best_model: GAMLSSModel
    rows: tuple[FitDistRow, ...]
    failed: tuple[str, ...]


def _fit_dist_available_families() -> dict[str, tuple[str, ...]]:
    available = {
        "realline": ("NO", "LO", "TF", "JSU"),
        "realplus": ("EXP", "GA", "IG", "LOGNO", "WEI", "BCCG", "BCT", "BCPE"),
        "real0to1": ("BE",),
        "counts": ("PO", "GEOM", "NBI", "ZIP"),
        "binom": ("BI",),
    }
    available["realall"] = tuple(dict.fromkeys(available["realline"] + available["realplus"]))
    return available


def fit_dist_data(
    y: Sequence[float] | np.ndarray,
    k: float = 2.0,
    type: str = "realAll",
    extra: Sequence[str] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> FitDistResult:
    """R reference: `gamlss/R/fitDist.R::fitDist`.

    Staged behavior:
    - Fits currently supported staged families for one selected family group.
    - Ranks successful fits by GAIC/AIC-style criterion.
    - Returns the best model, ranking rows, and failed family names.
    """

    from .fitting import gamlss_ml

    available = _fit_dist_available_families()
    selected_type = str(type).strip()
    key = selected_type.lower()
    if key not in available:
        raise ValueError("type must be one of 'realAll', 'realline', 'realplus', 'real0to1', 'counts', or 'binom'")

    family_names = list(available[key])
    if extra is not None:
        family_names.extend(str(name).strip().upper() for name in extra if str(name).strip())
    family_names = list(dict.fromkeys(family_names))

    y_array = np.asarray(y, dtype=np.float64).ravel()
    if y_array.size == 0:
        raise ValueError("y must contain at least one observation")
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w is not None and w.size != y_array.size:
        raise ValueError("weights must have the same length as y")

    rows: list[FitDistRow] = []
    failed: list[str] = []
    best_model: GAMLSSModel | None = None
    best_criterion = float("inf")

    for family_name in family_names:
        try:
            fit = gamlss_ml(
                formula="y ~ 1",
                family=family_name,
                data={"y": y_array},
                weights=w,
            )
        except Exception:
            failed.append(family_name)
            continue
        criterion = float(gaic(fit, k=float(k)))
        row = FitDistRow(
            family=str(getattr(fit.family, "name", family_name)),
            criterion=criterion,
            df_fit=float(fit.df_fit),
            global_deviance=float(fit.g_dev),
        )
        rows.append(row)
        if criterion < best_criterion:
            best_criterion = criterion
            best_model = fit

    if best_model is None:
        raise ValueError("no candidate family could be fitted")

    ordered_rows = tuple(sorted(rows, key=lambda row: (row.criterion, row.family)))
    return FitDistResult(
        type=selected_type,
        k=float(k),
        best_family=str(getattr(best_model.family, "name", best_model.family)),
        best_model=best_model,
        rows=ordered_rows,
        failed=tuple(failed),
    )


def fit_dist(
    y: Sequence[float] | np.ndarray,
    k: float = 2.0,
    type: str = "realAll",
    extra: Sequence[str] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> FitDistResult:
    """R reference: `gamlss/R/fitDist.R::fitDist`."""

    return fit_dist_data(y=y, k=k, type=type, extra=extra, weights=weights)

fitDist = fit_dist

__all__ = [
    "FitDistResult",
    "FitDistRow",
    "fitDist",
    "fit_dist",
    "fit_dist_data",
]
