"""R-aligned partial effect interfaces.

R source reference:
- file: `gamlss/R/getPEF.R`
- functions: `getPEF`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import GAMLSSModel


@dataclass(frozen=True)
class PartialEffectResult:
    """Structured staged `getPEF` partial-effect payload."""

    term: str
    what: str
    type: str
    x: np.ndarray
    effect: np.ndarray
    derivative: np.ndarray
    fixed_at: dict[str, float]


def get_pef_data(
    object: GAMLSSModel,
    term: str,
    what: str = "mu",
    type: str = "response",
    n_points: int = 100,
    how: str = "median",
    fixed_at: dict[str, float] | None = None,
) -> PartialEffectResult:
    """R reference: `gamlss/R/getPEF.R::getPEF`.

    Staged behavior:
    - Builds a one-term partial-effect curve over the observed term range.
    - Fixes all other variables at median or last observed values.
    - Returns the curve plus a numerical derivative instead of an R splinefun.
    """

    from .methods import _require_gamlss_method, predict

    _require_gamlss_method(object)
    selected = what.strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")
    requested_type = type.strip().lower()
    if requested_type not in {"response", "link"}:
        raise ValueError("type must be 'response' or 'link'")
    strategy = how.strip().lower()
    if strategy not in {"median", "last"}:
        raise ValueError("how must be 'median' or 'last'")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("get_pef_data requires call['data']")
    if term not in call_data:
        raise KeyError(f"term {term!r} not found in data")

    term_values = np.asarray(call_data[term], dtype=np.float64).ravel()
    if term_values.size == 0:
        raise ValueError("term data is empty")
    x_grid = np.linspace(float(np.min(term_values)), float(np.max(term_values)), int(n_points), dtype=np.float64)

    fixed_values = dict(fixed_at or {})
    newdata: dict[str, np.ndarray] = {}
    for name, values in call_data.items():
        array = np.asarray(values, dtype=np.float64).ravel()
        if name == term:
            newdata[name] = x_grid
            continue
        if name in fixed_values:
            fill_value = float(fixed_values[name])
        elif strategy == "last":
            fill_value = float(array[-1])
        else:
            fill_value = float(np.median(array))
        fixed_values[name] = fill_value
        newdata[name] = np.full(int(n_points), fill_value, dtype=np.float64)

    effect = np.asarray(
        predict(object, what=selected, type=requested_type, newdata=newdata),
        dtype=np.float64,
    ).ravel()
    derivative = np.gradient(effect, x_grid).astype(np.float64, copy=False)
    return PartialEffectResult(
        term=term,
        what=selected,
        type=requested_type,
        x=x_grid,
        effect=effect,
        derivative=derivative,
        fixed_at={key: float(value) for key, value in fixed_values.items() if key != term},
    )


getPEF = get_pef_data

__all__ = [
    "PartialEffectResult",
    "getPEF",
    "get_pef_data",
]
