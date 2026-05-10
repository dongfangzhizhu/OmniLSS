"""R-aligned two-way effect plot interfaces.

R source reference:
- file: `gamlss/R/plot2way.R`
- functions: `plot2way`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import GAMLSSModel


@dataclass(frozen=True)
class Plot2WayResult:
    """Structured staged `plot2way` payload."""

    what: str
    terms: tuple[str, str]
    x_levels: np.ndarray
    y_levels: np.ndarray
    contribution: np.ndarray


def plot2way_data(
    object: GAMLSSModel,
    terms: tuple[str, str] | list[str],
    what: str = "mu",
) -> Plot2WayResult:
    """R reference: `gamlss/R/plot2way.R::plot2way`.

    Staged behavior:
    - Returns the two-way term contribution table instead of plotting.
    - Uses stored `lpred(..., type="terms")` values and observed data grouping.
    """

    from .methods import _require_gamlss_method
    from .operations import lpred

    _require_gamlss_method(object)
    selected = what.strip().lower()
    requested_terms = tuple(str(term).strip() for term in terms)
    if len(requested_terms) != 2 or not all(requested_terms):
        raise ValueError("terms must contain exactly two non-empty term names")

    lpred_result = lpred(object, what=selected, type="terms", terms=requested_terms, se_fit=False)
    term_matrix = np.asarray(lpred_result, dtype=np.float64)
    if term_matrix.ndim == 1:
        term_matrix = term_matrix[:, None]
    if term_matrix.shape[1] != 2:
        raise ValueError("plot2way_data requires exactly two term columns")

    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("plot2way_data requires call['data']")

    term1, term2 = requested_terms
    if term1 not in call_data or term2 not in call_data:
        raise KeyError("both requested terms must exist in call['data']")

    x_values = np.asarray(call_data[term1], dtype=np.float64).ravel()
    y_values = np.asarray(call_data[term2], dtype=np.float64).ravel()
    if x_values.shape != y_values.shape or x_values.shape[0] != term_matrix.shape[0]:
        raise ValueError("term data and term contributions must have compatible lengths")

    x_levels = np.unique(x_values)
    y_levels = np.unique(y_values)
    contribution = np.full((x_levels.size, y_levels.size), np.nan, dtype=np.float64)
    joint_term = np.sum(term_matrix, axis=1)

    for i, x_level in enumerate(x_levels):
        for j, y_level in enumerate(y_levels):
            mask = (x_values == x_level) & (y_values == y_level)
            if np.any(mask):
                contribution[i, j] = float(np.mean(joint_term[mask]))

    return Plot2WayResult(
        what=selected,
        terms=requested_terms,
        x_levels=x_levels,
        y_levels=y_levels,
        contribution=contribution,
    )


plot2way = plot2way_data

__all__ = [
    "Plot2WayResult",
    "plot2way",
    "plot2way_data",
]
