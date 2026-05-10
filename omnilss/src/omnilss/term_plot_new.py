"""R-aligned term plot interfaces.

R source reference:
- file: `gamlss/R/term.plot-new.R`
- functions: `term.plot`
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .model import GAMLSSModel


@dataclass(frozen=True)
class TermPlotEntry:
    """One staged term-plot data bundle for a single term."""

    term: str
    x: np.ndarray
    fit: np.ndarray
    se_fit: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None


@dataclass(frozen=True)
class TermPlotResult:
    """Structured staged `term.plot` data payload."""

    what: str
    entries: tuple[TermPlotEntry, ...]


def term_plot_data(
    object: GAMLSSModel,
    what: str = "mu",
    terms: list[str] | tuple[str, ...] | None = None,
    se: bool = True,
    level: float = 0.95,
) -> TermPlotResult:
    """R reference: `gamlss/R/term.plot-new.R::term.plot`.

    Staged behavior:
    - Returns linear term plot data instead of drawing.
    - Uses stored `lpred(..., type="terms")` matrices and call data columns.
    """

    from .methods import _require_gamlss_method
    from .operations import lpred

    _require_gamlss_method(object)
    selected = what.strip().lower()
    if selected not in object.par:
        raise ValueError(f"{selected} is not a parameter in the object")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    term_info = terms if terms is not None else list(object.terms.get(selected, {}).get("term_labels", []))
    requested_terms = [str(term).strip() for term in term_info if str(term).strip()]
    if not requested_terms:
        raise ValueError("no terms available for term plot data")

    lpred_result = lpred(object, what=selected, type="terms", terms=requested_terms, se_fit=se)
    if isinstance(lpred_result, dict):
        fit_matrix = np.asarray(lpred_result["fit"], dtype=np.float64)
        se_matrix = np.asarray(lpred_result["se.fit"], dtype=np.float64)
    else:
        fit_matrix = np.asarray(lpred_result, dtype=np.float64)
        se_matrix = None
    if fit_matrix.ndim == 1:
        fit_matrix = fit_matrix[:, None]
    if se_matrix is not None and se_matrix.ndim == 1:
        se_matrix = se_matrix[:, None]

    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("term plot data requires call['data']")

    alpha = (1.0 - level) / 2.0
    z_value = NormalDist().inv_cdf(1.0 - alpha)
    entries: list[TermPlotEntry] = []
    for index, term_name in enumerate(requested_terms):
        if term_name not in call_data:
            raise KeyError(f"term {term_name!r} not found in data")
        x_values = np.asarray(call_data[term_name], dtype=np.float64).ravel()
        fit_values = fit_matrix[:, index].astype(np.float64, copy=False)
        order = np.argsort(x_values)
        ordered_x = x_values[order]
        ordered_fit = fit_values[order]
        if se_matrix is None:
            ordered_se = None
            lower = None
            upper = None
        else:
            ordered_se = se_matrix[:, index].astype(np.float64, copy=False)[order]
            lower = ordered_fit - z_value * ordered_se
            upper = ordered_fit + z_value * ordered_se
        entries.append(
            TermPlotEntry(
                term=term_name,
                x=ordered_x,
                fit=ordered_fit,
                se_fit=ordered_se,
                lower=lower,
                upper=upper,
            )
        )

    return TermPlotResult(what=selected, entries=tuple(entries))


term_plot = term_plot_data

__all__ = [
    "TermPlotEntry",
    "TermPlotResult",
    "term_plot",
    "term_plot_data",
]
