"""R gamlss edf(): effective degrees of freedom of the smooth terms.

R source: R/gamlss/R/edf.R.

R reads obj$<what>.coefSmo (one fitted smoother per smooth term) and returns a
named numeric vector of their edf slot.  This engine keeps the fitted smoothers
under additional_slots["smooth_infos"][parameter].smooth_fits, and the final
lambda of each term is updated in place by the RS cycles, so the edf has to be
recomputed from the final design matrix, weights and penalty (the edf recorded
when the design was first built is the seed value, not the fitted one).

Like R, edf() returns the scalar 0 when the parameter has no smooth terms.

Deviation: R names the entries after the columns of the smooth design matrix;
this engine names them after the smooth term variable, because the basis columns
are not individually named here.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

_PARAMETERS = ("mu", "sigma", "nu", "tau")


def smooth_fits_for_parameter(obj: Any, parameter: str) -> list[Any]:
    """Return the fitted smoother records for a parameter (possibly empty)."""

    slots = getattr(obj, "additional_slots", {}) or {}
    infos = slots.get("smooth_infos") or {}
    design = infos.get(parameter)
    if design is not None:
        return list(getattr(design, "smooth_fits", None) or [])
    per_parameter = slots.get("smooth_fits") or {}
    return list(per_parameter.get(parameter) or [])


def smooth_term_edf(obj: Any, parameter: str = "mu") -> dict[str, float]:
    """Effective df per smooth term, recomputed from the final lambda.

    Returns an empty mapping when the parameter has no smooth terms.
    """

    fits = smooth_fits_for_parameter(obj, parameter)
    if not fits:
        return {}

    design_matrices = getattr(obj, "design_matrices", None) or {}
    design = design_matrices.get(parameter)
    weights = getattr(obj, "weights", None)
    if design is None or weights is None:
        # No design available (e.g. a hand-built model): fall back to the edf
        # recorded by the smoother itself.
        return {
            str(getattr(entry, "variable", index)): float(getattr(entry, "edf", 0.0))
            for index, entry in enumerate(fits)
        }

    from .smoothers.penalties import effective_df

    matrix = np.asarray(design)
    weight_vector = np.asarray(weights)
    result: dict[str, float] = {}
    for index, entry in enumerate(fits):
        basis_columns = getattr(entry, "basis_columns", None)
        if basis_columns is None:
            result[str(getattr(entry, "variable", index))] = float(getattr(entry, "edf", 0.0))
            continue
        start, end = basis_columns
        value = effective_df(
            jnp.asarray(matrix[:, start:end]),
            jnp.asarray(entry.penalty),
            entry.lambda_,
            jnp.asarray(weight_vector),
        )
        result[str(getattr(entry, "variable", index))] = float(value)
    return result


def edf(
    obj: Any,
    what: str = "mu",
    parameter: str | None = None,
    print: bool = True,
    **_kwargs: Any,
):
    """R edf: effective degrees of freedom of the smooth terms of a parameter.

    Parameters
    ----------
    obj : GAMLSSModel
        A fitted model.
    what : str, default "mu"
        Which distribution parameter to report (mu, sigma, nu, tau).
    parameter : str, optional
        R compatibility alias; when given it overrides what, as in R.
    print : bool, default True
        Print the R banner line.
    """

    selected = parameter if parameter is not None else what
    if selected not in _PARAMETERS:
        raise ValueError(f"'arg' should be one of {list(_PARAMETERS)}, got {selected!r}")

    if print:
        print(f"Effective df for {selected} model")

    per_term = smooth_term_edf(obj, selected)
    if not per_term:
        return 0
    return per_term


__all__ = ["edf", "smooth_fits_for_parameter", "smooth_term_edf"]
