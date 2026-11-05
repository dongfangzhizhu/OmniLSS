"""R gamlss.dist family means for the inflated beta models.

R source (live gamlss.dist 6.1.1, not present in the vendored tree):
    meanBEOI <- function(obj) {
      if (obj$family[1] != "BEOI") stop("the object do not have a BEOI distribution")
      (fitted(obj, "nu")) + ((1 - fitted(obj, "nu")) * fitted(obj, "mu"))
    }
    meanBEZI <- function(obj) {
      if (obj$family[1] != "BEZI") stop("the object do not have a BEZI distribution")
      (1 - fitted(obj, "nu")) * fitted(obj, "mu")
    }

The fitted values come from the model, so the Python port reads
GAMLSSModel.fitted_values and returns a JAX array of length n.
"""
from __future__ import annotations

from typing import Any

from .families import FamilyDefinition


def _family_name(obj: Any) -> str:
    family = getattr(obj, "family", None)
    if isinstance(family, FamilyDefinition):
        return str(family.name)
    if isinstance(family, str):
        return family
    return str(getattr(obj, "family_name", ""))


def _fitted(obj: Any, parameter: str) -> Any:
    values = getattr(obj, "fitted_values", None)
    if values is None or parameter not in values:
        raise ValueError(f"the model does not have fitted values for {parameter!r}")
    return values[parameter]


def meanBEZI(obj: Any) -> Any:
    """R meanBEZI: mean of a zero-inflated beta fit, (1 - nu) * mu."""

    if _family_name(obj) != "BEZI":
        raise ValueError("the object do not have a BEZI distribution")
    return (1.0 - _fitted(obj, "nu")) * _fitted(obj, "mu")


def meanBEOI(obj: Any) -> Any:
    """R meanBEOI: mean of a one-inflated beta fit, nu + (1 - nu) * mu."""

    if _family_name(obj) != "BEOI":
        raise ValueError("the object do not have a BEOI distribution")
    nu = _fitted(obj, "nu")
    return nu + (1.0 - nu) * _fitted(obj, "mu")


__all__ = ["meanBEOI", "meanBEZI"]
