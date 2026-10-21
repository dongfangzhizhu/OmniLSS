"""Family response-domain (support) validation — R parity for ``family$y.valid``.

R source references:
- file: ``gamlss/R/gamlss-5.R`` (function ``gamlss``, fitting entry point).
  The check runs **before** any IRLS step::

      ##checking the permissible y values
      if (!family$y.valid(y))
          stop( "response variable out of range")

- file: ``gamlss.dist/R/<FAMILY>.R`` — each family object carries its own
  ``y.valid`` closure.  Across the 119 distribution files the predicate
  collapses into a handful of shapes.

Why this is a hard error
------------------------
R uses ``stop()``, not ``warning()``: an out-of-support response is a caller
error, not a recoverable fit condition.  Continuing silently yields a "fitted"
model whose deviance is meaningless — exactly the class of silent failure this
project has been removing (see D2.2 convergence reporting).  The R message is
reproduced verbatim so downstream log matching keeps working.

This is complementary to :mod:`omnilss.family_validation`, which checks the
*estimated parameters* (``sigma > 0`` and friends).  This module checks the
*observed response*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Predicate applied to the response array.  Returns a boolean array.
ResponseDomain = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ResponseDomainError(ValueError):
    """Raised when the response falls outside a family's support.

    Attributes
    ----------
    family : str
        Family name whose domain was violated.
    offending : int
        Number of observations that failed the check.
    sample : tuple[float, ...]
        Up to five offending values, quoted in the message.
    """

    family: str
    offending: int
    sample: tuple[float, ...]

    def __str__(self) -> str:
        return (
            f"response variable out of range: {self.offending} observation(s) "
            f"outside the support of family {self.family!r}; "
            f"offending values (first 5): {list(self.sample)}"
        )


# ---------------------------------------------------------------------------
# Domain predicates.  Each mirrors one R ``y.valid`` body verbatim.
# ---------------------------------------------------------------------------

def _all_positive(y: np.ndarray) -> np.ndarray:
    """R: ``all(y > 0)``"""
    return y > 0.0


def _all_non_negative(y: np.ndarray) -> np.ndarray:
    """R: ``all(y >= 0)``"""
    return y >= 0.0


def _all_finite(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y)


def _in_unit_excl_zero(y: np.ndarray) -> np.ndarray:
    """R: ``all(y >= 0 & y < 1)``"""
    return (y >= 0.0) & (y < 1.0)


def _in_closed_unit(y: np.ndarray) -> np.ndarray:
    """R: ``all(y >= 0 & y <= 1)``"""
    return (y >= 0.0) & (y <= 1.0)


def _in_unit_excl_one(y: np.ndarray) -> np.ndarray:
    """R: ``all(y > 0 & y <= 1)``"""
    return (y > 0.0) & (y <= 1.0)


def _above_one(y: np.ndarray) -> np.ndarray:
    """R: ``all(y > 1)``"""
    return y > 1.0


def _at_least_one(y: np.ndarray) -> np.ndarray:
    """R: ``all(y >= 1)``"""
    return y >= 1.0


def _in_closed_1_4(y: np.ndarray) -> np.ndarray:
    """R: ``all(y >= 1 & y <= 4)``"""
    return (y >= 1.0) & (y <= 4.0)


def _in_closed_1_5(y: np.ndarray) -> np.ndarray:
    """R: ``all(y >= 1 & y <= 5)``"""
    return (y >= 1.0) & (y <= 5.0)


# Family name -> response domain, keyed by family name to match R's dispatch
# on the family object.
#
# Provenance rule for this table: an entry may only exist if the R family source
# actually declares ``y.valid``.  Families with no ``y.valid`` in R (PO, ZINB,
# ALPHA, BETA, DAGUM, LOGGAMMA, LOMAX, NOF, FGA, GGA, MIXED, BCPEO*, ...) fall
# through to the permissive default.  Inventing a restriction R does not
# enforce would reject data R happily fits, and a false rejection is worse than
# a missing check.
#
# Each entry below was read out of ``gamlss.dist/R/<FAMILY>.R`` verbatim.
_RESPONSE_DOMAINS: dict[str, ResponseDomain] = {
    # --- all(y > 0) ---
    "GA": _all_positive,
    "EXP": _all_positive,
    # GIG.R:124 -- y.valid = function(y) all(y>0).  Added with the D2.48
    # delivery: live R's gamlss() rejects a negative GIG response with
    # "response variable out of range", so the check belongs here.
    "GIG": _all_positive,
    "PARETO1": _all_positive,
    "WEI": _all_positive,
    "WEI2": _all_positive,
    "WEI3": _all_positive,
    # --- all(y >= 0) ---
    "ZIP": _all_non_negative,
    "ZABB": _all_non_negative,
    "ZIBB": _all_non_negative,
    # --- unit-interval supports ---
    "BEINF": _in_closed_unit,      # all(y >= 0 & y <= 1)
    "BEINF0": _in_unit_excl_zero,  # all(y >= 0 & y < 1)
    "BEINF1": _in_unit_excl_one,   # all(y > 0 & y <= 1)
    "BEOI": _in_unit_excl_one,     # all(y > 0 & y <= 1)
    "BEZI": _in_unit_excl_zero,    # all(y >= 0 & y < 1)
    # --- shifted / truncated supports ---
    "MN4": _in_closed_1_4,         # all(y >= 1 & y <= 4)
    "MN5": _in_closed_1_5,         # all(y >= 1 & y <= 5)
    "PARETO": _above_one,          # all(y > 1)
    "ZIPF": _at_least_one,         # all(y >= 1)
}

# Families whose R ``y.valid`` is a three-level factor check
# (``if (is.factor(y)) nlevels(y) == 3``).  A numeric array is taken as
# already-coded 0/1/2, which is what a coded factor arrives as; the level
# count is enforced by the family itself, not here.
_FACTOR_LEVEL_FAMILIES = frozenset({"MN3"})


def response_domain_for(family_name: str) -> ResponseDomain:
    """Return the R-equivalent ``y.valid`` predicate for ``family_name``.

    Unknown families get :func:`_all_finite`, standing in for R's very common
    ``y.valid = function(y) TRUE`` while still rejecting NaN/Inf, which no
    family can legitimately fit.
    """
    if family_name in _FACTOR_LEVEL_FAMILIES:
        return _all_finite
    return _RESPONSE_DOMAINS.get(family_name, _all_finite)


def validate_response(
    family_name: str,
    y: np.ndarray,
    *,
    raise_on_violation: bool = True,
) -> int:
    """Check ``y`` against the family's response domain.

    Parameters
    ----------
    family_name : str
        Family the response is being fitted with.
    y : np.ndarray
        Observed response; any shape is accepted.
    raise_on_violation : bool
        ``True`` (default) raises :class:`ResponseDomainError`, mirroring R's
        ``stop()``.  ``False`` returns the offending count instead, for
        callers that collect issues rather than abort.

    Returns
    -------
    int
        Number of observations outside the support; always ``0`` on success.

    Raises
    ------
    ResponseDomainError
        If any value is out of support and ``raise_on_violation`` is ``True``.
    """
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0
    ok = np.asarray(response_domain_for(family_name)(arr), dtype=bool)
    # Every predicate above yields False for NaN; the explicit check also
    # covers the permissive default so both paths agree.
    ok &= np.isfinite(arr)
    bad = arr[~ok]
    if bad.size == 0:
        return 0
    if raise_on_violation:
        raise ResponseDomainError(
            family=family_name,
            offending=int(bad.size),
            sample=tuple(float(v) for v in bad[:5]),
        )
    return int(bad.size)


def response_violation_count(family_name: str, y: np.ndarray) -> int:
    """Non-raising variant of :func:`validate_response`."""
    return validate_response(family_name, y, raise_on_violation=False)


__all__ = [
    "ResponseDomainError",
    "response_domain_for",
    "response_violation_count",
    "validate_response",
]
