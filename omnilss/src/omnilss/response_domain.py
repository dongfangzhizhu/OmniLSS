"""Family response-domain (support) validation — R parity for `family$y.valid`.

R source references:
- file: `gamlss/R/gamlss-5.R` (function `gamlss`, fitting entry point).
  The check runs **before** any IRLS step::

      ##checking the permissible y values
      if (!family$y.valid(y))
          stop( "response variable out of range")

- file: `gamlss.dist/R/<FAMILY>.R` — each family object carries its own
  `y.valid` closure.  Across the 120 exported families the predicate collapses
  into nine shapes, all captured by the table below.

Why this is a hard error
------------------------
R uses `stop()`, not `warning()`: an out-of-support response is a caller error,
not a recoverable fit condition.  Continuing silently yields a "fitted" model
whose deviance is meaningless — exactly the class of silent failure this
project has been removing (see D2.2 convergence reporting).  The R message is
reproduced verbatim so downstream log matching keeps working.

Provenance (D2.129)
-------------------
Every entry below is pinned by
`omnilss/tests/fixtures/r_golden/response_domains_y_valid.json`, generated from
**live R** by `tools/generate_response_domain_golden.py` (it runs
`tools/r_y_valid_dump.R`, which deparses `body(<FAMILY>()$y.valid)` for every
export of the installed `gamlss.dist`).  The artifact records both the R
predicate text and the mapping from our dispatch name to R's name, so
`test_response_domain.py` can prove the table still matches live R without
needing R at test time.

An entry may only exist if R actually restricts the support.  Families whose
`y.valid` is `function(y) TRUE` (NO, SHASH, ST*, JSU*, PARETO2*, ...) fall
through to the permissive default; inventing a restriction R does not enforce
would reject data R happily fits, and a false rejection is worse than a missing
check.  Conversely, a restriction R *does* enforce but we skip is a silent
acceptance of data R refuses — that was the D2.128 finding (24 of 120
predicates covered; PO / SI / NBI / GG / BCT ... all missing).

This is complementary to :mod:`omnilss.family_validation`, which checks the
*estimated parameters* (`sigma > 0` and friends).  This module checks the
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
# Domain predicates.  Each mirrors one R `y.valid` body verbatim.
# ---------------------------------------------------------------------------


def _all_positive(y: np.ndarray) -> np.ndarray:
    """R: `all(y > 0)`"""
    return y > 0.0


def _all_non_negative(y: np.ndarray) -> np.ndarray:
    """R: `all(y >= 0)`"""
    return y >= 0.0


def _all_finite(y: np.ndarray) -> np.ndarray:
    """R: `function(y) TRUE` — no support restriction, only finiteness."""
    return np.isfinite(y)


def _in_unit_excl_zero(y: np.ndarray) -> np.ndarray:
    """R: `all(y >= 0 & y < 1)`"""
    return (y >= 0.0) & (y < 1.0)


def _in_closed_unit(y: np.ndarray) -> np.ndarray:
    """R: `all(y >= 0 & y <= 1)`"""
    return (y >= 0.0) & (y <= 1.0)


def _in_unit_excl_one(y: np.ndarray) -> np.ndarray:
    """R: `all(y > 0 & y <= 1)`"""
    return (y > 0.0) & (y <= 1.0)


def _in_open_unit(y: np.ndarray) -> np.ndarray:
    """R: `all(y > 0 & y < 1)`"""
    return (y > 0.0) & (y < 1.0)


def _above_one(y: np.ndarray) -> np.ndarray:
    """R: `all(y > 1)`"""
    return y > 1.0


def _at_least_one(y: np.ndarray) -> np.ndarray:
    """R: `all(y >= 1)`"""
    return y >= 1.0


def _in_closed_1_4(y: np.ndarray) -> np.ndarray:
    """R: `all(y >= 1 & y <= 4)`"""
    return (y >= 1.0) & (y <= 4.0)


def _in_closed_1_5(y: np.ndarray) -> np.ndarray:
    """R: `all(y >= 1 & y <= 5)`"""
    return (y >= 1.0) & (y <= 5.0)


# ---------------------------------------------------------------------------
# Family name -> response domain, keyed by the name `gamlss()` passes in
# (`family.name`, which is what the R family object calls itself).
#
# R names are shown per group; the three families whose Python dispatch name
# differs from R's only in case (BCCGUNTR / BCPEUNTR / BCTUNTR are R's
# BCCGuntr / BCPEuntr / BCTuntr) are noted inline.  Four families reach an R
# kernel under a different name entirely (IGA->IG, DPO1->DPO,
# GENERALISEDPOISSON->GPO, DELAPORT->DEL) and are keyed by the name we dispatch
# on.
# ---------------------------------------------------------------------------
_RESPONSE_DOMAINS: dict[str, ResponseDomain] = {
    # --- all(y > 0) ---
    "BCCG": _all_positive,
    "BCCGo": _all_positive,
    "BCCGUNTR": _all_positive,  # R: BCCGuntr
    "BCPE": _all_positive,
    "BCPEo": _all_positive,
    "BCPEUNTR": _all_positive,  # R: BCPEuntr
    "BCT": _all_positive,
    "BCTo": _all_positive,
    "BCTUNTR": _all_positive,  # R: BCTuntr
    "EXP": _all_positive,
    "GA": _all_positive,
    "GAF": _all_positive,
    "GG": _all_positive,
    "GIG": _all_positive,
    "GP": _all_positive,
    "IG": _all_positive,
    "IGA": _all_positive,  # R kernel: IG
    "LG": _all_positive,
    "LNO": _all_positive,
    "LOGNO": _all_positive,
    "LOGNO2": _all_positive,
    "PARETO1": _all_positive,
    "WEI": _all_positive,
    "WEI2": _all_positive,
    "WEI3": _all_positive,
    # --- all(y >= 0) ---
    "BB": _all_non_negative,
    "BI": _all_non_negative,
    "BNB": _all_non_negative,
    "DBI": _all_non_negative,
    "DBURR12": _all_non_negative,
    "DEL": _all_non_negative,
    "DELAPORT": _all_non_negative,  # R kernel: DEL
    "DPO": _all_non_negative,
    "DPO1": _all_non_negative,  # R kernel: DPO
    "GENERALISEDPOISSON": _all_non_negative,  # R kernel: GPO
    "GEOM": _all_non_negative,
    "GEOMo": _all_non_negative,
    "GPO": _all_non_negative,
    "NBF": _all_non_negative,
    "NBI": _all_non_negative,
    "NBII": _all_non_negative,
    "PIG": _all_non_negative,
    "PIG2": _all_non_negative,
    "PO": _all_non_negative,
    "SI": _all_non_negative,
    "SICHEL": _all_non_negative,
    "WARING": _all_non_negative,
    "YULE": _all_non_negative,
    "ZABB": _all_non_negative,
    "ZABI": _all_non_negative,
    "ZABNB": _all_non_negative,
    "ZAGA": _all_non_negative,
    "ZAIG": _all_non_negative,
    "ZALG": _all_non_negative,
    "ZANBI": _all_non_negative,
    "ZAP": _all_non_negative,
    "ZAPIG": _all_non_negative,
    "ZASICHEL": _all_non_negative,
    "ZAZIPF": _all_non_negative,
    "ZIBB": _all_non_negative,
    "ZIBI": _all_non_negative,
    "ZIBNB": _all_non_negative,
    "ZINBF": _all_non_negative,
    "ZINBI": _all_non_negative,
    "ZIP": _all_non_negative,
    "ZIP2": _all_non_negative,
    "ZIPIG": _all_non_negative,
    "ZISICHEL": _all_non_negative,
    # --- all(y > 0 & y < 1) ---
    "BE": _in_open_unit,
    "BEo": _in_open_unit,
    "GB1": _in_open_unit,
    "LOGITNO": _in_open_unit,
    "SIMPLEX": _in_open_unit,
    # --- unit-interval supports (endpoints differ per family) ---
    "BEINF": _in_closed_unit,  # all(y >= 0 & y <= 1)
    "BEINF0": _in_unit_excl_zero,  # all(y >= 0 & y < 1)
    "BEINF1": _in_unit_excl_one,  # all(y > 0 & y <= 1)
    "BEOI": _in_unit_excl_one,  # all(y > 0 & y <= 1)  -- one-inflated
    "BEZI": _in_unit_excl_zero,  # all(y >= 0 & y < 1)  -- zero-inflated
    # --- shifted / truncated supports ---
    "MN4": _in_closed_1_4,  # all(y >= 1 & y <= 4)
    "MN5": _in_closed_1_5,  # all(y >= 1 & y <= 5)
    "PARETO": _above_one,  # all(y > 1)
    "ZIPF": _at_least_one,  # all(y >= 1)
}

#: Case-insensitive index over the table.  `gamlss()` passes `family.name`,
#: and for several families that is not the registry key (`resolve("BCCGUNTR")`
#: is named `"BCCGUNTR"` while R calls it `BCCGuntr`).  A key that silently
#: misses here is invisible — the fit simply proceeds without the guard — so
#: lookup degrades to a case-insensitive match instead of falling straight
#: through to the permissive default.
_LOWER_RESPONSE_DOMAINS: dict[str, ResponseDomain] = {}
for _name, _domain in _RESPONSE_DOMAINS.items():
    _LOWER_RESPONSE_DOMAINS.setdefault(_name.lower(), _domain)

# Families whose R `y.valid` is a factor-level check
# (`if (is.factor(y)) nlevels(y) == 3`).  A numeric array is taken as
# already-coded levels, which is what a coded factor arrives as; the level
# count is enforced by the family itself, not here.
_FACTOR_LEVEL_FAMILIES = frozenset({"MN3", "MULTIN"})


def response_domain_for(family_name: str) -> ResponseDomain:
    """Return the R-equivalent `y.valid` predicate for `family_name`.

    Unknown families get :func:`_all_finite`, standing in for R's very common
    `y.valid = function(y) TRUE` while still rejecting NaN/Inf, which no family
    can legitimately fit.
    """
    if family_name in _FACTOR_LEVEL_FAMILIES:
        return _all_finite
    predicate = _RESPONSE_DOMAINS.get(family_name)
    if predicate is not None:
        return predicate
    return _LOWER_RESPONSE_DOMAINS.get(family_name.lower(), _all_finite)


def validate_response(
    family_name: str,
    y: np.ndarray,
    *,
    raise_on_violation: bool = True,
) -> int:
    """Check `y` against the family's response domain.

    Parameters
    ----------
    family_name : str
        Family the response is being fitted with.
    y : np.ndarray
        Observed response; any shape is accepted.
    raise_on_violation : bool
        `True` (default) raises :class:`ResponseDomainError`, mirroring R's
        `stop()`.  `False` returns the offending count instead, for callers
        that collect issues rather than abort.

    Returns
    -------
    int
        Number of observations outside the support; always `0` on success.

    Raises
    ------
    ResponseDomainError
        If any value is out of support and `raise_on_violation` is `True`.
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
