# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add additive terms: ga() / ba() and their fitting callbacks (D2.248).

R's ga()/ba() are *term constructors*: they run mgcv::gam()/bam() with fit=FALSE
to obtain a basis, then gamlss.ga()/gamlss.ba() refit that basis for the working
response z with weights w.  The port keeps that interface and the same object
shape, but delegates the basis and the penalty to OmniLSS's existing smooth
machinery:

    gamlss(y ~ ga(~ pb(x)))  is exactly  gamlss(y ~ pb(x))

That equality is the contract this module pins.  It is deliberately *not* a
claim of numerical parity with mgcv's REML smoothing-parameter selection: mgcv
is a separate estimator with its own penalty and is not reimplemented here.
Calling the callback directly (gamlss_ga / gamlss_ba) performs the weighted
least-squares refit of the stored basis, which is the part of R's callback that
does not depend on mgcv.

R source: R/gamlss.add/R/ga_new.R (ga) and R/gamlss.add/R/ba.R (ba, which uses
mgcv::bam -- "big additive models", not backfitting).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .add_controls import ga_control
from .formula_parser import ParsedFormula, build_design_matrix, parse_formula

_TERM_RESPONSE = "__additive_response__"


@dataclass(frozen=True)
class AdditiveTerm:
    """R's ga()/ba() term: an inner formula plus its control list."""

    kind: str
    formula: str
    parsed: ParsedFormula
    control: Mapping[str, Any]


@dataclass(frozen=True)
class AdditiveFit:
    """Result of a gamlss.ga / gamlss.ba callback on a working response."""

    kind: str
    formula: str
    fitted: np.ndarray
    coefficients: np.ndarray
    design: np.ndarray
    smooth_info: Mapping[str, Any]
    iterations: int = 1


def _term(kind: str, formula: Any, control: Any, kwargs: Mapping[str, Any]) -> AdditiveTerm:
    if not isinstance(formula, str) or not formula.strip().startswith("~"):
        raise ValueError(
            f"formula argument in {kind}() needs a formula starting with ~"
        )
    if control is None:
        settings = ga_control(**dict(kwargs))
    elif isinstance(control, Mapping):
        settings = dict(control)
    else:
        settings = control
    parsed = parse_formula(f"{_TERM_RESPONSE} {formula.strip()}")
    return AdditiveTerm(kind=kind, formula=formula.strip(), parsed=parsed, control=settings)


def ga(formula: Any = None, control: Any = None, **kwargs: Any) -> AdditiveTerm:
    """R ga(): declare an additive term fitted with gam-style splines."""

    return _term("ga", formula, control, kwargs)


def ba(formula: Any = None, control: Any = None, **kwargs: Any) -> AdditiveTerm:
    """R ba(): declare an additive term fitted with bam-style splines.

    mgcv's bam is a large-data variant of gam, not a backfitting loop; the port
    therefore shares ga's design and fit path and only records the different
    kind.
    """

    return _term("ba", formula, control, kwargs)


def _fit_additive(
    kind: str,
    data: Mapping[str, Any],
    z: Any,
    w: Any,
    formula: Any,
) -> AdditiveFit:
    if data is None:
        raise ValueError("the data argument is needed")
    if not isinstance(formula, str) or not formula.strip().startswith("~"):
        raise ValueError(
            f"formula argument in {kind}() needs a formula starting with ~"
        )
    working = np.asarray(z, dtype=np.float64).ravel()
    weights = np.ones(working.size) if w is None else np.asarray(w, dtype=np.float64).ravel()
    if weights.size != working.size:
        raise ValueError("z and w must have the same length")
    parsed = parse_formula(f"{_TERM_RESPONSE} {formula.strip()}")
    augmented = dict(data)
    augmented[_TERM_RESPONSE] = working
    design, smooth_info = build_design_matrix(parsed, augmented, fit_smooths=True)
    design = np.asarray(design, dtype=np.float64)
    root_w = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(design * root_w[:, None], working * root_w, rcond=None)
    fitted = design @ coefficients
    return AdditiveFit(
        kind=kind,
        formula=formula.strip(),
        fitted=fitted,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        design=design,
        smooth_info=smooth_info,
        iterations=1,
    )


def gamlss_ga(
    data: Mapping[str, Any] = None,
    z: Any = None,
    w: Any = None,
    formula: Any = None,
    control: Any = None,
    **kwargs: Any,
) -> AdditiveFit:
    """R gamlss.ga: weighted refit of a ga() basis for the working response z."""

    return _fit_additive("ga", data, z, w, formula)


def gamlss_ba(
    data: Mapping[str, Any] = None,
    z: Any = None,
    w: Any = None,
    formula: Any = None,
    control: Any = None,
    **kwargs: Any,
) -> AdditiveFit:
    """R gamlss.ba: weighted refit of a ba() basis for the working response z."""

    return _fit_additive("ba", data, z, w, formula)


__all__ = [
    "AdditiveFit",
    "AdditiveTerm",
    "ba",
    "ga",
    "gamlss_ba",
    "gamlss_ga",
]
