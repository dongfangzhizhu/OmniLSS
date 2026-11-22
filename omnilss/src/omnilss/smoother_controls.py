"""R gamlss smoother control functions.

R source: R/gamlss/R/pb.R, PS.R, cubicSplines-10-08-12.R, cy.R, pbc.R, pbm.R,
pbo_original.R, pbp.R, pb_goingtozero.R, pvc.R.

Each control function returns the settings list gamlss() passes to the
corresponding gamlss.<smoother> fitting routine.  The validation is a set of
"if the value is out of range, warn and substitute the default" guards; the
returned mapping keeps R's key names so a ported call site reads the same
fields.

The smoothers themselves are documented in omnislss.smoother_admission: only
pb, ps and cs are admitted to the fitting routes, and these control objects are
the settings layer for all of them.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

_DEFAULT_METHODS = ("ML", "GAIC", "GCV")


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


def _match_arg(value: str | Sequence[str], choices: Sequence[str]) -> str:
    if isinstance(value, str):
        if value not in choices:
            raise ValueError(f"'arg' should be one of {list(choices)}, got {value!r}")
        return value
    raise ValueError(f"'arg' should be one of {list(choices)}")


def _base_defaults(
    inter: int,
    degree: int,
    order: int,
    k: float,
    *,
    min_order: int = 0,
) -> tuple[int, int, int, float]:
    if inter <= 0:
        _warn(
            "the value of inter supplied is less than 0, the value of 10 was used instead"
        )
        inter = 10
    if degree <= 0:
        _warn(
            "the value of degree supplied is less than zero or negative the default value of 3 was used instead"
        )
        degree = 3
    if order < min_order:
        _warn(
            f"the value of order supplied is less than {min_order} the default value of {max(min_order, 2)} was used instead"
        )
        order = max(min_order, 2)
    if k <= 0:
        _warn(
            "the value of GAIC/GCV penalty supplied is less than zero the default value of 2 was used instead"
        )
        k = 2
    return inter, degree, order, k


def pb_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    quantiles: bool = False,
    method: str = "ML",
    k: float = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pb.control: settings for the P-spline smoother pb()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "quantiles": bool(quantiles),
        "method": _match_arg(method, _DEFAULT_METHODS),
        "k": k,
    }


def cs_control(
    cv: bool = False,
    all_knots: bool = True,
    nknots: int | None = None,
    keep_data: bool = True,
    df_offset: float = 0,
    penalty: float = 1.4,
    control_spar: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """R cs.control: settings for the cubic smoothing spline cs()."""

    if df_offset < 0:
        _warn(
            "the value of df.offset supplied is negative the default value of 0 was used instead"
        )
        df_offset = 0
    if penalty < 0:
        _warn(
            "the value of penalty supplied is negative the default value of 1 was used instead"
        )
        penalty = 1
    return {
        "cv": cv,
        "all.knots": all_knots,
        "nknots": nknots,
        "keep.data": keep_data,
        "df.offset": df_offset,
        "penalty": penalty,
        "control.spar": dict(control_spar or {}),
    }


def cy_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    method: str = "ML",
    k: float = 2,
    ts: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """R cy.control: settings for the cyclic P-spline cy()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "method": _match_arg(method, (*_DEFAULT_METHODS, "EM", "ML-1")),
        "k": k,
        "ts": bool(ts),
    }


def pbc_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    method: str = "ML",
    k: float = 2,
    sin: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pbc.control: settings for the convex/concave P-spline pbc()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "method": _match_arg(method, _DEFAULT_METHODS),
        "k": k,
        "sin": bool(sin),
    }


def pbm_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    quantiles: bool = False,
    method: str = "ML",
    k: float = 2,
    kappa: float = 1e10,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pbm.control: settings for the monotone P-spline pbm()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "quantiles": bool(quantiles),
        "method": _match_arg(method, _DEFAULT_METHODS),
        "k": k,
        "kappa": kappa,
    }


def pbo_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    quantiles: bool = False,
    method: str = "ML",
    k: float = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pbo.control: settings for the original P-spline pbo()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "quantiles": bool(quantiles),
        "method": _match_arg(method, (*_DEFAULT_METHODS, "EM", "ML-1")),
        "k": k,
    }


def pbp_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    quantiles: bool = False,
    method: str = "ML",
    k: float = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pbp.control: settings for the P-spline with positivity pbp()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "quantiles": bool(quantiles),
        "method": _match_arg(method, _DEFAULT_METHODS),
        "k": k,
    }


def pbz_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: Sequence[float] = (0.0001, 0.0001),
    quantiles: bool = False,
    method: str = "ML",
    k: float = 2,
    lim: float = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pbz.control: settings for the going-to-zero P-spline pbz().

    R requires order >= 2 here, so the guard substitutes 2 rather than 0.
    """

    inter, degree, order, k = _base_defaults(inter, degree, order, k, min_order=2)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": list(start),
        "quantiles": bool(quantiles),
        "method": _match_arg(method, _DEFAULT_METHODS),
        "k": k,
        "lim": lim,
    }


def pvc_control(
    inter: int = 20,
    degree: int = 3,
    order: int = 2,
    start: float = 10,
    quantiles: bool = False,
    method: str = "ML",
    k: float = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """R pvc.control: settings for the varying-coefficient term pvc()."""

    inter, degree, order, k = _base_defaults(inter, degree, order, k)
    return {
        "inter": inter,
        "degree": degree,
        "order": order,
        "start": start,
        "quantiles": bool(quantiles),
        "method": _match_arg(method, _DEFAULT_METHODS),
        "k": k,
    }


__all__ = [
    "cs_control",
    "cy_control",
    "pbc_control",
    "pbm_control",
    "pbo_control",
    "pbp_control",
    "pb_control",
    "pbz_control",
    "pvc_control",
]
