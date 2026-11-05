"""R gamlss.tr truncated-distribution helpers (trun.d / trun.p / trun.q / trun.r).

R source: R/gamlss.tr/R/trun.{d,p,q,r}-*.R.

Each helper returns a closure over (family, type, par), exactly like R: for a
base family with density d and CDF p, the truncated distribution keeps only the
mass inside the truncation interval and renormalises.  The varying-truncation
mode is not ported yet and raises NotImplementedError.
"""
from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .distribution_registry import resolve
from .families import FamilyDefinition

_TYPES = ("left", "right", "both")


def _resolve(family: Any) -> FamilyDefinition:
    return family if isinstance(family, FamilyDefinition) else resolve(family)


def _is_discrete(fam: FamilyDefinition) -> bool:
    return str(fam.type).strip().lower().startswith("discrete")


def _pv(fam: FamilyDefinition, params: dict[str, Any]) -> list[Any]:
    try:
        return [params[name] for name in fam.parameters]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"missing family parameter {exc.args[0]!r} for {fam.name}") from exc


def _check_par(par: Any, trunc_type: str) -> tuple[float, ...]:
    if trunc_type not in _TYPES:
        raise ValueError(f"type must be one of {_TYPES}, got {trunc_type!r}")
    values = tuple(np.atleast_1d(np.asarray(par, dtype=float)).tolist())
    if trunc_type == "both" and len(values) != 2:
        raise ValueError("the length of par should be 2")
    if trunc_type != "both" and len(values) != 1:
        raise ValueError("the length of par should be 1")
    return values


def _unsupported(varying: bool) -> None:
    if varying:
        raise NotImplementedError("varying truncation is not ported yet")


def trun_d(par: Any, family: Any = "NO", type: str = "left", varying: bool = False) -> Callable[..., Any]:
    """R trun.d: truncated density closure fun(x, log=False, **params)."""

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(x: Any, log: bool = False, **params: Any) -> Any:
        xs = jnp.asarray(x, dtype=jnp.float64)
        pv = _pv(fam, params)
        log_density = fam.d(xs, *pv, log=True)
        if type == "left":
            norm = jnp.log(1.0 - fam.p(p0, *pv))
            outside = (xs <= p0) if discrete else (xs < p0)
        elif type == "right":
            base = p0 - 1 if discrete else p0
            norm = jnp.log(fam.p(base, *pv))
            outside = (xs >= p0) if discrete else (xs > p0)
        else:
            if discrete:
                norm = jnp.log(fam.p(p1, *pv) - fam.p(p0, *pv))
                outside = (xs <= p0) | (xs >= p1)
            else:
                norm = jnp.log(fam.p(p1, *pv) - fam.p(p0, *pv))
                outside = (xs < p0) | (xs > p1)
        out = log_density - norm
        if log:
            return jnp.where(outside, jnp.nan, out)
        return jnp.where(outside, 0.0, jnp.exp(out))

    return fun


def trun_p(par: Any, family: Any = "NO", type: str = "left", varying: bool = False) -> Callable[..., Any]:
    """R trun.p: truncated CDF closure fun(q, lower_tail=True, log_p=False, **params)."""

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(q: Any, lower_tail: bool = True, log_p: bool = False, **params: Any) -> Any:
        qs = jnp.asarray(q, dtype=jnp.float64)
        pv = _pv(fam, params)
        if type == "left":
            cof = (fam.p(qs, *pv) - fam.p(p0, *pv)) / (1.0 - fam.p(p0, *pv))
            outside = (qs <= p0) if discrete else (qs < p0)
        elif type == "right":
            base = p0 - 1 if discrete else p0
            cof = fam.p(qs, *pv) / fam.p(base, *pv)
            outside = (qs >= p0) if discrete else (qs > p0)
        else:
            if discrete:
                cof = (fam.p(qs, *pv) - fam.p(p0, *pv)) / (fam.p(p1 - 1, *pv) - fam.p(p0, *pv))
                outside = (qs <= p0) | (qs >= p1)
            else:
                cof = (fam.p(qs, *pv) - fam.p(p0, *pv)) / (fam.p(p1, *pv) - fam.p(p0, *pv))
                outside = (qs < p0) | (qs > p1)
        cof = cof if lower_tail else 1.0 - cof
        if log_p:
            return jnp.where(outside, jnp.nan, jnp.log(cof))
        return jnp.where(outside, 0.0, cof)

    return fun


def trun_q(par: Any, family: Any = "NO", type: str = "left", varying: bool = False) -> Callable[..., Any]:
    """R trun.q: truncated quantile closure fun(p, lower_tail=True, log_p=False, **params)."""

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(prob: Any, lower_tail: bool = True, log_p: bool = False, **params: Any) -> Any:
        prob = jnp.asarray(prob, dtype=jnp.float64)
        if log_p:
            prob = jnp.exp(prob)
        if bool(jnp.any(prob <= 0.0)) or bool(jnp.any(prob >= 1.0)):
            raise ValueError("p must be between 0 and 1")
        prob = prob if lower_tail else 1.0 - prob
        pv = _pv(fam, params)
        if type == "left":
            pp = fam.p(p0, *pv)
            return fam.q(pp + prob * (1.0 - pp), *pv)
        if type == "right":
            pp = fam.p(p0 - 1, *pv) if discrete else fam.p(p0, *pv)
            return fam.q(prob * pp, *pv)
        pp1 = fam.p(p0, *pv)
        pp2 = fam.p(p1 - 1, *pv) if discrete else fam.p(p1, *pv)
        return fam.q(prob * (pp2 - pp1) + pp1, *pv)

    return fun


def trun_r(par: Any, family: Any = "NO", type: str = "left", varying: bool = False) -> Callable[..., Any]:
    """R trun.r: truncated random-generation closure fun(n, **params).

    R draws runif(n); exact stream parity is impossible, so this returns a
    numerically correct inverse-CDF sample (same distribution, different stream).
    """

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(n: int, **params: Any) -> Any:
        if n <= 0:
            raise ValueError("n must be a positive integer")
        n = int(np.ceil(n))
        u = jnp.asarray(np.random.uniform(size=n), dtype=jnp.float64)
        pv = _pv(fam, params)
        if type == "left":
            pp = fam.p(p0, *pv)
            return fam.q(pp + u * (1.0 - pp), *pv)
        if type == "right":
            pp = fam.p(p0 - 1, *pv) if discrete else fam.p(p0, *pv)
            return fam.q(u * pp, *pv)
        pp1 = fam.p(p0, *pv)
        pp2 = fam.p(p1 - 1, *pv) if discrete else fam.p(p1, *pv)
        return fam.q(u * (pp2 - pp1) + pp1, *pv)

    return fun


__all__ = ["trun_d", "trun_p", "trun_q", "trun_r"]
