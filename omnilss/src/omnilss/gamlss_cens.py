"""R gamlss.cens censored-family helpers.

R source: R/gamlss.cens/R/{cens.d,cens.p,gen.cens,cens}.R.

A censored response is a Surv-like record: (time, status) for right/left
censoring, or (time1, time2, status) for interval censoring.  Status codes
follow gamlss: 0 = right censored, 1 = event/death, 2 = left censored,
3 = interval censored.

cens_d / cens_p return closures over (family, type); cens_q returns the base
family quantile function (R's dummy generator); cens() builds a censored family
object carrying the censored density, PIT and log-likelihood; gen_cens() returns
the generated d/p/q/family in a mapping (R assigns them in the caller frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .distribution_registry import resolve
from .families import FamilyDefinition

_TYPES = ("right", "left", "interval")


def _resolve(family: Any) -> FamilyDefinition:
    return family if isinstance(family, FamilyDefinition) else resolve(family)


def _pv(fam: FamilyDefinition, params: dict[str, Any]) -> list[Any]:
    try:
        return [params[name] for name in fam.parameters]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"missing family parameter {exc.args[0]!r} for {fam.name}") from exc


def _check_type(type_: str) -> None:
    if type_ not in _TYPES:
        raise ValueError(f"type must be one of {_TYPES}, got {type_!r}")


def _surv(y: Any) -> tuple[Any, Any, Any]:
    """Return (time1, time2, status) arrays from a Surv-like object / mapping / matrix."""

    if isinstance(y, dict):
        t1 = np.asarray(y["time"], dtype=float)
        status = np.asarray(y["status"], dtype=float)
        t2 = np.asarray(y.get("time2", t1), dtype=float)
    elif hasattr(y, "time") and hasattr(y, "status"):
        t1 = np.asarray(y.time, dtype=float)
        status = np.asarray(y.status, dtype=float)
        t2 = np.asarray(getattr(y, "time2", t1), dtype=float)
    else:
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError("censored response must be a (n,2) or (n,3) array")
        if arr.shape[1] == 2:
            t1, status = arr[:, 0], arr[:, 1]
            t2 = t1
        else:
            t1, t2, status = arr[:, 0], arr[:, 1], arr[:, 2]
    return jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(status)


def cens_d(family: Any = "NO", type: str = "right", **kwargs: Any) -> Callable[..., Any]:
    """R cens.d: censored log-density closure fun(y, log=False, **params)."""

    _check_type(type)
    fam = _resolve(family)

    def fun(y: Any, log: bool = False, **params: Any) -> Any:
        t1, t2, status = _surv(y)
        pv = _pv(fam, params)
        event = status == 1.0
        log_pdf = fam.d(t1, *pv, log=True)
        if type == "left":
            out = jnp.where(event, log_pdf, fam.p(t1, *pv, log_p=True))
        elif type == "right":
            out = jnp.where(event, log_pdf, fam.p(t1, *pv, lower_tail=False, log_p=True))
        else:
            right = jnp.where(status == 0.0, fam.p(t1, *pv, lower_tail=False, log_p=True), 0.0)
            death = jnp.where(status == 1.0, log_pdf, 0.0)
            left = jnp.where(status == 2.0, fam.p(t1, *pv, log_p=True), 0.0)
            interval = jnp.where(status == 3.0, jnp.log(fam.p(t2, *pv) - fam.p(t1, *pv)), 0.0)
            out = right + death + left + interval
        return out if log else jnp.exp(out)

    return fun


def cens_p(family: Any = "NO", type: str = "right", **kwargs: Any) -> Callable[..., Any]:
    """R cens.p: randomised PIT closure fun(y, log=False, **params).

    The randomised branch uses numpy's RNG (R uses runif), so exact stream parity
    is impossible; event observations are deterministic and match R exactly.
    """

    _check_type(type)
    fam = _resolve(family)

    def fun(y: Any, log: bool = False, **params: Any) -> Any:
        t1, t2, status = _surv(y)
        pv = _pv(fam, params)
        n = int(t1.shape[0])
        cdf1 = np.asarray(fam.p(t1, *pv), dtype=float)
        event = np.asarray(status) == 1.0
        if type == "left":
            lower = np.random.uniform(0.0, cdf1, size=n)
            out = np.where(event, cdf1, lower)
        elif type == "right":
            upper = np.random.uniform(cdf1, 1.0, size=n)
            out = np.where(event, cdf1, upper)
        else:
            cdf2 = np.asarray(fam.p(t2, *pv), dtype=float)
            right = np.where(np.asarray(status) == 0.0, np.random.uniform(cdf1, 1.0, size=n), 0.0)
            death = np.where(np.asarray(status) == 1.0, cdf1, 0.0)
            left = np.where(np.asarray(status) == 2.0, np.random.uniform(0.0, cdf1, size=n), 0.0)
            interval = np.where(
                np.asarray(status) == 3.0,
                np.random.uniform(cdf1, cdf2, size=n),
                0.0,
            )
            out = right + death + left + interval
        result = jnp.asarray(out, dtype=jnp.float64)
        return jnp.log(result) if log else result

    return fun


def cens_q(family: Any = "NO", **kwargs: Any) -> Callable[..., Any]:
    """R cens.q: the base family quantile function (dummy generator)."""

    return _resolve(family).q


@dataclass
class CensoredFamily:
    """A censored family: base family + censoring type, with censored d/p/loglik."""

    name: str
    base_family: FamilyDefinition
    type: str
    d: Callable[..., Any]
    p: Callable[..., Any]
    q: Callable[..., Any]
    parameters: tuple[str, ...] = field(default_factory=tuple)

    def loglik(self, y: Any, **params: Any) -> Any:
        return self.d(y, log=True, **params)

    def g_dev_inc(self, y: Any, *params: Any) -> Any:
        """Censored global deviance increment (-2 * total log-likelihood)."""

        names = self.parameters
        kwargs = dict(zip(names, params))
        loglik = self.d(y, log=True, **kwargs)
        return -2.0 * jnp.sum(loglik)


def cens(
    family: Any = "NO",
    type: str = "right",
    name: str = "cens",
    local: bool = True,
    delta: Any = None,
    **kwargs: Any,
) -> CensoredFamily:
    """R cens(): build a censored family object from a base family."""

    _check_type(type)
    fam = _resolve(family)
    generated = f"{fam.name}{type[0]}{str(name)[0]}"
    return CensoredFamily(
        name=generated,
        base_family=fam,
        type=type,
        d=cens_d(fam, type),
        p=cens_p(fam, type),
        q=fam.q,
        parameters=tuple(fam.parameters),
    )


def gen_cens(
    family: Any = "NO",
    type: str = "right",
    name: str = "cens",
    **kwargs: Any,
) -> dict[str, Any]:
    """R gen.cens: return the generated d/p/q/family (R assigns them in-frame).

    The mapping carries both generic keys ("d", "p", "q", "family") and the
    R-style generated names (e.g. "dNOrc", "NOrc") so a ported call site can use
    either.
    """

    _check_type(type)
    fam = _resolve(family)
    generated = f"{fam.name}{type[0]}{str(name)[0]}"
    d = cens_d(fam, type)
    p = cens_p(fam, type)
    q = fam.q
    cf = CensoredFamily(
        name=generated,
        base_family=fam,
        type=type,
        d=d,
        p=p,
        q=q,
        parameters=tuple(fam.parameters),
    )
    return {
        "d": d,
        "p": p,
        "q": q,
        "family": cf,
        f"d{generated}": d,
        f"p{generated}": p,
        f"q{generated}": q,
        generated: cf,
    }


__all__ = ["CensoredFamily", "cens", "cens_d", "cens_p", "cens_q", "gen_cens"]
