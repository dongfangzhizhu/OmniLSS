"""R gamlss.cens censored-family helpers.

R source: R/gamlss.cens/R/{cens.d,cens.p,gen.cens,cens}.R.

A censored response is a Surv-like record: (time, status) for right/left
censoring, or (time1, time2, status) for interval censoring.  Status codes
follow gamlss: 0 = right censored, 1 = event/death, 2 = left censored,
3 = interval censored.

cens_d / cens_p return closures over (family, type); cens_q returns the base
family quantile function; cens() builds a CensoredFamily (a FamilyDefinition
with a censored density/deviance) suitable for the fitting engine; gen_cens()
returns the generated d/p/q/family in a mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .distribution_registry import resolve
from .families import FamilyDefinition

_TYPES = ("right", "left", "interval")
_SURV_RE = re.compile(r"^\s*Surv\s*\((.*)\)\s*$", re.IGNORECASE)


def _resolve(family: Any) -> FamilyDefinition:
    return family if isinstance(family, FamilyDefinition) else resolve(family)


def _pv(fam: FamilyDefinition, params: dict[str, Any]) -> list[Any]:
    try:
        return [params[name] for name in fam.parameters]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"missing family parameter {exc.args[0]!r} for {fam.name}"
        ) from exc


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


def is_surv_formula(formula: str) -> bool:
    """Whether the formula response is a Surv(...) expression."""

    if "~" not in formula:
        return False
    return _SURV_RE.match(formula.split("~", 1)[0].strip()) is not None


def extract_surv_response(formula: str, data: Any) -> dict[str, Any]:
    """Parse Surv(time, status) / Surv(t1, t2, status) into structured arrays."""

    if "~" not in formula:
        raise ValueError("formula must contain '~'")
    response = formula.split("~", 1)[0].strip()
    match = _SURV_RE.match(response)
    if match is None:
        raise ValueError(f"response is not a Surv(...) expression: {response!r}")
    raw_args = [part.strip() for part in match.group(1).split(",")]
    names = [part for part in raw_args if "=" not in part]
    if len(names) == 2:
        time_name, status_name = names
        t1 = np.asarray(data[time_name], dtype=float)
        status = np.asarray(data[status_name], dtype=float)
        t2 = t1
    elif len(names) == 3:
        t1_name, t2_name, status_name = names
        t1 = np.asarray(data[t1_name], dtype=float)
        t2 = np.asarray(data[t2_name], dtype=float)
        status = np.asarray(data[status_name], dtype=float)
    else:
        raise ValueError("Surv() needs (time, status) or (time1, time2, status)")
    return {"time": t1, "time2": t2, "status": status}


def cens_d(
    family: Any = "NO", type: str = "right", **kwargs: Any
) -> Callable[..., Any]:
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
            out = jnp.where(
                event, log_pdf, fam.p(t1, *pv, lower_tail=False, log_p=True)
            )
        else:
            right = jnp.where(
                status == 0.0, fam.p(t1, *pv, lower_tail=False, log_p=True), 0.0
            )
            death = jnp.where(status == 1.0, log_pdf, 0.0)
            left = jnp.where(status == 2.0, fam.p(t1, *pv, log_p=True), 0.0)
            interval = jnp.where(
                status == 3.0, jnp.log(fam.p(t2, *pv) - fam.p(t1, *pv)), 0.0
            )
            out = right + death + left + interval
        return out if log else jnp.exp(out)

    return fun


def cens_p(
    family: Any = "NO", type: str = "right", **kwargs: Any
) -> Callable[..., Any]:
    """R cens.p: randomised PIT closure fun(y, log=False, **params)."""

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
            right = np.where(
                np.asarray(status) == 0.0, np.random.uniform(cdf1, 1.0, size=n), 0.0
            )
            death = np.where(np.asarray(status) == 1.0, cdf1, 0.0)
            left = np.where(
                np.asarray(status) == 2.0, np.random.uniform(0.0, cdf1, size=n), 0.0
            )
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


@dataclass(frozen=True)
class CensoredFamily(FamilyDefinition):
    """A censored family: a FamilyDefinition carrying the censoring type."""

    censoring_type: str = "right"
    base_name: str = "NO"
    base_family: Any = None

    def loglik(self, y: Any, **params: Any) -> Any:
        """Censored log-likelihood contributions (elementwise)."""

        return self.d(y, log=True, **params)


def _build_family(base: FamilyDefinition, type_: str, name: str) -> CensoredFamily:
    _check_type(type_)
    d = cens_d(base, type_)
    p = cens_p(base, type_)

    def g_dev_inc(y: Any, **kwargs: Any) -> Any:
        params = {key: kwargs[key] for key in base.parameters}
        return -2.0 * d(y, log=True, **params)

    generated = f"{base.name}{type_[0]}{str(name)[0]}"
    return CensoredFamily(
        name=generated,
        parameters=tuple(base.parameters),
        g_dev_inc=g_dev_inc,
        type=base.type,
        links=base.links,
        link_functions=base.link_functions,
        link_inverses=base.link_inverses,
        link_derivatives=base.link_derivatives,
        score_functions=base.score_functions,
        hessian_functions=base.hessian_functions,
        fixed_parameters=base.fixed_parameters,
        d=d,
        p=p,
        q=base.q,
        constructor_parameters=base.constructor_parameters,
        fixed_parameter_defaults=base.fixed_parameter_defaults,
        censoring_type=type_,
        base_name=base.name,
        base_family=base,
    )


def cens(
    family: Any = "NO",
    type: str = "right",
    name: str = "cens",
    local: bool = True,
    delta: Any = None,
    **kwargs: Any,
) -> CensoredFamily:
    """R cens(): build a censored family object from a base family."""

    return _build_family(_resolve(family), type, name)


def gen_cens(
    family: Any = "NO",
    type: str = "right",
    name: str = "cens",
    **kwargs: Any,
) -> dict[str, Any]:
    """R gen.cens: return the generated d/p/q/family (R assigns them in-frame)."""

    base = _resolve(family)
    cf = _build_family(base, type, name)
    generated = cf.name
    return {
        "d": cf.d,
        "p": cf.p,
        "q": cf.q,
        "family": cf,
        f"d{generated}": cf.d,
        f"p{generated}": cf.p,
        f"q{generated}": cf.q,
        generated: cf,
    }


__all__ = [
    "CensoredFamily",
    "cens",
    "cens_d",
    "cens_p",
    "cens_q",
    "extract_surv_response",
    "gen_cens",
    "is_surv_formula",
]
