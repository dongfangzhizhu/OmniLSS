"""R gamlss.tr truncated-distribution helpers (trun.d / trun.p / trun.q / trun.r).

R source: R/gamlss.tr/R/trun.{d,p,q,r}-*.R.

Each helper returns a closure over (family, type, par), exactly like R: for a
base family with density d and CDF p, the truncated distribution keeps only the
mass inside the truncation interval and renormalises.  The varying-truncation
mode is not ported yet and raises NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class TruncatedFamily(FamilyDefinition):
    """A truncated family: a FamilyDefinition carrying the truncation spec."""

    truncation_type: str = "left"
    truncation_parameters: tuple[float, ...] = ()
    base_name: str = "NO"
    base_family: Any = None
    extra_name: str = "tr"

    def support(self) -> tuple[float | None, float | None]:
        """(lower, upper) bounds of the truncated support; None means open."""

        if self.truncation_type == "left":
            return (float(self.truncation_parameters[0]), None)
        if self.truncation_type == "right":
            return (None, float(self.truncation_parameters[0]))
        return (
            float(self.truncation_parameters[0]),
            float(self.truncation_parameters[1]),
        )

    def y_valid(self, y: Any) -> Any:
        """Response support implied by the truncation masks in trun.d/trun.p.

        Continuous truncation keeps the boundaries (outside means strictly
        beyond them); discrete truncation excludes the boundary value, because
        the mass at the cut belongs to the removed tail.
        """

        values = jnp.asarray(y, dtype=jnp.float64)
        lower, upper = self.support()
        discrete = str(self.type).strip().lower().startswith("discrete")
        result = jnp.asarray(True)
        if lower is not None:
            result = result & (values >= lower + 1.0 if discrete else values >= lower)
        if upper is not None:
            result = result & (values < upper if discrete else values <= upper)
        return result


def trun(
    par: Any = 0,
    family: Any = "NO",
    type: str = "left",
    extra_name: str = "tr",
    local: bool = True,
    delta: Any = None,
    varying: bool = False,
    **kwargs: Any,
) -> TruncatedFamily:
    """R trun(): build the truncated family object.

    local is accepted for call-site compatibility (R uses it to choose the frame
    that receives the generated d/p); Python returns them on the family object.
    """

    base = _resolve(family)
    par_values = _check_par(par, type)
    d = trun_d(par_values, base, type, varying=varying)
    p = trun_p(par_values, base, type, varying=varying)
    # R's trun() regenerates only d and p; q and r stay the base family's.
    # Use trun_q / trun_r directly when the truncation must be reflected in
    # quantiles or draws.
    q = base.q
    r = base.r

    def g_dev_inc(y: Any, **params: Any) -> Any:
        return -2.0 * d(y, log=True, **params)

    return TruncatedFamily(
        name=f"{base.name}{extra_name}",
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
        q=q,
        r=r,
        constructor_parameters=base.constructor_parameters,
        fixed_parameter_defaults=base.fixed_parameter_defaults,
        truncation_type=type,
        truncation_parameters=tuple(float(value) for value in par_values),
        base_name=base.name,
        base_family=base,
        extra_name=str(extra_name),
    )


def gen_trun(
    par: Any = 0,
    family: Any = "NO",
    type: str = "left",
    extra_name: str = "tr",
    **kwargs: Any,
) -> dict[str, Any]:
    """R gen.trun: return the generated d/p/q/r plus the family object.

    R assigns dNOtr / pNOtr / qNOtr / rNOtr / NOtr into the caller frame;
    Python returns a mapping carrying both those names and the generic keys.
    """

    generated = trun(par=par, family=family, type=type, extra_name=extra_name, local=False)
    base = generated.base_name
    suffix = generated.extra_name
    d_name, p_name, q_name, r_name = (
        f"d{base}{suffix}",
        f"p{base}{suffix}",
        f"q{base}{suffix}",
        f"r{base}{suffix}",
    )
    fam_name = generated.name
    return {
        "d": generated.d,
        "p": generated.p,
        "q": generated.q,
        "r": generated.r,
        "family": generated,
        d_name: generated.d,
        p_name: generated.p,
        q_name: generated.q,
        r_name: generated.r,
        fam_name: generated,
    }


__all__ = [
    "TruncatedFamily",
    "gen_trun",
    "trun",
    "trun_d",
    "trun_p",
    "trun_q",
    "trun_r",
]
