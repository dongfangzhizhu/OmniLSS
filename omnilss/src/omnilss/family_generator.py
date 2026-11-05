"""R gamlss.dist family generation: transformed families and distributions3 objects.

R source: R/gamlss.dist/R/genFamily.R and R/gamlss.dist/R/distributions3.R.

genFamily.R builds, from a continuous family defined on the whole real line, a
new family whose response is log- or logit-transformed:

    d_log(x)   = d_base(log(x)) * 1/x          (i.e. log-density - log(x))
    d_logit(x) = d_base(logit(x)) / (x (1-x))
    p_log(q)   = p_base(log(q))                p_logit(q) = p_base(logit(q))
    q_log(p)   = exp(q_base(p))                q_logit(p) = logistic(q_base(p))
    r_log(n)   = exp(r_base(n))                r_logit(n) = logistic(r_base(n))

R builds these by rewriting function bodies with deparse/gsub and assigning into
the caller frame; Python has no caller-frame assignment, so gen_Family returns a
mapping of the generated d/p/q/r and the family object instead (the same choice
made for gen_cens).

distributions3.R exposes GAMLSS(), a distribution object that binds one family
to a set of parameter values and carries pdf/cdf/quantile/random/mean/variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition

_TYPES = ("log", "logit")


def _check_type(type_: str) -> str:
    if type_ not in _TYPES:
        raise ValueError(f"'arg' should be one of {list(_TYPES)}, got {type_!r}")
    return type_


def _resolve(family: Any) -> FamilyDefinition:
    from .distribution_registry import resolve

    if isinstance(family, FamilyDefinition):
        return family
    if callable(family) and not isinstance(family, str):
        produced = family()
        if isinstance(produced, FamilyDefinition):
            return produced
        return resolve(str(produced))
    return resolve(str(family))


def _pv(fam: FamilyDefinition, params: Mapping[str, Any]) -> list[Any]:
    try:
        return [params[name] for name in fam.parameters]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"missing family parameter {exc.args[0]!r} for {fam.name}"
        ) from exc


def _transform(x: Any, type_: str) -> Any:
    xs = jnp.asarray(x, dtype=jnp.float64)
    if type_ == "log":
        return jnp.log(xs)
    return jnp.log(xs / (1.0 - xs))


def _inverse_transform(z: Any, type_: str) -> Any:
    zs = jnp.asarray(z, dtype=jnp.float64)
    if type_ == "log":
        return jnp.exp(zs)
    return 1.0 / (1.0 + jnp.exp(-zs))


def _log_jacobian(x: Any, type_: str) -> Any:
    xs = jnp.asarray(x, dtype=jnp.float64)
    if type_ == "log":
        return jnp.log(xs)
    return jnp.log(xs) + jnp.log1p(-xs)


def family_d(family: Any = "NO", type: str = "log", **kwargs: Any) -> Callable[..., Any]:
    """R Family.d: density closure for the log / logit transformed family."""

    type_ = _check_type(type)
    fam = _resolve(family)

    def fun(x: Any, log: bool = False, **params: Any) -> Any:
        pv = _pv(fam, params)
        xs = jnp.asarray(x, dtype=jnp.float64)
        log_density = fam.d(_transform(xs, type_), *pv, log=True) - _log_jacobian(xs, type_)
        return log_density if log else jnp.exp(log_density)

    return fun


def family_p(family: Any = "NO", type: str = "log", **kwargs: Any) -> Callable[..., Any]:
    """R Family.p: CDF closure for the log / logit transformed family."""

    type_ = _check_type(type)
    fam = _resolve(family)

    def fun(q: Any, lower_tail: bool = True, log_p: bool = False, **params: Any) -> Any:
        pv = _pv(fam, params)
        return fam.p(
            _transform(q, type_), *pv, lower_tail=lower_tail, log_p=log_p
        )

    return fun


def family_q(family: Any = "NO", type: str = "log", **kwargs: Any) -> Callable[..., Any]:
    """R Family.q: quantile closure for the log / logit transformed family."""

    type_ = _check_type(type)
    fam = _resolve(family)

    def fun(p: Any, lower_tail: bool = True, log_p: bool = False, **params: Any) -> Any:
        pv = _pv(fam, params)
        return _inverse_transform(
            fam.q(p, *pv, lower_tail=lower_tail, log_p=log_p), type_
        )

    return fun


def family_r(family: Any = "NO", type: str = "log", **kwargs: Any) -> Callable[..., Any]:
    """R Family.r: random-generation closure for the transformed family."""

    type_ = _check_type(type)
    fam = _resolve(family)

    def fun(n: int, key: Any = None, **params: Any) -> Any:
        pv = _pv(fam, params)
        if key is None:
            key = jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1)))
        return _inverse_transform(fam.r(key, int(np.ceil(n)), *pv), type_)

    return fun


@dataclass(frozen=True)
class GeneratedFamily(FamilyDefinition):
    """A log / logit transformed continuous family (R's Family() result)."""

    transform_type: str = "log"
    base_name: str = "NO"
    base_family: Any = None

    def y_valid(self, y: Any) -> Any:
        values = jnp.asarray(y, dtype=jnp.float64)
        if self.transform_type == "log":
            return jnp.all(values > 0.0)
        return jnp.all((values > 0.0) & (values < 1.0))


def Family(family: Any = "NO", type: str = "log", local: bool = True, **kwargs: Any) -> GeneratedFamily:
    """R Family: build the log / logit transformed family object.

    local is accepted for call-site compatibility with R (which uses it to pick
    the environment that receives the generated d/p/q/r); Python returns the
    generated callables on the family object instead.
    """

    type_ = _check_type(type)
    fam = _resolve(family)
    if not str(fam.type).strip().lower().startswith("continuous"):
        raise ValueError("the gamlss.family distribution should be continuous")

    d = family_d(fam, type_)
    p = family_p(fam, type_)
    q = family_q(fam, type_)
    r = family_r(fam, type_)

    def g_dev_inc(y: Any, **params: Any) -> Any:
        return -2.0 * d(y, log=True, **params)

    name = f"{type_}{fam.name}"
    return GeneratedFamily(
        name=name,
        parameters=tuple(fam.parameters),
        g_dev_inc=g_dev_inc,
        type=fam.type,
        links=fam.links,
        link_functions=fam.link_functions,
        link_inverses=fam.link_inverses,
        link_derivatives=fam.link_derivatives,
        score_functions=fam.score_functions,
        hessian_functions=fam.hessian_functions,
        fixed_parameters=fam.fixed_parameters,
        d=d,
        p=p,
        q=q,
        r=r,
        constructor_parameters=fam.constructor_parameters,
        fixed_parameter_defaults=fam.fixed_parameter_defaults,
        transform_type=type_,
        base_name=fam.name,
        base_family=fam,
    )


def gen_Family(family: Any = "NO", type: str = "log", **kwargs: Any) -> dict[str, Any]:
    """R gen.Family: return the generated d/p/q/r plus the family object.

    R assigns these into the parent frame under names such as dlogNO / logNO;
    Python returns a mapping with both the R-style names and the generic keys.
    """

    type_ = _check_type(type)
    fam = _resolve(family)
    generated = Family(fam, type_, local=False)
    prefix = type_[:5]
    base = fam.name
    d_name, p_name, q_name, r_name, fam_name = (
        f"d{prefix}{base}",
        f"p{prefix}{base}",
        f"q{prefix}{base}",
        f"r{prefix}{base}",
        f"{prefix}{base}",
    )
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


@dataclass
class GAMLSSDistribution:
    """R GAMLSS(): a family bound to one set of parameter values.

    Mirrors the distributions3 object: pdf / cdf / quantile / random plus
    mean / variance when the family provides them.
    """

    family_name: str
    family: FamilyDefinition
    params: dict[str, float] = field(default_factory=dict)

    def _pv(self) -> list[float]:
        missing = [name for name in self.family.parameters if name not in self.params]
        if missing:
            raise ValueError(f"missing parameters for {self.family_name}: {missing}")
        return [self.params[name] for name in self.family.parameters]

    def pdf(self, x: Any, log: bool = False) -> Any:
        return self.family.d(x, *self._pv(), log=log)

    def cdf(self, q: Any, lower_tail: bool = True, log_p: bool = False) -> Any:
        return self.family.p(q, *self._pv(), lower_tail=lower_tail, log_p=log_p)

    def quantile(self, p: Any, lower_tail: bool = True, log_p: bool = False) -> Any:
        return self.family.q(p, *self._pv(), lower_tail=lower_tail, log_p=log_p)

    def random(self, n: int, key: Any = None) -> Any:
        if key is None:
            key = jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1)))
        return self.family.r(key, int(np.ceil(n)), *self._pv())

    def mean(self) -> Any:
        fn = getattr(self.family, "mean", None)
        if fn is None:
            raise NotImplementedError(
                f"the mean is not implemented for the {self.family_name} family"
            )
        return fn(*self._pv())

    def variance(self) -> Any:
        fn = getattr(self.family, "variance", None)
        if fn is None:
            raise NotImplementedError(
                f"the variance is not implemented for the {self.family_name} family"
            )
        return fn(*self._pv())

    def __repr__(self) -> str:
        shown = ", ".join(f"{k} = {v}" for k, v in self.params.items())
        return f"GAMLSS distribution: {self.family_name} ({shown})"


def GAMLSS(family: Any = "NO", **params: Any) -> GAMLSSDistribution:
    """R GAMLSS(): bind a family to parameter values.

    R's signature is GAMLSS(family, mu, sigma, tau, nu); the Python version
    accepts the parameters by name so families with other parameter sets work.
    Warnings for parameters the family does not have are replicated as
    UserWarning.
    """

    fam = _resolve(family)
    import warnings

    recognized = set(fam.parameters)
    for name in params:
        if name not in recognized:
            warnings.warn(
                f"'{name}' is not a parameter of the '{fam.name}' family",
                UserWarning,
                stacklevel=2,
            )
    values = {name: params[name] for name in fam.parameters if name in params}
    return GAMLSSDistribution(family_name=fam.name, family=fam, params=values)


__all__ = [
    "Family",
    "Family_d",
    "Family_p",
    "Family_q",
    "Family_r",
    "GAMLSS",
    "GAMLSSDistribution",
    "GeneratedFamily",
    "family_d",
    "family_p",
    "family_q",
    "family_r",
    "gen_Family",
]

# R-style aliases (the coverage matcher normalises dots to underscores).
Family_d = family_d
Family_p = family_p
Family_q = family_q
Family_r = family_r
