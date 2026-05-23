"""Lightweight distribution interface utilities over :class:`FamilyDefinition`.

R reference: `gamlss.dist/R/distributions3.R` S4 helper surface (`pdf`, `cdf`,
`quantile`, `random`, support/moment predicates).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition


def pdf(family: FamilyDefinition, x: Any, *params: Any, log: bool = False) -> np.ndarray:
    vals = np.asarray(family.pdf(jnp.asarray(x), *[jnp.asarray(p) for p in params]))
    if log:
        return np.log(np.clip(vals, 1e-300, None))
    return vals


def log_pdf(family: FamilyDefinition, x: Any, *params: Any) -> np.ndarray:
    return pdf(family, x, *params, log=True)


def cdf(family: FamilyDefinition, q: Any, *params: Any) -> np.ndarray:
    if family.p is None:
        raise NotImplementedError(f"Family {family.name!r} does not define a CDF function")
    return np.asarray(family.p(jnp.asarray(q), *[jnp.asarray(p) for p in params]))


def quantile(family: FamilyDefinition, p: Any, *params: Any) -> np.ndarray:
    if family.q is None:
        raise NotImplementedError(f"Family {family.name!r} does not define a quantile function")
    return np.asarray(family.q(jnp.asarray(p), *[jnp.asarray(pv) for pv in params]))


def random(family: FamilyDefinition, key: Any, n: int, *params: Any) -> np.ndarray:
    if family.r is None:
        raise NotImplementedError(f"Family {family.name!r} does not define a random generator")
    return np.asarray(family.r(key, int(n), *[jnp.asarray(p) for p in params]))


def is_continuous(family: FamilyDefinition) -> bool:
    return family.type.lower().startswith("cont")


def is_discrete(family: FamilyDefinition) -> bool:
    return family.type.lower().startswith("disc")


def support(family: FamilyDefinition) -> str:
    t = family.type.lower()
    if "disc" in t:
        return "integers"
    if family.name in {"BE", "BEINF", "SIMPLEX", "GB1", "BEO", "BEOI", "BEZI"}:
        return "(0,1)"
    if family.name in {"GA", "IG", "WEI", "GG", "LOGNO", "EXP", "GIG", "PO", "NBI", "ZIP"}:
        return "(0,∞)"
    return "(-∞,∞)"


def mean(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if params:
        if fallback == "analytic":
            return float(np.mean(np.asarray(params[0], dtype=float)))
        if fallback == "sample":
            return _sample_moments(family, *params, n=n, seed=seed)[0]
    raise NotImplementedError(f"Mean is not available for family {family.name!r}; set fallback='sample' when random generation exists")

def variance(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if family.name in {"NO", "TF", "TF2"} and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        return float(np.mean(np.square(sigma)))
    if fallback == "sample":
        return _sample_moments(family, *params, n=n, seed=seed)[1]
    raise NotImplementedError(f"Variance helper is not implemented analytically for family {family.name!r}")

def skewness(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if family.name in {"NO"}:
        return 0.0
    if fallback == "sample":
        return _sample_moments(family, *params, n=n, seed=seed)[2]
    raise NotImplementedError(f"Skewness helper is not implemented analytically for family {family.name!r}")

def kurtosis(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if family.name in {"NO"}:
        return 3.0
    if fallback == "sample":
        return _sample_moments(family, *params, n=n, seed=seed)[3]
    raise NotImplementedError(f"Kurtosis helper is not implemented analytically for family {family.name!r}")

def _sample_moments(family: FamilyDefinition, *params: Any, n: int = 20000, seed: int = 12345) -> tuple[float, float, float, float]:
    """Monte-Carlo fallback moments using family random generator when available."""
    if family.r is None:
        raise NotImplementedError(f"Family {family.name!r} does not define random generation for fallback moments")
    from jax import random as jrandom

    key = jrandom.PRNGKey(seed)
    scalar_params = []
    for p in params:
        arr = jnp.asarray(p)
        if arr.ndim > 0:
            arr = jnp.asarray(arr).reshape(-1)[0]
        scalar_params.append(arr)
    draws = np.asarray(family.r(key, int(n), *scalar_params), dtype=float).ravel()
    if draws.size == 0:
        raise NotImplementedError(f"Family {family.name!r} produced no draws for fallback moments")
    mu = float(np.mean(draws))
    centered = draws - mu
    var = float(np.mean(centered ** 2))
    if var <= 0:
        return mu, var, 0.0, 0.0
    sd = float(np.sqrt(var))
    skew = float(np.mean((centered / sd) ** 3))
    kurt = float(np.mean((centered / sd) ** 4))
    return mu, var, skew, kurt
