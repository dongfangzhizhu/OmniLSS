"""Lightweight distribution interface utilities over :class:`FamilyDefinition`.

R reference: `gamlss.dist/R/distributions3.R` S4 helper surface (`pdf`, `cdf`,
`quantile`, `random`, support/moment predicates).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition


_ANALYTIC_MEAN_FAMILIES = frozenset({"NO", "NO2", "TF", "TF2", "GA", "LOGNO", "PO", "EXP", "BI"})
_ANALYTIC_VARIANCE_FAMILIES = frozenset({"NO", "NO2", "TF", "TF2", "GA", "LOGNO", "EXP", "PO", "BI"})
_ANALYTIC_SKEWNESS_FAMILIES = frozenset({"NO", "NO2", "GA", "LOGNO", "EXP", "PO", "BI"})
_ANALYTIC_KURTOSIS_FAMILIES = frozenset({"NO", "NO2", "GA", "LOGNO", "EXP", "PO", "BI"})


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


def moment_availability(family: FamilyDefinition) -> dict[str, str]:
    """Declare availability tier for distribution-moment helpers.

    Returns a mapping for `mean`, `variance`, `skewness`, `kurtosis` with values:
    - ``analytic``: closed-form helper exists;
    - ``sample_fallback``: no analytic helper but `family.r` is available;
    - ``unavailable``: neither analytic helper nor sample fallback is available.
    """
    name = family.name
    has_r = family.r is not None
    tiers: dict[str, str] = {}
    for key, analytic_families in (
        ("mean", _ANALYTIC_MEAN_FAMILIES),
        ("variance", _ANALYTIC_VARIANCE_FAMILIES),
        ("skewness", _ANALYTIC_SKEWNESS_FAMILIES),
        ("kurtosis", _ANALYTIC_KURTOSIS_FAMILIES),
    ):
        if name in analytic_families:
            tiers[key] = "analytic"
        elif has_r:
            tiers[key] = "sample_fallback"
        else:
            tiers[key] = "unavailable"
    return tiers


def mean(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if params and fallback == "analytic":
        if family.name == "LOGNO" and len(params) >= 2:
            mu = np.asarray(params[0], dtype=float)
            sigma = np.asarray(params[1], dtype=float)
            return float(np.mean(np.exp(mu + 0.5 * np.square(sigma))))
        mu = float(np.mean(np.asarray(params[0], dtype=float)))
        if family.name in _ANALYTIC_MEAN_FAMILIES:
            return mu
    if params and fallback == "sample":
        return _sample_moments(family, *params, n=n, seed=seed)[0]
    raise NotImplementedError(f"Mean is not available for family {family.name!r}; set fallback='sample' when random generation exists")

def variance(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if family.name in {"NO", "NO2", "TF", "TF2"} and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        return float(np.mean(np.square(sigma)))
    if family.name == "GA" and len(params) >= 2:
        mu = np.asarray(params[0], dtype=float)
        sigma = np.asarray(params[1], dtype=float)
        return float(np.mean(np.square(sigma * mu)))
    if family.name == "LOGNO" and len(params) >= 2:
        mu = np.asarray(params[0], dtype=float)
        sigma = np.asarray(params[1], dtype=float)
        s2 = np.square(sigma)
        return float(np.mean((np.exp(s2) - 1.0) * np.exp(2.0 * mu + s2)))
    if family.name == "EXP" and len(params) >= 1:
        mu = np.asarray(params[0], dtype=float)
        return float(np.mean(np.square(mu)))
    if family.name == "PO" and len(params) >= 1:
        mu = np.asarray(params[0], dtype=float)
        return float(np.mean(mu))
    if family.name == "BI" and len(params) >= 1:
        mu = np.asarray(params[0], dtype=float)
        return float(np.mean(mu * (1.0 - mu)))
    if fallback == "sample":
        return _sample_moments(family, *params, n=n, seed=seed)[1]
    raise NotImplementedError(f"Variance helper is not implemented analytically for family {family.name!r}")

def skewness(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if family.name in {"NO", "NO2"}:
        return 0.0
    if family.name == "GA" and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        # GA parameterization in this project: Var(Y)=sigma^2*mu^2 (shape=1/sigma^2)
        # Skewness of Gamma(shape=k) is 2/sqrt(k) -> 2*sigma
        return float(np.mean(2.0 * sigma))
    if family.name == "LOGNO" and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        s2 = np.square(sigma)
        return float(np.mean((np.exp(s2) + 2.0) * np.sqrt(np.maximum(np.exp(s2) - 1.0, 1e-12))))
    if family.name == "EXP":
        return 2.0
    if family.name == "PO" and len(params) >= 1:
        mu = float(np.mean(np.asarray(params[0], dtype=float)))
        return float(1.0 / np.sqrt(max(mu, 1e-12)))
    if family.name == "BI" and len(params) >= 1:
        p = float(np.mean(np.asarray(params[0], dtype=float)))
        return float((1.0 - 2.0 * p) / np.sqrt(max(p * (1.0 - p), 1e-12)))
    if fallback == "sample":
        return _sample_moments(family, *params, n=n, seed=seed)[2]
    raise NotImplementedError(f"Skewness helper is not implemented analytically for family {family.name!r}")

def kurtosis(family: FamilyDefinition, *params: Any, fallback: str = "analytic", n: int = 20000, seed: int = 12345) -> float:
    if family.name in {"NO", "NO2"}:
        return 3.0
    if family.name == "GA" and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        # Pearson kurtosis of Gamma(shape=k) is 3 + 6/k -> 3 + 6*sigma^2
        return float(np.mean(3.0 + 6.0 * np.square(sigma)))
    if family.name == "LOGNO" and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        s2 = np.square(sigma)
        return float(np.mean(np.exp(4.0 * s2) + 2.0 * np.exp(3.0 * s2) + 3.0 * np.exp(2.0 * s2)))
    if family.name == "EXP":
        return 9.0
    if family.name == "PO" and len(params) >= 1:
        mu = float(np.mean(np.asarray(params[0], dtype=float)))
        return float(3.0 + 1.0 / max(mu, 1e-12))
    if family.name == "BI" and len(params) >= 1:
        p = float(np.mean(np.asarray(params[0], dtype=float)))
        var = max(p * (1.0 - p), 1e-12)
        return float(3.0 + (1.0 - 6.0 * var) / var)
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
