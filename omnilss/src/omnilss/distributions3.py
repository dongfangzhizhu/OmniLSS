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


def mean(family: FamilyDefinition, *params: Any) -> float:
    if params:
        return float(np.mean(np.asarray(params[0], dtype=float)))
    raise NotImplementedError(f"Mean is not parameterized for family {family.name!r} without params")


def variance(family: FamilyDefinition, *params: Any) -> float:
    if family.name in {"NO", "TF", "TF2"} and len(params) >= 2:
        sigma = np.asarray(params[1], dtype=float)
        return float(np.mean(np.square(sigma)))
    raise NotImplementedError(f"Variance helper is not implemented for family {family.name!r}")


def skewness(family: FamilyDefinition, *params: Any) -> float:
    if family.name in {"NO"}:
        return 0.0
    raise NotImplementedError(f"Skewness helper is not implemented for family {family.name!r}")


def kurtosis(family: FamilyDefinition, *params: Any) -> float:
    if family.name in {"NO"}:
        return 3.0
    raise NotImplementedError(f"Kurtosis helper is not implemented for family {family.name!r}")
