"""Residual and randomized quantile residual helpers extracted from fitting.py."""

from __future__ import annotations

import math
from statistics import NormalDist

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition

_STANDARD_NORMAL = NormalDist()


def _normal_quantile(probabilities: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    flattened = probs.ravel()
    quantiles = np.array(
        [_STANDARD_NORMAL.inv_cdf(float(value)) for value in flattened],
        dtype=np.float64,
    )
    return quantiles.reshape(probs.shape)




def _poisson_cdf_scalar(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mu), np.finfo(np.float64).eps)
    pmf = math.exp(-mu)
    total = pmf
    for index in range(1, k + 1):
        pmf *= mu / index
        total += pmf
    return float(min(max(total, 0.0), 1.0))




def _negative_binomial_cdf_scalar(k: int, mu: float, sigma: float) -> float:
    if k < 0:
        return 0.0
    eps = np.finfo(np.float64).eps
    mu = max(float(mu), eps)
    sigma = max(float(sigma), eps)
    size = 1.0 / sigma
    pmf = math.exp(size * math.log(size / (size + mu)))
    total = pmf
    for index in range(1, k + 1):
        pmf *= ((index - 1 + size) / index) * (mu / (size + mu))
        total += pmf
    return float(min(max(total, 0.0), 1.0))




def _geometric_cdf_scalar(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mu), np.finfo(np.float64).eps)
    ratio = mu / (1.0 + mu)
    return float(min(max(1.0 - ratio ** (k + 1), 0.0), 1.0))




def _zip_cdf_scalar(k: int, mu: float, sigma: float) -> float:
    if k < 0:
        return 0.0
    eps = np.finfo(np.float64).eps
    sigma = min(max(float(sigma), eps), 1.0 - eps)
    poisson_cdf = _poisson_cdf_scalar(k, mu)
    return float(min(max(sigma + (1.0 - sigma) * poisson_cdf, 0.0), 1.0))




def _discrete_midpoint_rqres(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    midpoint = np.zeros_like(y_arr, dtype=np.float64)

    for index, value in enumerate(y_arr):
        observed = int(np.floor(value))
        mu_value = float(mu_arr[index])
        sigma_value = None if sigma_arr is None else float(sigma_arr[index])

        if family.name == "PO":
            lower = _poisson_cdf_scalar(observed - 1, mu_value)
            upper = _poisson_cdf_scalar(observed, mu_value)
        elif family.name == "BI":
            prob = min(max(mu_value, np.finfo(np.float64).eps), 1.0 - np.finfo(np.float64).eps)
            lower = 0.0 if observed <= 0 else 1.0 - prob
            upper = 1.0 - prob if observed <= 0 else 1.0
        elif family.name == "GEOM":
            lower = _geometric_cdf_scalar(observed - 1, mu_value)
            upper = _geometric_cdf_scalar(observed, mu_value)
        elif family.name == "NBI":
            lower = _negative_binomial_cdf_scalar(observed - 1, mu_value, float(sigma_value))
            upper = _negative_binomial_cdf_scalar(observed, mu_value, float(sigma_value))
        elif family.name == "ZIP":
            lower = _zip_cdf_scalar(observed - 1, mu_value, float(sigma_value))
            upper = _zip_cdf_scalar(observed, mu_value, float(sigma_value))
        else:
            raise NotImplementedError(f"rqres is not implemented for family {family.name!r}")

        midpoint[index] = 0.5 * (lower + upper)

    return _normal_quantile(midpoint)




def _build_rqres_callable(family: FamilyDefinition) -> Any | None:
    if family.name not in {"PO", "BI", "GEOM", "NBI", "ZIP"}:
        return None

    def rqres(**kwargs: Any) -> jnp.ndarray:
        y = np.asarray(kwargs["y"], dtype=np.float64)
        mu = np.asarray(kwargs["mu"], dtype=np.float64)
        sigma_value = kwargs.get("sigma")
        sigma = None if sigma_value is None else np.asarray(sigma_value, dtype=np.float64)
        values = _discrete_midpoint_rqres(family, y=y, mu=mu, sigma=sigma)
        return jnp.asarray(values, dtype=jnp.float64)

    return rqres




def _compute_residuals(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | None,
) -> jnp.ndarray:
    if family.name == "LOGNO" and sigma is not None:
        log_y = np.log(np.maximum(y, np.finfo(np.float64).eps))
        return jnp.asarray((log_y - mu) / sigma, dtype=jnp.float64)
    if family.name == "GA" and sigma is not None and family.p is not None:
        probabilities = np.asarray(family.p(y, mu, sigma), dtype=np.float64)
        return jnp.asarray(_normal_quantile(probabilities), dtype=jnp.float64)
    if sigma is not None:
        return jnp.asarray((y - mu) / sigma, dtype=jnp.float64)
    return jnp.asarray((y - mu) / np.sqrt(np.maximum(mu, 1e-12)), dtype=jnp.float64)




__all__ = ["_build_rqres_callable", "_compute_residuals", "_discrete_midpoint_rqres", "_geometric_cdf_scalar", "_negative_binomial_cdf_scalar", "_normal_quantile", "_poisson_cdf_scalar", "_zip_cdf_scalar"]
