"""R-aligned covariance interfaces.

R source reference:
- file: `gamlss/R/vcov-gamlss.R`
- functions: `vcov.gamlss`
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .model import GAMLSSModel
from .operations import coef


def _require_gamlss_method(object: GAMLSSModel) -> None:
    """Validate that the object is a GAMLSSModel."""
    if not hasattr(object, "g_dev"):
        raise TypeError("object must be a GAMLSSModel")


def _flatten_coefficients(object: GAMLSSModel) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Flatten all parameter coefficients into a single vector with layout."""
    blocks: list[np.ndarray] = []
    layout: list[tuple[str, int]] = []
    for parameter in object.par:
        values = np.asarray(coef(object, parameter), dtype=np.float64).ravel()
        blocks.append(values)
        layout.append((parameter, values.size))
    flat = np.concatenate(blocks) if blocks else np.array([], dtype=np.float64)
    return flat, layout


def _family_negative_loglik(
    object: GAMLSSModel,
    flat_coef: np.ndarray,
    layout: list[tuple[str, int]],
) -> float:
    """Compute negative log-likelihood for numerical Hessian."""
    family = object.family
    offset = 0
    parameter_values: dict[str, np.ndarray] = {}
    for parameter, size in layout:
        beta = flat_coef[offset : offset + size]
        x = np.asarray(object.design_matrices[parameter], dtype=np.float64)
        eta = x @ beta
        parameter_values[parameter] = np.asarray(
            family.link_inverses[parameter](jnp.asarray(eta, dtype=jnp.float64)),
            dtype=np.float64,
        )
        offset += size

    deviance_kwargs: dict[str, Any] = {"y": np.asarray(object.y, dtype=np.float64)}
    deviance_kwargs.update(parameter_values)
    weights = np.asarray(object.weights, dtype=np.float64) if object.weights is not None else np.ones(object.n, dtype=np.float64)
    deviance = np.asarray(family.g_dev_inc(**deviance_kwargs), dtype=np.float64)
    return float(np.sum(0.5 * weights * deviance))


def _numerical_hessian(
    flat_coef: np.ndarray,
    objective: callable,
) -> np.ndarray:
    """Compute numerical Hessian using central differences."""
    n = flat_coef.size
    hessian = np.zeros((n, n), dtype=np.float64)
    if n == 0:
        return hessian
    eps = np.cbrt(np.finfo(np.float64).eps)
    steps = np.maximum(np.abs(flat_coef) * eps, 1e-4)
    f0 = objective(flat_coef)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(n):
            ei = np.zeros(n, dtype=np.float64)
            ei[i] = steps[i]
            f_plus = objective(flat_coef + ei)
            f_minus = objective(flat_coef - ei)
            hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)
            for j in range(i + 1, n):
                ej = np.zeros(n, dtype=np.float64)
                ej[j] = steps[j]
                f_pp = objective(flat_coef + ei + ej)
                f_pm = objective(flat_coef + ei - ej)
                f_mp = objective(flat_coef - ei + ej)
                f_mm = objective(flat_coef - ei - ej)
                value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
                hessian[i, j] = value
                hessian[j, i] = value
    return hessian


def _stabilize_covariance_from_hessian(hessian: np.ndarray) -> np.ndarray:
    """Convert a numerical Hessian into a finite staged covariance matrix."""

    hessian_array = np.asarray(hessian, dtype=np.float64)
    if hessian_array.size == 0:
        return hessian_array.reshape(0, 0)
    safe_hessian = np.nan_to_num(hessian_array, nan=0.0, posinf=1e12, neginf=-1e12)
    ridge = np.eye(safe_hessian.shape[0], dtype=np.float64) * 1e-6
    try:
        cov = np.linalg.inv(safe_hessian + ridge)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(safe_hessian + ridge, rcond=1e-10)
    if np.isfinite(cov).all():
        return cov.astype(np.float64, copy=False)

    diagonal = np.abs(np.diag(safe_hessian))
    diagonal = np.where(np.isfinite(diagonal), diagonal, 0.0)
    diagonal = np.maximum(diagonal, 1e-8)
    return np.diag(1.0 / diagonal).astype(np.float64, copy=False)


def vcov(
    object: GAMLSSModel,
    type: str = "vcov",
) -> Any:
    """R reference: `gamlss/R/vcov-gamlss.R::vcov.gamlss`.

    Staged behavior:
    - Uses a stored covariance matrix when available.
    - Falls back to diagonal `NA`/`nan` values matching coefficient count.
    """

    _require_gamlss_method(object)
    valid_types = {"vcov", "cor", "se", "coef", "all"}
    if type not in valid_types:
        raise ValueError(f"type must be one of {sorted(valid_types)}")

    flat_coef, layout = _flatten_coefficients(object)
    stored = object.additional_slots.get("vcov")
    if stored is None:
        if flat_coef.size and getattr(object.family, "link_inverses", None):
            hessian = _numerical_hessian(
                flat_coef,
                lambda beta: _family_negative_loglik(object, beta, layout),
            )
            cov = _stabilize_covariance_from_hessian(hessian)
        else:
            cov = np.full((flat_coef.size, flat_coef.size), np.nan, dtype=np.float64)
    else:
        cov = np.asarray(stored, dtype=np.float64)

    if cov.size:
        diag = np.diag(cov)
        diag = np.where(np.isnan(diag), np.nan, np.maximum(diag, 0.0))
        se = np.sqrt(diag)
    else:
        se = np.array([], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cor = cov / np.outer(se, se) if cov.size else cov

    if type == "vcov":
        return cov
    if type == "cor":
        return cor
    if type == "se":
        return se
    if type == "coef":
        return flat_coef
    return {"coef": flat_coef, "se": se, "vcov": cov, "cor": cor}


__all__ = [
    "vcov",
]