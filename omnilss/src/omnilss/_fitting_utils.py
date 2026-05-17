"""Internal utility helpers extracted from :mod:`omnilss.fitting`."""

from __future__ import annotations

from collections.abc import Mapping
import math
from statistics import NormalDist
from typing import Any

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition
from .formula_parser import parse_formula as parse_full_formula

_STANDARD_NORMAL = NormalDist()


def _eval_linear_term(term: str, data: Mapping[str, Any], n: int) -> np.ndarray:
    if term in data:
        arr = np.asarray(data[term], dtype=np.float64)
        if len(arr) != n:
            raise ValueError(f"Variable '{term}' has wrong length")
        return arr

    if ":" in term and "(" not in term:
        parts = [p.strip() for p in term.split(":") if p.strip()]
        if len(parts) == 2:
            return _eval_linear_term(parts[0], data, n) * _eval_linear_term(parts[1], data, n)

    safe_env = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    safe_env.update({"np": np, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "I": lambda x: x})
    try:
        arr = np.asarray(eval(term, {"__builtins__": {}}, safe_env), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Unable to evaluate term '{term}'") from exc
    if arr.ndim == 0:
        arr = np.full(n, float(arr), dtype=np.float64)
    if len(arr) != n:
        raise ValueError(f"Term '{term}' has wrong length")
    return arr


def _normalize_parameter_formula(response: str, formula: str) -> str:
    text = str(formula).strip()
    if text.startswith("~"):
        return f"{response} {text}"
    if "~" in text:
        return text
    return f"{response} {text}".strip()


def _resolve_parameter_formulas(
    response: str,
    family: FamilyDefinition,
    mu_formula: str,
    sigma_formula: str,
    parameter_formulas: Mapping[str, str] | None = None,
) -> dict[str, str]:
    fixed_parameters = set(family.fixed_parameters or ())
    resolved = {"mu": _normalize_parameter_formula(response, mu_formula)}
    for parameter in family.parameters:
        if parameter == "mu":
            continue
        if parameter in fixed_parameters:
            continue
        if parameter == "sigma":
            resolved["sigma"] = _normalize_parameter_formula(response, sigma_formula)
            continue
        resolved[parameter] = _normalize_parameter_formula(response, "~1")

    if parameter_formulas is None:
        return resolved

    for parameter, formula_text in parameter_formulas.items():
        key = str(parameter).strip().lower()
        if key in fixed_parameters:
            raise ValueError(f"family {family.name!r} uses fixed parameter {key!r}; provide it in data instead")
        if key not in {"mu", "sigma", "nu", "tau"}:
            raise ValueError(f"unknown parameter formula {parameter!r}")
        if key not in family.parameters:
            raise ValueError(f"family {family.name!r} does not support parameter {key!r}")
        resolved[key] = _normalize_parameter_formula(response, str(formula_text))
    return resolved


def _is_intercept_only_formula(formula: str) -> bool:
    parsed = parse_full_formula(formula)
    return parsed.has_intercept and not parsed.linear_terms and not parsed.smooth_terms


def _weighted_least_squares(
    x: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    smooth_info: Any = None,
) -> np.ndarray:
    x_array = np.asarray(x, dtype=np.float64)
    z_array = np.asarray(z, dtype=np.float64)
    w_array = np.asarray(w, dtype=np.float64)
    safe_w = np.nan_to_num(w_array, nan=1.0, posinf=1e10, neginf=1e-10)
    safe_w = np.clip(safe_w, 1e-10, 1e10)
    safe_x = np.nan_to_num(x_array, nan=0.0, posinf=1e10, neginf=-1e10)
    safe_z = np.nan_to_num(z_array, nan=0.0, posinf=1e10, neginf=-1e10)

    if smooth_info is not None and len(smooth_info.smooth_fits) > 0:
        from .smooth_fitting import fit_penalized_wls

        return fit_penalized_wls(safe_x, safe_z, safe_w, smooth_info.smooth_fits)

    sqrt_w = np.sqrt(safe_w)
    wx = safe_x * sqrt_w[:, None]
    wz = safe_z * sqrt_w
    try:
        coef, _, _, _ = np.linalg.lstsq(wx, wz, rcond=None)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(wx, rcond=1e-10) @ wz
    return coef.astype(np.float64, copy=False)


def _apply_method_step(previous_beta: np.ndarray, proposed_beta: np.ndarray, method_name: str) -> np.ndarray:
    if method_name == "CG":
        return previous_beta + 0.5 * (proposed_beta - previous_beta)
    if method_name == "MIXED":
        return previous_beta + 0.75 * (proposed_beta - previous_beta)
    return proposed_beta


__all__ = [
    "_apply_method_step",
    "_eval_linear_term",
    "_is_intercept_only_formula",
    "_normalize_parameter_formula",
    "_resolve_parameter_formulas",
    "_weighted_least_squares",
]
