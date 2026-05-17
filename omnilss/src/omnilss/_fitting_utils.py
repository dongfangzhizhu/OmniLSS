"""Internal utility helpers extracted from :mod:`omnilss.fitting`."""

from __future__ import annotations

from collections.abc import Mapping
import ast
import math
from statistics import NormalDist
from typing import Any

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition
from .formula_parser import parse_formula as parse_full_formula

_STANDARD_NORMAL = NormalDist()


_ALLOWED_FORMULA_FUNCTIONS = {
    "abs": np.abs,
    "cos": np.cos,
    "exp": np.exp,
    "I": lambda x: x,
    "log": np.log,
    "sin": np.sin,
    "sqrt": np.sqrt,
}
_MAX_FORMULA_AST_DEPTH = 16
_BANNED_FORMULA_NODES = (
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
)


def _validate_formula_ast(node: ast.AST, *, depth: int = 0) -> None:
    """Reject unsafe or overly complex expression syntax before evaluation."""

    if depth > _MAX_FORMULA_AST_DEPTH:
        raise ValueError("formula expression is too deeply nested")
    if isinstance(node, _BANNED_FORMULA_NODES):
        raise ValueError(f"unsupported expression node '{type(node).__name__}'")
    for child in ast.iter_child_nodes(node):
        _validate_formula_ast(child, depth=depth + 1)


def _safe_eval_formula_expression(
    expression: str,
    data: Mapping[str, Any],
    n: int,
) -> np.ndarray:
    """Evaluate a numeric formula expression with a strict AST whitelist."""

    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.Name):
            if node.id in arrays:
                value = arrays[node.id]
                if len(value) != n:
                    raise ValueError(f"Variable '{node.id}' has wrong length")
                return value
            raise ValueError(f"unknown variable or function '{node.id}'")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("unsupported binary operator")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            else:
                raise ValueError(
                    "only direct calls to allowlisted functions are allowed"
                )
            if func_name not in _ALLOWED_FORMULA_FUNCTIONS:
                raise ValueError(f"function '{func_name}' is not allowed")
            if node.keywords:
                raise ValueError(
                    "keyword arguments are not allowed in formula expressions"
                )
            args = [_eval(arg) for arg in node.args]
            return _ALLOWED_FORMULA_FUNCTIONS[func_name](*args)
        raise ValueError(f"unsupported expression node '{type(node).__name__}'")

    try:
        parsed = ast.parse(expression, mode="eval")
        _validate_formula_ast(parsed)
        value = _eval(parsed)
        arr = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Unable to evaluate term '{expression}': {exc}") from exc

    if arr.ndim == 0:
        arr = np.full(n, float(arr), dtype=np.float64)
    if len(arr) != n:
        raise ValueError(f"Term '{expression}' has wrong length")
    return arr


def _eval_linear_term(term: str, data: Mapping[str, Any], n: int) -> np.ndarray:
    if term in data:
        arr = np.asarray(data[term], dtype=np.float64)
        if len(arr) != n:
            raise ValueError(f"Variable '{term}' has wrong length")
        return arr

    if ":" in term and "(" not in term:
        parts = [p.strip() for p in term.split(":") if p.strip()]
        if len(parts) == 2:
            return _eval_linear_term(parts[0], data, n) * _eval_linear_term(
                parts[1], data, n
            )

    return _safe_eval_formula_expression(term, data, n)


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
            raise ValueError(
                f"family {family.name!r} uses fixed parameter {key!r}; provide it in data instead"
            )
        if key not in {"mu", "sigma", "nu", "tau"}:
            raise ValueError(f"unknown parameter formula {parameter!r}")
        if key not in family.parameters:
            raise ValueError(
                f"family {family.name!r} does not support parameter {key!r}"
            )
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


def _apply_method_step(
    previous_beta: np.ndarray, proposed_beta: np.ndarray, method_name: str
) -> np.ndarray:
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
