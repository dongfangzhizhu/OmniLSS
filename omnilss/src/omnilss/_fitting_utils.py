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
    # D2.134: factor(g) is handled by the design builder (categorical
    # expansion); listing it here only stops the variable collector from
    # demanding a data column literally named "factor".
    "factor": lambda x: x,
    "cos": np.cos,
    "exp": np.exp,
    "I": lambda x: x,
    "log": np.log,
    "sin": np.sin,
    "sqrt": np.sqrt,
}
_MAX_FORMULA_AST_DEPTH = 16


class FormulaEvaluationError(ValueError):
    """Structured formula expression evaluation failure."""

    def __init__(self, term: str, reason: str) -> None:
        self.term = term
        self.reason = reason
        super().__init__(f"Unable to evaluate term {term!r}: {reason}")


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
    except FormulaEvaluationError:
        raise
    except Exception as exc:
        raise FormulaEvaluationError(expression, str(exc)) from exc

    if arr.ndim == 0:
        arr = np.full(n, float(arr), dtype=np.float64)
    if len(arr) != n:
        raise FormulaEvaluationError(
            expression, f"wrong length {len(arr)}; expected {n}"
        )
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


def _formula_term_names(term: str) -> list[str]:
    """Extract data variable names referenced by a single formula term.

    Bare identifiers and expressions such as ``log(x)`` or the interaction
    ``x1:x2`` are decomposed into the underlying variable names.  Numeric
    constants and allowlisted function names are not treated as variables.
    Unparseable terms return no names so that the downstream term evaluator
    reports the original error.
    """

    text = str(term).strip()
    if not text:
        return []
    if ":" in text and "(" not in text:
        names: list[str] = []
        for part in text.split(":"):
            names.extend(_formula_term_names(part))
        return names
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FORMULA_FUNCTIONS:
            names.append(node.id)
    return names


def _collect_formula_variables(formula_text: str, response: str) -> list[str]:
    """Collect every data variable referenced by one (possibly partial) formula."""

    parsed = parse_full_formula(_normalize_parameter_formula(response, formula_text))
    names: list[str] = []
    for term in parsed.linear_terms:
        names.extend(_formula_term_names(term.variable))
    for term in parsed.smooth_terms:
        names.extend(_formula_term_names(term.variable))
    for term in parsed.tensor_terms:
        for variable in term.variables:
            names.extend(_formula_term_names(variable))
    return names


def _validate_fitting_inputs(
    formula: str,
    data: Mapping[str, Any],
    weights: Any | None = None,
    *,
    extra_formulas: Mapping[str, str] | None = None,
) -> np.ndarray | None:
    """Validate the shared data contract for the fitting entry points.

    Enforces one consistent contract for :func:`omnilss.fitting.gamlss` and
    :func:`omnilss.fitting.gamlss_ml` before any algorithm-specific routing:

    - the response must exist in ``data`` and be a non-empty, one-dimensional,
      numeric, finite array;
    - every variable referenced by the mu formula plus the extra (sigma or
      parameter) formulas must exist in ``data``, be one-dimensional, match
      the response length, and be finite when numeric;
    - ``weights`` (when provided) must be numeric, one-dimensional, match the
      response length, be finite and non-negative, and contain at least one
      positive entry.

    Returns the validated weights converted to ``float64``, or ``None`` when
    no weights were supplied.  Raises ``ValueError`` on the first contract
    violation with a message naming the offending variable and constraint.
    """

    parsed = parse_full_formula(str(formula))
    response = parsed.response

    if response not in data:
        raise ValueError(f"data is missing response variable {response!r}")
    y = np.asarray(data[response])
    if y.ndim != 1:
        raise ValueError(
            f"response variable {response!r} must be one-dimensional, got shape {y.shape}"
        )
    if y.shape[0] == 0:
        raise ValueError(
            f"response variable {response!r} is empty; "
            "at least one observation is required"
        )
    try:
        y_numeric = y.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"response variable {response!r} must be numeric, got dtype {y.dtype}"
        ) from exc
    if not np.all(np.isfinite(y_numeric)):
        n_bad = int(np.count_nonzero(~np.isfinite(y_numeric)))
        raise ValueError(
            f"response variable {response!r} contains {n_bad} non-finite value(s) "
            "(NaN or Inf); remove or impute them before fitting"
        )
    n = int(y.shape[0])

    formula_specs: list[tuple[str, str]] = [("formula", str(formula))]
    if extra_formulas:
        formula_specs.extend(
            (f"{label} formula", str(text))
            for label, text in extra_formulas.items()
            if text is not None
        )

    checked: set[str] = set()
    for label, text in formula_specs:
        for name in _collect_formula_variables(text, response):
            if name in checked or name == response:
                continue
            checked.add(name)
            if name not in data:
                raise ValueError(
                    f"{label} {text!r} references variable {name!r} "
                    "that is missing from data"
                )
            values = np.asarray(data[name])
            if values.ndim != 1:
                raise ValueError(
                    f"variable {name!r} referenced by {label} must be "
                    f"one-dimensional, got shape {values.shape}"
                )
            if values.shape[0] != n:
                raise ValueError(
                    f"variable {name!r} has length {values.shape[0]}, but response "
                    f"{response!r} has length {n}"
                )
            if values.dtype.kind in "fiu":
                try:
                    finite = np.isfinite(values.astype(np.float64, copy=False))
                except (TypeError, ValueError):
                    continue
                if not np.all(finite):
                    n_bad = int(np.count_nonzero(~finite))
                    raise ValueError(
                        f"variable {name!r} referenced by {label} contains "
                        f"{n_bad} non-finite value(s) (NaN or Inf)"
                    )

    if weights is None:
        return None
    try:
        w = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weights must be numeric: {exc}") from exc
    if w.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {w.shape}")
    if w.shape[0] != n:
        raise ValueError(
            f"weights has length {w.shape[0]}, but response "
            f"{response!r} has length {n}"
        )
    if not np.all(np.isfinite(w)):
        n_bad = int(np.count_nonzero(~np.isfinite(w)))
        raise ValueError(
            f"weights contains {n_bad} non-finite value(s) (NaN or Inf)"
        )
    if np.any(w < 0):
        raise ValueError(
            "weights must be non-negative; "
            f"found {int(np.count_nonzero(w < 0))} negative value(s)"
        )
    if not np.any(w > 0):
        raise ValueError("weights must contain at least one positive value")
    return w


__all__ = [
    "FormulaEvaluationError",
    "_apply_method_step",
    "_eval_linear_term",
    "_is_intercept_only_formula",
    "_normalize_parameter_formula",
    "_resolve_parameter_formulas",
    "_validate_fitting_inputs",
    "_weighted_least_squares",
]
