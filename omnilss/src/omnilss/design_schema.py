"""Design-matrix schema helpers for reproducible prediction artifacts.

The schema intentionally stores only auditable metadata needed to rebuild a
prediction design matrix from formulas. Large training matrices remain outside
JSON metadata, while prediction still has enough information to reject silent
schema drift.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from .formula_parser import parse_formula

SCHEMA_VERSION = 2
ARTIFACT_VERSION = 2


def _split_formula_terms(rhs: str) -> list[str]:
    """Split a formula RHS on top-level plus signs."""

    terms: list[str] = []
    current: list[str] = []
    depth = 0
    for char in rhs:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == "+" and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []
        else:
            current.append(char)
    term = "".join(current).strip()
    if term:
        terms.append(term)
    return terms


def _formula_rhs(formula: str) -> str:
    if "~" not in formula:
        return ""
    return formula.split("~", 1)[1].strip()


def _normalized_term_order(formula: str, fallback_terms: list[str]) -> list[str]:
    """Return parsed term order, preserving transform expressions when possible."""

    rhs = _formula_rhs(formula)
    if not rhs:
        return list(fallback_terms)
    try:
        parsed = parse_formula(formula)
        ordered = [term.variable for term in parsed.linear_terms]
        ordered.extend(
            f"{term.smoother}({term.variable})" for term in parsed.smooth_terms
        )
        ordered.extend(
            f"{term.smoother}({', '.join(term.variables)})"
            for term in parsed.tensor_terms
        )
        if ordered:
            return ordered
    except Exception:
        pass

    terms = [term for term in _split_formula_terms(rhs) if term not in {"1", "0", "-1"}]
    return terms or list(fallback_terms)


def _expression_ast(term: str) -> str | None:
    """Return a stable AST representation for transform terms."""

    if not any(ch in term for ch in "()+-*/"):
        return None
    try:
        return ast.dump(ast.parse(term, mode="eval"), include_attributes=False)
    except SyntaxError:
        return None


def _column_names(
    term_labels: list[str], has_intercept: bool, n_columns: int | None
) -> list[str]:
    names: list[str] = []
    if has_intercept:
        names.append("(Intercept)")
    names.extend(term_labels)
    if n_columns is not None and len(names) != n_columns:
        names = [f"x{i}" for i in range(n_columns)]
    return names


def _checksum(values: list[str]) -> str:
    payload = "\x1f".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_data(model: Any) -> Mapping[str, Any]:
    call = getattr(model, "call", {}) or {}
    data = call.get("data", {}) if isinstance(call, Mapping) else {}
    return data if isinstance(data, Mapping) else {}


def _factor_levels(
    term_order: list[str], data: Mapping[str, Any]
) -> dict[str, list[Any]]:
    levels: dict[str, list[Any]] = {}
    for term in term_order:
        if not term.startswith("factor(") or not term.endswith(")"):
            continue
        variable = term[len("factor(") : -1].strip()
        if variable not in data:
            continue
        unique = np.unique(np.asarray(data[variable]))
        levels[variable] = [
            item.item() if hasattr(item, "item") else item for item in unique
        ]
    return levels


def _smooth_basis_metadata(
    parameter: str, smooth_infos: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return compact smooth basis metadata for a schema parameter."""

    info = smooth_infos.get(parameter) if isinstance(smooth_infos, Mapping) else None
    smooth_fits = getattr(info, "smooth_fits", None)
    if smooth_fits is None and isinstance(info, list):
        smooth_fits = info
    if not smooth_fits:
        return []

    entries: list[dict[str, Any]] = []
    for smooth in smooth_fits:
        if isinstance(smooth, Mapping):
            entries.append(dict(smooth))
            continue
        knots = getattr(smooth, "knots", None)
        entries.append(
            {
                "term_index": int(getattr(smooth, "term_index", -1)),
                "variable": str(getattr(smooth, "variable", "")),
                "smoother": str(getattr(smooth, "smoother", "")),
                "basis_smoother": str(
                    getattr(smooth, "basis_smoother", None)
                    or getattr(smooth, "smoother", "")
                ),
                "basis_columns": list(getattr(smooth, "basis_columns", (0, 0))),
                "knots": (
                    np.asarray(knots, dtype=np.float64).tolist()
                    if knots is not None and np.asarray(knots).size > 0
                    else None
                ),
                "degree": getattr(smooth, "degree", None),
                "order": getattr(smooth, "order", None),
            }
        )
    return entries


def build_design_matrix_schema(model: Any) -> dict[str, Any]:
    """Build a serializable schema for every model parameter design matrix."""

    schema: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "parameters": {},
    }
    formulas = dict(getattr(model, "formulas", {}) or {})
    terms = dict(getattr(model, "terms", {}) or {})
    design_matrices = dict(getattr(model, "design_matrices", {}) or {})
    coefficients = dict(getattr(model, "coefficients", {}) or {})
    data = _training_data(model)
    smooth_infos = dict(getattr(model, "additional_slots", {}) or {}).get(
        "smooth_infos", {}
    )

    for parameter in getattr(model, "parameters", ()):
        term_info = terms.get(parameter, {}) or {}
        formula = str(formulas.get(parameter, term_info.get("formula", "")))
        term_formula = str(term_info.get("formula", ""))
        if _formula_rhs(formula) == "." and term_formula:
            # Prefer expanded term metadata over an unevaluated dot formula for
            # portable prediction artifacts.  A literal ``.`` depends on the
            # original training frame and cannot be rebuilt safely after
            # serialization without recording the expanded term order.
            formula = term_formula
        raw_labels = list(term_info.get("term_labels", []) or [])
        has_intercept = bool(term_info.get("intercept", True))
        design = design_matrices.get(parameter)
        n_columns = None
        if design is not None:
            arr = np.asarray(design)
            if arr.ndim == 2:
                n_columns = int(arr.shape[1])
        coefficient_count = None
        coef = coefficients.get(parameter)
        if coef is not None:
            coefficient_count = int(np.asarray(coef).size)
        term_order = _normalized_term_order(formula, raw_labels)
        column_names = _column_names(raw_labels, has_intercept, n_columns)
        schema["parameters"][parameter] = {
            "parameter": parameter,
            "formula": formula,
            "raw_formula": formula,
            "term_order": term_order,
            "term_labels": raw_labels,
            "column_names": column_names,
            "has_intercept": has_intercept,
            "factor_levels": _factor_levels(term_order, data),
            "numeric_transform_ast": {
                term: ast_repr
                for term in term_order
                if (ast_repr := _expression_ast(term)) is not None
            },
            "smooth_metadata_required": any(
                "(" in term
                and term.split("(", 1)[0] in {"pb", "ps", "cs", "s", "lo", "te", "ti"}
                for term in term_order
            ),
            "smooth_basis_metadata": _smooth_basis_metadata(parameter, smooth_infos),
            "n_columns": n_columns,
            "coefficient_count": coefficient_count,
            "training_column_checksum": _checksum(column_names),
        }
    return schema


def ensure_model_design_schema(model: Any) -> dict[str, Any]:
    """Return the model schema, materializing it in additional_slots if absent."""

    slots = dict(getattr(model, "additional_slots", {}) or {})
    existing = slots.get("design_matrix_schema")
    if isinstance(existing, Mapping) and existing.get("parameters"):
        parameters = existing.get("parameters", {}) or {}
        needs_smooth_enrichment = any(
            param_schema.get("smooth_metadata_required")
            and "smooth_basis_metadata" not in param_schema
            for param_schema in parameters.values()
            if isinstance(param_schema, Mapping)
        )
        if not needs_smooth_enrichment:
            return dict(existing)
    schema = build_design_matrix_schema(model)
    try:
        model.additional_slots = {**slots, "design_matrix_schema": schema}
    except Exception:
        pass
    return schema
