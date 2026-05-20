"""Family metadata and validation schema for runtime safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterConstraint:
    name: str
    check: Callable[[np.ndarray], np.ndarray]
    message: str


@dataclass(frozen=True)
class FamilyMetadata:
    support: str
    parameter_constraints: Mapping[str, ParameterConstraint]
    default_links: Mapping[str, str]
    tail_behavior: str
    moments: Mapping[str, str]


@dataclass(frozen=True)
class ValidationIssue:
    parameter: str
    message: str


class FamilyValidationError(ValueError):
    pass


def _as_array(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def positive_constraint(name: str) -> ParameterConstraint:
    return ParameterConstraint(
        name=name,
        check=lambda x: _as_array(x) > 0.0,
        message=f"{name} must be > 0",
    )


def finite_constraint(name: str) -> ParameterConstraint:
    return ParameterConstraint(
        name=name,
        check=lambda x: np.isfinite(_as_array(x)),
        message=f"{name} must be finite",
    )


def range_constraint(name: str, lower: float, upper: float) -> ParameterConstraint:
    return ParameterConstraint(
        name=name,
        check=lambda x: (_as_array(x) >= lower) & (_as_array(x) <= upper),
        message=f"{name} must be in [{lower}, {upper}]",
    )


def validate_family_parameters(
    metadata: FamilyMetadata,
    params: Mapping[str, np.ndarray | float],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, constraint in metadata.parameter_constraints.items():
        if name not in params:
            continue
        ok_mask = constraint.check(_as_array(params[name]))
        if not np.all(ok_mask):
            issues.append(ValidationIssue(parameter=name, message=constraint.message))
    return issues


def ensure_family_parameters_valid(
    metadata: FamilyMetadata,
    params: Mapping[str, np.ndarray | float],
) -> None:
    issues = validate_family_parameters(metadata, params)
    if issues:
        joined = "; ".join(f"{i.parameter}: {i.message}" for i in issues)
        raise FamilyValidationError(joined)
