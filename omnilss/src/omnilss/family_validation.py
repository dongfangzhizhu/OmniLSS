"""Validation engine wiring FamilyDefinition to metadata constraints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .families import FamilyDefinition
from .family_schema import (
    FamilyMetadata,
    ValidationIssue,
    ensure_family_parameters_valid,
    finite_constraint,
    positive_constraint,
    validate_family_parameters,
)


def default_metadata_for_family(family: FamilyDefinition) -> FamilyMetadata:
    constraints = {}
    for param in family.parameters:
        constraints[param] = finite_constraint(param)
        if param in {"sigma", "tau"}:
            constraints[param] = positive_constraint(param)
    return FamilyMetadata(
        support=family.type,
        parameter_constraints=constraints,
        default_links=dict(family.links or {}),
        tail_behavior="unknown",
        moments={},
    )


def validate_family_runtime_inputs(
    family: FamilyDefinition,
    parameter_values: Mapping[str, np.ndarray | float],
) -> list[ValidationIssue]:
    metadata = default_metadata_for_family(family)
    return validate_family_parameters(metadata, parameter_values)


def ensure_valid_likelihood_inputs(
    family: FamilyDefinition,
    parameter_values: Mapping[str, np.ndarray | float],
) -> None:
    metadata = default_metadata_for_family(family)
    ensure_family_parameters_valid(metadata, parameter_values)
