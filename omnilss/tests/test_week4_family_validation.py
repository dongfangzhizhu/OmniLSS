from __future__ import annotations

import numpy as np
import pytest

from omnilss.families import FamilyDefinition
from omnilss.family_schema import FamilyMetadata, finite_constraint, positive_constraint, validate_family_parameters, FamilyValidationError, ensure_family_parameters_valid
from omnilss.family_validation import ensure_valid_likelihood_inputs


def _dummy_dev(**kwargs):
    return np.ones(3)


def test_family_schema_constraints_detect_invalid_sigma():
    metadata = FamilyMetadata(
        support="Continuous",
        parameter_constraints={"sigma": positive_constraint("sigma")},
        default_links={"sigma": "log"},
        tail_behavior="light",
        moments={"mean": "finite"},
    )
    issues = validate_family_parameters(metadata, {"sigma": np.array([-1.0, 1.0])})
    assert len(issues) == 1


def test_validation_engine_raises_for_non_positive_sigma():
    family = FamilyDefinition(
        name="TEST",
        parameters=("mu", "sigma"),
        g_dev_inc=_dummy_dev,
        score_functions={"mu": lambda **_: np.ones(1), "sigma": lambda **_: np.ones(1)},
        hessian_functions={"mu": lambda **_: -np.ones(1), "sigma": lambda **_: -np.ones(1)},
    )
    with pytest.raises(FamilyValidationError):
        ensure_valid_likelihood_inputs(family, {"mu": np.array([0.0]), "sigma": np.array([0.0])})


def test_ensure_family_parameters_valid_passes_for_finite_positive_sigma():
    metadata = FamilyMetadata(
        support="Continuous",
        parameter_constraints={
            "mu": finite_constraint("mu"),
            "sigma": positive_constraint("sigma"),
        },
        default_links={"mu": "identity", "sigma": "log"},
        tail_behavior="medium",
        moments={},
    )
    ensure_family_parameters_valid(metadata, {"mu": np.array([0.0, 1.0]), "sigma": np.array([0.1, 2.0])})
