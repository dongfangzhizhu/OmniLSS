from __future__ import annotations

import numpy as np
import pytest

from omnilss.runtime.errors import ConvergenceError, NumericalError
from omnilss.validation.hardening import (
    check_numerical_state,
    enforce_iteration_budget,
    enforce_no_nan_propagation,
)


def test_stress_check_detects_nan_and_inf_states():
    result = check_numerical_state({"a": np.array([1.0, np.nan]), "b": np.array([np.inf])})
    assert result.has_nan is True
    assert result.has_inf is True


def test_enforce_no_nan_propagation_raises_numerical_error():
    with pytest.raises(NumericalError):
        enforce_no_nan_propagation({"x": np.array([0.0, np.nan])})


def test_enforce_iteration_budget_prevents_infinite_loop_pattern():
    with pytest.raises(ConvergenceError):
        enforce_iteration_budget(iteration=200, max_iter=200)

    enforce_iteration_budget(iteration=199, max_iter=200)
