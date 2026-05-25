from __future__ import annotations

import numpy as np

from omnilss.Formulae2data import Formulae2data


def test_formulae2data_basic_linear_formula() -> None:
    data = {
        "y": np.array([1.0, 2.0, 3.0, 4.0]),
        "x1": np.array([0.0, 1.0, 0.5, -1.0]),
        "x2": np.array([2.0, 2.0, 1.0, 0.0]),
    }

    result = Formulae2data("y ~ x1 + x2", data)

    assert result.response_name == "y"
    assert result.response.shape == (4,)
    assert result.weights.shape == (4,)
    np.testing.assert_allclose(result.weights, np.ones(4))
    assert result.design_matrix.shape == (4, 3)  # intercept + x1 + x2
    assert result.column_labels == ("x1", "x2")
    assert result.has_smooth_terms is False


def test_formulae2data_supports_weights_and_validation() -> None:
    data = {
        "y": np.array([1.0, 2.0, 3.0]),
        "x": np.array([0.0, 1.0, 2.0]),
        "weights": np.array([1.0, 0.5, 2.0]),
    }
    result = Formulae2data("y ~ x", data)
    np.testing.assert_allclose(result.weights, np.array([1.0, 0.5, 2.0]))

    bad = dict(data)
    bad["weights"] = np.array([1.0, -1.0, 2.0])
    try:
        Formulae2data("y ~ x", bad)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-negative" in str(exc)
