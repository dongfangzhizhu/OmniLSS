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
    assert result.design_matrix.shape == (4, 3)  # intercept + x1 + x2
    assert result.column_labels == ("x1", "x2")
    assert result.has_smooth_terms is False
