# SPDX-License-Identifier: GPL-3.0-or-later
"""D2.274 (A3): internal warning hygiene and r_consistency key-case parity."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from omnilss import gamlss
from omnilss.family_capabilities import _R_CONSISTENCY_EVIDENCE, capability_matrix
from omnilss.fitting import _parse_formula


@pytest.mark.parametrize("method", ["RS", "CG", "MIXED"])
def test_fitting_routes_emit_no_internal_parse_formula_deprecation(method: str) -> None:
    rng = np.random.default_rng(20261121)
    x = rng.normal(size=40)
    y = 1.5 + 0.8 * x + rng.normal(scale=0.3, size=40)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, method=method, max_iter=15)

    assert np.isfinite(model.g_dev)
    parse_deprecations = [
        str(item.message)
        for item in caught
        if issubclass(item.category, DeprecationWarning)
        and "_parse_formula" in str(item.message)
    ]
    assert parse_deprecations == []


def test_deprecated_parse_formula_shim_remains_available() -> None:
    with pytest.warns(DeprecationWarning, match="_parse_formula"):
        response, predictors = _parse_formula("y ~ x1 + x2")
    assert response == "y"
    assert predictors == ["x1", "x2"]


def test_r_consistency_evidence_and_capability_matrix_agree_on_case() -> None:
    matrix = capability_matrix()["families"]
    validated = {
        name
        for name, entry in matrix.items()
        if entry["features"]["r_consistency"] == "validated"
    }
    expected = {name.upper() for name in _R_CONSISTENCY_EVIDENCE}
    assert "JSUO" in validated
    assert validated == expected
