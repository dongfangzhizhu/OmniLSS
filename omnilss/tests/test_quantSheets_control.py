from __future__ import annotations

import pytest

from omnilss.quantSheets_control import quantSheets_control


def test_quantSheets_control_defaults_and_override() -> None:
    c0 = quantSheets_control()
    assert c0["n_intervals"] == 10
    c1 = quantSheets_control(n_intervals=20, degree=2, lambda_=0.5, max_iter=30, tol=1e-5)
    assert c1["n_intervals"] == 20
    assert c1["degree"] == 2
    assert c1["lambda_"] == 0.5


def test_quantSheets_control_validation() -> None:
    with pytest.raises(ValueError):
        quantSheets_control(n_intervals=0)
