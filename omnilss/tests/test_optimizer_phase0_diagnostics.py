import pytest

from omnilss.fitting import gamlss

pytest.importorskip("optax")


def test_joint_optimizer_exposes_phase0_diag_slots():
    data = {"y": [1.0, 2.0, 1.5, 2.2, 1.8], "x": [0.0, 1.0, 0.5, 1.2, 0.8]}
    model = gamlss("y ~ x", family="NO", data=data, method="joint")
    slots = model.additional_slots
    assert "gradient_norm" in slots
    assert "step_size" in slots
    assert "condition_number" in slots
