import numpy as np

from omnilss.fitting import gamlss


def test_rs_additional_slots_expose_phase0_diagnostics():
    rng = np.random.default_rng(42)
    n = 64
    x = rng.normal(size=n)
    y = 1.5 + 0.8 * x + rng.normal(scale=0.5, size=n)
    model = gamlss("y ~ x", family="NO", data={"x": x, "y": y}, method="RS")

    slots = model.additional_slots
    assert "rs_step_halving_by_param" in slots
    assert "rs_last_condition_number_by_param" in slots

    sh = slots["rs_step_halving_by_param"]
    cond = slots["rs_last_condition_number_by_param"]
    assert isinstance(sh, dict)
    assert isinstance(cond, dict)
    assert "mu" in sh
    assert "mu" in cond
    assert np.isfinite(float(cond["mu"])) or np.isnan(float(cond["mu"]))

    assert "gradient_norm" in slots
    assert "condition_number" in slots
    assert "step_size_by_param" in slots
    assert np.isfinite(float(slots["gradient_norm"])) or np.isnan(float(slots["gradient_norm"]))
    assert np.isfinite(float(slots["condition_number"])) or np.isnan(float(slots["condition_number"]))
