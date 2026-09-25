"""R-compatible RS-to-CG mixed fitting contracts."""

import numpy as np
import pytest

from omnilss import gamlss
from omnilss.algorithms.cg_algorithm_v2 import cg_fit_v2
from omnilss.algorithms.mixed_algorithm import mixed_fit
from omnilss.algorithms.rs_algorithm import rs_fit


def _normal_data():
    rng = np.random.default_rng(20260924)
    x = np.linspace(-1.0, 1.0, 80)
    sigma = np.exp(-0.4 + 0.2 * x)
    y = 1.0 + 1.5 * x + rng.normal(scale=sigma)
    weights = np.linspace(0.5, 1.5, len(x))
    return {"y": y, "x": x}, weights


def test_mixed_matches_explicit_rs_then_cg_pipeline():
    data, weights = _normal_data()
    rs_model = rs_fit(
        "y ~ x",
        family="NO",
        data=data,
        sigma_formula="~ x",
        weights=weights,
        max_iter=1,
    )
    expected = cg_fit_v2(
        "y ~ x",
        family=rs_model.family,
        data=data,
        sigma_formula="~ x",
        weights=weights,
        max_iter=4,
        initial_model=rs_model,
    )

    actual = mixed_fit(
        "y ~ x",
        family="NO",
        data=data,
        sigma_formula="~ x",
        weights=weights,
        n1=1,
        n2=4,
    )

    for parameter in actual.family.estimable_parameters:
        np.testing.assert_allclose(
            actual.coefficients[parameter], expected.coefficients[parameter]
        )
        np.testing.assert_allclose(
            actual.fitted_values[parameter], expected.fitted_values[parameter]
        )
    assert actual.g_dev == pytest.approx(expected.g_dev)
    np.testing.assert_allclose(actual.weights, weights)


def test_zero_cg_cycles_preserve_complete_rs_boundary_state():
    data, weights = _normal_data()
    rs_model = rs_fit(
        "y ~ x",
        family="NO",
        data=data,
        sigma_formula="~ x",
        weights=weights,
        max_iter=2,
    )
    mixed_model = mixed_fit(
        "y ~ x",
        family="NO",
        data=data,
        sigma_formula="~ x",
        weights=weights,
        n1=2,
        n2=0,
    )

    for parameter in rs_model.family.estimable_parameters:
        np.testing.assert_allclose(
            mixed_model.coefficients[parameter], rs_model.coefficients[parameter]
        )
        np.testing.assert_allclose(
            mixed_model.linear_predictors[parameter],
            rs_model.linear_predictors[parameter],
        )
        np.testing.assert_allclose(
            mixed_model.fitted_values[parameter], rs_model.fitted_values[parameter]
        )
    assert mixed_model.g_dev == pytest.approx(rs_model.g_dev)
    assert mixed_model.iter == 0
    assert mixed_model.additional_slots["mixed_rs_iterations"] == 2
    assert mixed_model.additional_slots["mixed_cg_iterations"] == 0
    assert mixed_model.additional_slots["mixed_total_iterations"] == 2
    assert mixed_model.additional_slots["mixed_boundary_deviance"] == pytest.approx(
        rs_model.g_dev
    )
    assert mixed_model.additional_slots["cg_initial_model_used"] is True


def test_public_mixed_exposes_stage_controls_and_metadata():
    data, _ = _normal_data()
    model = gamlss(
        "y ~ x",
        family="NO",
        data=data,
        method="MIXED",
        mixed_n1=2,
        mixed_n2=3,
    )

    assert model.additional_slots["method"] == "MIXED"
    assert model.additional_slots["mixed_rs_n1"] == 2
    assert model.additional_slots["mixed_cg_n2"] == 3
    assert model.additional_slots["mixed_cg_backend"] == "CG_IRLS_CROSS"
    assert model.additional_slots["mixed_state_parameters"] == ("mu", "sigma")
    assert model.call["method"] == "MIXED"
    assert model.call["mixed_n1"] == 2
    assert model.call["mixed_n2"] == 3


@pytest.mark.parametrize("name,value", [("n1", -1), ("n2", -1), ("n1", True), ("n2", 1.5)])
def test_mixed_rejects_invalid_stage_cycles(name, value):
    data, _ = _normal_data()
    kwargs = {"n1": 1, "n2": 1, name: value}
    with pytest.raises(ValueError, match=rf"{name} must"):
        mixed_fit("y ~ x", family="NO", data=data, **kwargs)


def test_public_hybrid_retains_historical_auto_dispatch():
    data, _ = _normal_data()
    model = gamlss("y ~ x", family="NO", data=data, method="HYBRID", max_iter=2)

    assert model.additional_slots["method"] == "HYBRID"
    assert model.additional_slots["hybrid_algorithm_used"] == "rs"
    assert model.additional_slots["hybrid_auto_selected"] is True
