"""Contract tests for the unified fitting data validation.

``gamlss()`` and ``gamlss_ml()`` must reject malformed inputs with clear,
consistent ``ValueError`` messages before any algorithm-specific routing so
that broadcasting bugs, silent NaN corruption, and per-path drift cannot reach
the fitting loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnilss import gamlss, gamlss_ml
from omnilss.fitting import _validate_fitting_inputs


def _data(n: int = 30) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x}


ENTRY_POINTS = [gamlss, gamlss_ml]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_valid_data_passes(entry):
    model = entry("y ~ x", family="NO", data=_data())
    mu = np.asarray(model.fitted_values["mu"], dtype=np.float64)
    assert mu.shape == (30,)
    assert np.all(np.isfinite(mu))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_missing_response_rejected(entry):
    with pytest.raises(ValueError, match="missing response variable 'y'"):
        entry("y ~ x", family="NO", data={"x": np.arange(5.0)})


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_empty_response_rejected(entry):
    with pytest.raises(ValueError, match="'y' is empty"):
        entry("y ~ x", family="NO", data={"y": np.array([]), "x": np.array([])})


def test_two_dimensional_response_rejected():
    with pytest.raises(ValueError, match="response variable 'y' must be one-dimensional"):
        gamlss("y ~ x", family="NO", data={"y": np.ones((4, 1)), "x": np.ones(4)})


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_non_finite_response_rejected(entry):
    data = _data()
    data["y"][3] = np.nan
    with pytest.raises(ValueError, match="response variable 'y' contains 1 non-finite"):
        entry("y ~ x", family="NO", data=data)


def test_non_numeric_response_rejected():
    data = _data()
    data["y"] = np.array(["a"] * 30, dtype=object)
    with pytest.raises(ValueError, match="response variable 'y' must be numeric"):
        gamlss("y ~ x", family="NO", data=data)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_missing_predictor_rejected(entry):
    with pytest.raises(ValueError, match="references variable 'z' that is missing"):
        entry("y ~ x + z", family="NO", data=_data())


def test_predictor_length_mismatch_rejected():
    data = _data()
    data["x"] = data["x"][:-1]
    with pytest.raises(ValueError, match="variable 'x' has length 29"):
        gamlss("y ~ x", family="NO", data=data)


def test_non_finite_predictor_rejected():
    data = _data()
    data["x"][2] = np.inf
    with pytest.raises(ValueError, match="variable 'x'"):
        gamlss("y ~ x", family="NO", data=data)


def test_expression_term_variables_checked():
    with pytest.raises(ValueError, match="variable 'z'"):
        gamlss("y ~ log(z)", family="NO", data=_data())


def test_interaction_term_variables_checked():
    with pytest.raises(ValueError, match="variable 'z'"):
        gamlss("y ~ x:z", family="NO", data=_data())


def test_sigma_formula_variable_checked():
    with pytest.raises(ValueError, match="sigma formula"):
        gamlss("y ~ x", sigma_formula="~ z", family="NO", data=_data())


def test_parameter_formula_variable_checked():
    with pytest.raises(ValueError, match="nu formula"):
        gamlss(
            "y ~ x",
            family="GG",
            data=_data(),
            parameter_formulas={"nu": "~ z"},
        )


def test_weights_wrong_length_rejected():
    with pytest.raises(ValueError, match="weights has length 5"):
        gamlss("y ~ x", family="NO", data=_data(), weights=np.ones(5))


def test_weights_two_dimensional_rejected():
    with pytest.raises(ValueError, match="weights must be one-dimensional"):
        gamlss("y ~ x", family="NO", data=_data(), weights=np.ones((30, 2)))


def test_non_finite_weights_rejected():
    w = np.ones(30)
    w[0] = np.nan
    with pytest.raises(ValueError, match="weights contains 1 non-finite"):
        gamlss("y ~ x", family="NO", data=_data(), weights=w)


def test_negative_weights_rejected():
    w = np.ones(30)
    w[0] = -1.0
    with pytest.raises(ValueError, match="weights must be non-negative"):
        gamlss("y ~ x", family="NO", data=_data(), weights=w)


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError, match="at least one positive value"):
        gamlss("y ~ x", family="NO", data=_data(), weights=np.zeros(30))


def test_non_numeric_weights_rejected():
    with pytest.raises(ValueError, match="weights must be numeric"):
        gamlss("y ~ x", family="NO", data=_data(), weights=["a"] * 30)


def test_zero_weight_entries_allowed():
    data = _data()
    w = np.ones(30)
    w[0] = 0.0
    model = gamlss("y ~ x", family="NO", data=data, weights=w)
    mu = np.asarray(model.fitted_values["mu"], dtype=np.float64)
    assert np.all(np.isfinite(mu))


def test_validator_normalizes_weights_to_float64():
    data = _data()
    w = _validate_fitting_inputs("y ~ x", data, [1, 2, 3] + [1] * 27)
    assert w is not None
    assert w.dtype == np.float64
    assert w.shape == (30,)
    assert _validate_fitting_inputs("y ~ x", data, None) is None


def test_validator_accepts_partial_extra_formulas():
    data = _data()
    w = _validate_fitting_inputs(
        "y ~ x",
        data,
        None,
        extra_formulas={"sigma": "~ x"},
    )
    assert w is None
    with pytest.raises(ValueError, match="sigma formula"):
        _validate_fitting_inputs(
            "y ~ x",
            data,
            None,
            extra_formulas={"sigma": "~ missing_col"},
        )
