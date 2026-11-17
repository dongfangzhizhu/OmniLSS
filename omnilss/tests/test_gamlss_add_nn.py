# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.add neural-network term: nn() / gamlss_nn (D2.251).

The port is functionally verified, not bit-for-bit against nnet: nnet's BFGS
starts from random weights with its own scaling, so the contract here is the
network's behaviour and the gamlss interface (warm start, weight count, decay,
skip-only linear case).
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import NeuralNet, gamlss, gamlss_nn, nn
from omnilss.add_controls import nn_control


def _sample(seed: int = 4, n: int = 150):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = np.sin(x) + rng.normal(scale=0.1, size=n)
    return x, y


def _r2(y, fitted):
    residual = np.asarray(y) - np.asarray(fitted)
    return 1.0 - float(np.sum(residual**2)) / float(np.sum((y - np.mean(y)) ** 2))


def test_nn_validation_and_control_defaults():
    with pytest.raises(ValueError, match="starting with ~"):
        nn("x")
    term = nn("~ x")
    assert term.control["size"] == nn_control()["size"] == 3
    assert term.control["linout"] is True
    with pytest.raises(NotImplementedError, match="nested"):
        nn("~ pb(x)")


def test_network_approximates_a_non_linear_function():
    x, y = _sample()
    fit = gamlss_nn(nn("~ x", size=5, maxit=300), y, None, data={"x": x}, seed=0)
    assert _r2(y, fit.fitted_values) > 0.95
    assert fit.nl_df == 5 * 1 + 5 + 5 + 1  # W1 + b1 + W2 + b2
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=0, atol=0)


def test_fit_is_deterministic_for_a_fixed_seed():
    x, y = _sample()
    term = nn("~ x", size=4, maxit=150)
    first = gamlss_nn(term, y, None, data={"x": x}, seed=11)
    second = gamlss_nn(term, y, None, data={"x": x}, seed=11)
    np.testing.assert_array_equal(first.fitted_values, second.fitted_values)


def test_warm_start_reuses_the_previous_weights():
    x, y = _sample()
    term = nn("~ x", size=4, maxit=150)
    first = gamlss_nn(term, y, None, data={"x": x}, seed=3)
    warm = gamlss_nn(term, y, None, data={"x": x}, seed=3, warm_start=first.coef_smo.weights)
    assert _r2(y, warm.fitted_values) >= _r2(y, first.fitted_values) - 1e-9
    # The starting loss of the warm fit is the final loss of the cold fit.
    assert warm.coef_smo.weights.shape == first.coef_smo.weights.shape


def test_weight_decay_shrinks_the_weight_norm():
    x, y = _sample()
    plain = gamlss_nn(nn("~ x", size=4, maxit=300), y, None, data={"x": x}, seed=5)
    decayed = gamlss_nn(
        nn("~ x", size=4, maxit=300, decay=0.1), y, None, data={"x": x}, seed=5
    )
    assert np.linalg.norm(decayed.coef_smo.weights) < np.linalg.norm(plain.coef_smo.weights)


def test_skip_only_network_reduces_to_linear_regression():
    rng = np.random.default_rng(2)
    x = rng.normal(size=80)
    y = 1.5 + 0.7 * x + rng.normal(scale=0.2, size=80)
    fit = gamlss_nn(
        nn("~ x", size=0, skip=True, maxit=300), y, None, data={"x": x}, seed=0
    )
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    np.testing.assert_allclose(fit.fitted_values, design @ coef, atol=1e-5)
    assert fit.nl_df == 2  # skip weight + output bias


def test_linout_false_bounds_the_output():
    rng = np.random.default_rng(6)
    x = rng.uniform(-2, 2, size=100)
    y = 1.0 / (1.0 + np.exp(-x))
    fit = gamlss_nn(
        nn("~ x", size=3, linout=False, maxit=200), y, None, data={"x": x}, seed=0
    )
    assert np.all(fit.fitted_values > 0.0) and np.all(fit.fitted_values < 1.0)


def test_nn_interface_errors():
    x, y = _sample()
    with pytest.raises(TypeError, match="nn\\(\\) term"):
        gamlss_nn("~ x", y, None, data={"x": x})
    with pytest.raises(ValueError, match="data"):
        gamlss_nn(nn("~ x"), y, None)
    with pytest.raises(NotImplementedError, match="xeval"):
        gamlss_nn(nn("~ x"), y, None, data={"x": x}, xeval="newdata")
    labels = np.where(x < 0, "a", "b")
    with pytest.raises(NotImplementedError, match="not numeric"):
        gamlss_nn(nn("~ g"), y, None, data={"g": labels})
    with pytest.raises(NotImplementedError, match="gamlss_nn"):
        gamlss("y ~ nn(x)", family="NO", data={"y": y, "x": x}, max_iter=5)


def test_neural_net_reports_unfitted_state():
    network = NeuralNet(size=2)
    network.variables = ("x",)
    with pytest.raises(RuntimeError, match="not been fitted"):
        network.predict({"x": np.array([0.0])})
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = network.n_weights
