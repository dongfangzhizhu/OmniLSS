from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.model import GAMLSSModel
from omnilss.par_plot import par_plot


def test_par_plot_payload_shapes() -> None:
    y = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    mu = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    sigma = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=3,
        y=y,
        fitted_values={"mu": mu, "sigma": sigma},
        linear_predictors={"mu": mu, "sigma": np.log(sigma)},
    )

    out = par_plot(model)
    assert out.y.shape == (3,)
    assert len(out.series) == 2
    assert out.series[0].parameter == "mu"
    assert out.series[0].fitted.shape == (3,)


def test_par_plot_includes_residual_payload_and_weights() -> None:
    y = np.array([1.0, 0.5, -1.5], dtype=np.float64)
    mu = np.array([0.5, 0.0, -1.0], dtype=np.float64)
    sigma = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    w = np.array([1.0, 2.0, 0.5], dtype=np.float64)

    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=3,
        y=y,
        fitted_values={"mu": mu, "sigma": sigma},
        linear_predictors={"mu": mu, "sigma": np.log(sigma)},
        weights=w,
    )

    out = par_plot(model, parameters=("mu",))
    assert out.weights.shape == (3,)
    np.testing.assert_allclose(out.weights, w)
    np.testing.assert_allclose(out.series[0].response_residual, y - mu)
    np.testing.assert_allclose(out.series[0].weighted_response_residual, w * (y - mu))


def test_par_plot_rejects_weight_shape_mismatch() -> None:
    y = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=3,
        y=y,
        fitted_values={"mu": np.zeros(3), "sigma": np.ones(3)},
        linear_predictors={"mu": np.zeros(3), "sigma": np.zeros(3)},
        weights=np.array([1.0, 2.0], dtype=np.float64),
    )
    try:
        par_plot(model)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "weights must have same shape" in str(exc)
