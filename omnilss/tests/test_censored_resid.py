"""Residual support for censored (Surv) models.

Censored models use randomized quantile residuals rqres = Phi^-1(PIT), where
PIT is the censored probability integral transform (event -> CDF, censored ->
uniform between CDF and 1 or 0 and CDF).
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import omnilss  # noqa: E402
from omnilss.gamlss_cens import cens  # noqa: E402


def _fit():
    np.random.seed(1)
    x = np.linspace(-1.0, 1.0, 300)
    true_time = np.abs(np.random.normal(0.3 * x, 1.0, 300))
    censor = np.random.exponential(2.0, 300)
    time = np.minimum(true_time, censor)
    status = (true_time <= censor).astype(float)
    model = omnilss.gamlss(
        "Surv(time,status) ~ x",
        family=cens("NO", "right"),
        data={"time": time, "status": status, "x": x},
        max_iter=200,
    )
    return model, time, status


def test_residuals_are_populated() -> None:
    model, _, _ = _fit()
    residuals = np.asarray(model.residuals, dtype=float)
    assert residuals.shape == (model.n,)
    assert np.all(np.isfinite(residuals))


def test_residuals_function_returns_z_scores() -> None:
    model, _, _ = _fit()
    got = np.asarray(omnilss.residuals(model), dtype=float)
    np.testing.assert_allclose(got, np.asarray(model.residuals, dtype=float), rtol=1e-12)


def test_randomized_quantile_residuals_are_standard_normal() -> None:
    model, _, _ = _fit()
    residuals = np.asarray(model.residuals, dtype=float)
    assert abs(float(np.mean(residuals))) < 0.2
    assert 0.7 < float(np.std(residuals)) < 1.3


def test_working_residuals_per_parameter() -> None:
    model, _, _ = _fit()
    mu_resid = np.asarray(omnilss.residuals(model, what="mu"), dtype=float)
    sigma_resid = np.asarray(omnilss.residuals(model, what="sigma"), dtype=float)
    assert mu_resid.shape == (model.n,)
    assert sigma_resid.shape == (model.n,)
    assert np.all(np.isfinite(mu_resid))


def test_quantile_residuals_diagnostic() -> None:
    model, _, _ = _fit()
    result = omnilss.quantile_residuals(model)
    assert result.n == model.n
    assert np.isfinite(result.variance) and result.variance > 0.0


def test_event_rqres_is_standardised_time() -> None:
    model, _, _ = _fit()
    y = {"time": [1.0, 2.0], "status": [1.0, 1.0]}
    mu = np.array([0.5, 0.5])
    sigma = np.array([2.0, 2.0])
    got = np.asarray(model.rqres(y=y, mu=mu, sigma=sigma), dtype=float)
    np.testing.assert_allclose(got, (np.array([1.0, 2.0]) - 0.5) / 2.0, atol=1e-9)


def test_rqres_callable_is_attached() -> None:
    model, _, _ = _fit()
    assert model.rqres is not None
