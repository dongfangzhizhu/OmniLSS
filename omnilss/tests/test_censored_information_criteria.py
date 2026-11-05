"""logLik / AIC / GAIC / IC for censored (Surv) models.

Censored fits set g_dev = -2 * total censored log-likelihood, so the standard
criteria are exact: logLik = -g_dev/2, AIC = g_dev + 2*df, GAIC = g_dev + k*df.
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
    return model


def test_loglik_value() -> None:
    model = _fit()
    result = omnilss.logLik(model)
    assert result.value == pytest.approx(-model.g_dev / 2.0, rel=1e-12)
    assert result.nobs == model.n
    assert result.df == pytest.approx(model.df_fit)


def test_aic_equals_deviance_plus_two_df() -> None:
    model = _fit()
    assert omnilss.AIC(model) == pytest.approx(model.g_dev + 2.0 * model.df_fit, rel=1e-12)
    assert omnilss.GAIC(model) == pytest.approx(omnilss.AIC(model), rel=1e-12)
    assert omnilss.IC(model) == pytest.approx(omnilss.AIC(model), rel=1e-12)


def test_gaic_penalty() -> None:
    model = _fit()
    for k in (2.0, 3.84, np.log(model.n)):
        assert omnilss.GAIC(model, k=k) == pytest.approx(
            model.g_dev + k * model.df_fit, rel=1e-12
        )
        assert omnilss.IC(model, k=k) == pytest.approx(
            model.g_dev + k * model.df_fit, rel=1e-12
        )


def test_model_methods_match_functions() -> None:
    model = _fit()
    assert model.logLik().value == pytest.approx(omnilss.logLik(model).value, rel=1e-12)
    assert model.AIC() == pytest.approx(omnilss.AIC(model), rel=1e-12)
    assert model.GAIC(3.84) == pytest.approx(omnilss.GAIC(model, k=3.84), rel=1e-12)
    assert model.IC() == pytest.approx(omnilss.IC(model), rel=1e-12)


def test_criteria_are_exported() -> None:
    for name in ("AIC", "GAIC", "IC", "logLik", "extractAIC"):
        assert name in omnilss.__all__


def test_aic_matches_independent_mle() -> None:
    # The censored MLE deviance for this sample is 446.977476 (scipy); AIC adds 2*3.
    model = _fit()
    assert model.g_dev == pytest.approx(446.977476, rel=1e-6)
    assert omnilss.AIC(model) == pytest.approx(446.977476 + 2.0 * 3.0, rel=1e-6)
