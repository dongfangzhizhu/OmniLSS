"""Prediction support for censored (Surv) models.

Censored models predict distribution parameters (mu/sigma) exactly like any
GAMLSS: linkinv(X_new @ beta).  se_fit uses the per-parameter covariance stored
by the censored IRLS fit.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import omnilss  # noqa: E402
from omnilss.gamlss_cens import cens  # noqa: E402


def _fit():
    np.random.seed(1)
    x = np.linspace(-1.0, 1.0, 300)
    true_time = np.abs(np.random.normal(0.3 * x, 1.0, 300))
    censor = np.random.exponential(2.0, 300)
    time = np.minimum(true_time, censor)
    status = (true_time <= censor).astype(float)
    data = {"time": time, "status": status, "x": x}
    model = omnilss.gamlss(
        "Surv(time,status) ~ x", family=cens("NO", "right"), data=data, max_iter=200
    )
    newdata = {"x": np.array([-0.5, 0.0, 0.5])}
    return model, newdata


def test_predict_response_and_link() -> None:
    model, newdata = _fit()
    mu = np.asarray(model.predict(newdata), dtype=float)
    link = np.asarray(model.predict(newdata, type="link"), dtype=float)
    beta = np.asarray(model.coefficients["mu"], dtype=float)
    x_new = np.column_stack([np.ones(3), newdata["x"]])
    np.testing.assert_allclose(link, x_new @ beta, rtol=1e-12)
    np.testing.assert_allclose(mu, link, rtol=1e-12)  # identity mu link


def test_predict_what_sigma_and_all() -> None:
    model, newdata = _fit()
    sigma = np.asarray(model.predict(newdata, what="sigma"), dtype=float)
    expected = float(model.fitted_values["sigma"][0])
    np.testing.assert_allclose(sigma, np.full(3, expected), rtol=1e-10)
    all_params = model.predict(newdata, what="all")
    assert set(all_params) == {"mu", "sigma"}


def test_predict_se_fit_is_finite_and_positive() -> None:
    model, newdata = _fit()
    result = model.predict(newdata, se_fit=True)
    assert set(result) == {"fit", "se.fit"}
    se = np.asarray(result["se.fit"], dtype=float)
    assert se.shape == (3,)
    assert np.all(np.isfinite(se))
    assert np.all(se > 0.0)


def test_predict_terms_returns_term_matrix() -> None:
    model, newdata = _fit()
    terms = np.asarray(model.predict(newdata, type="terms"), dtype=float)
    assert terms.shape == (3, 1)
    # mu = intercept + slope * x, and term = slope * x.
    beta = np.asarray(model.coefficients["mu"], dtype=float)
    np.testing.assert_allclose(terms[:, 0], beta[1] * newdata["x"], rtol=1e-10)


def test_predict_without_newdata_returns_fitted() -> None:
    model, _ = _fit()
    fitted = np.asarray(model.predict(), dtype=float)
    assert fitted.shape == (model.n,)
    np.testing.assert_allclose(fitted, np.asarray(model.fitted_values["mu"]), rtol=1e-12)


def test_predict_params_matches_predict() -> None:
    model, newdata = _fit()
    params = model.predict_params(newdata)
    np.testing.assert_allclose(
        np.asarray(params["mu"]), np.asarray(model.predict(newdata)), rtol=1e-12
    )


def test_predict_all_se_fit() -> None:
    model, newdata = _fit()
    result = omnilss.predict_all(model, newdata, se_fit=True)
    for parameter in ("mu", "sigma"):
        entry = result.values[parameter]
        se = np.asarray(entry["se.fit"], dtype=float)
        assert se.shape == (3,)
        assert np.all(np.isfinite(se)) and np.all(se > 0.0)


def test_stored_covariance_shape() -> None:
    model, _ = _fit()
    covariance = model.additional_slots["parameter_covariance"]
    assert set(covariance) >= {"mu", "sigma"}
    assert np.asarray(covariance["mu"]).shape == (2, 2)
    assert np.asarray(covariance["sigma"]).shape == (1, 1)


def test_terms_metadata_populated() -> None:
    model, _ = _fit()
    assert model.terms["mu"]["term_labels"] == ["x"]
    assert model.terms["mu"]["response"] == "__surv__"
