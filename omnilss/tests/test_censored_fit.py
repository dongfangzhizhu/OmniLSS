"""End-to-end censored (Surv) fitting through gamlss().

The censored RS path is validated against an independent scipy maximum-likelihood
fit of the same censored normal likelihood.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

import jax

jax.config.update("jax_enable_x64", True)

from omnilss.families import FamilyDefinition  # noqa: E402
from omnilss.fitting import gamlss  # noqa: E402
from omnilss.gamlss_cens import CensoredFamily, cens  # noqa: E402


def _right_censored_data(seed: int = 1, n: int = 300, slope: float = 0.3):
    np.random.seed(seed)
    x = np.linspace(-1.0, 1.0, n)
    true_time = np.abs(np.random.normal(slope * x, 1.0, n))
    censor = np.random.exponential(2.0, n)
    time = np.minimum(true_time, censor)
    status = (true_time <= censor).astype(float)
    return {"time": time, "status": status, "x": x}


def _scipy_mle(data, slope: bool):
    time = data["time"]
    status = data["status"]
    x = data["x"]

    def negll(theta):
        if slope:
            a, b, lsig = theta
            mu = a + b * x
        else:
            a, lsig = theta
            mu = np.full_like(time, a)
        sig = np.exp(lsig)
        ll = np.where(status == 1, norm.logpdf(time, mu, sig), norm.logsf(time, mu, sig))
        return -np.sum(ll)

    start = [0.7, 0.3, 0.0] if slope else [0.7, 0.0]
    res = minimize(
        negll,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000},
    )
    return res.x, 2.0 * res.fun


def test_censored_family_is_a_family_definition() -> None:
    family = cens("NO", "right")
    assert isinstance(family, FamilyDefinition)
    assert isinstance(family, CensoredFamily)
    assert family.name == "NOrc"
    assert family.censoring_type == "right"


def test_intercept_only_fit_matches_scipy_mle() -> None:
    data = _right_censored_data()
    m = gamlss("Surv(time,status) ~ 1", family=cens("NO", "right"), data=data, max_iter=200)
    theta, dev = _scipy_mle(data, slope=False)
    assert m.converged
    assert m.additional_slots["method"] == "RS_censored"
    assert m.g_dev == pytest.approx(dev, rel=1e-6)
    assert float(m.fitted_values["mu"][0]) == pytest.approx(theta[0], rel=1e-4)
    assert float(m.fitted_values["sigma"][0]) == pytest.approx(np.exp(theta[1]), rel=1e-4)


def test_regression_fit_matches_scipy_mle() -> None:
    data = _right_censored_data()
    m = gamlss("Surv(time,status) ~ x", family=cens("NO", "right"), data=data, max_iter=200)
    theta, dev = _scipy_mle(data, slope=True)
    mu_coef = np.asarray(m.coefficients["mu"], dtype=float)
    sigma_coef = np.asarray(m.coefficients["sigma"], dtype=float)
    assert np.allclose(mu_coef, theta[:2], rtol=1e-3, atol=1e-4)
    # sigma intercept is on the log link.
    assert sigma_coef[0] == pytest.approx(theta[2], rel=1e-3)
    assert m.g_dev == pytest.approx(dev, rel=1e-5)


def test_left_censored_smoke() -> None:
    data = _right_censored_data(seed=2)
    m = gamlss("Surv(time,status) ~ 1", family=cens("NO", "left"), data=data, max_iter=100)
    assert np.isfinite(m.g_dev)
    assert m.fitted_values["sigma"][0] > 0


def test_interval_censored_smoke() -> None:
    np.random.seed(3)
    n = 150
    t1 = np.abs(np.random.normal(0.5, 1.0, n))
    t2 = t1 + np.abs(np.random.normal(0.5, 0.2, n))
    status = np.random.choice([0.0, 1.0, 2.0, 3.0], size=n)
    data = {"time": t1, "time2": t2, "status": status}
    m = gamlss("Surv(time,time2,status) ~ 1", family=cens("NO", "interval"), data=data, max_iter=50)
    assert np.isfinite(m.g_dev)


def test_surv_requires_censored_family() -> None:
    data = _right_censored_data()
    with pytest.raises(ValueError, match="censored family"):
        gamlss("Surv(time,status) ~ 1", family="NO", data=data)


def test_censored_supports_rs_only() -> None:
    data = _right_censored_data()
    with pytest.raises(ValueError, match="method='RS' only"):
        gamlss("Surv(time,status) ~ 1", family=cens("NO", "right"), data=data, method="CG")
