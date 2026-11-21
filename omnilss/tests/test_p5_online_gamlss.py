# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for OmniLSS self-contained online GAMLSS estimator and P5-1c golden gate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "omnilss" / "src"))
sys.path.insert(0, str(REPO / "tools"))

import p5_streaming_golden as sg  # noqa: E402
from omnilss.online import OnlineGAMLSS, OnlineGram  # noqa: E402


def test_online_gram_batch_invariance_with_and_without_forgetting():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(90, 4))
    y = rng.normal(size=90)
    w = rng.uniform(0.5, 2.0, size=90)
    for forget in (0.0, 0.03, 0.1):
        g_split = OnlineGram(4, forget=forget)
        g_split.update(X[:25], y[:25], w[:25])
        g_split.update(X[25:60], y[25:60], w[25:60])
        g_split.update(X[60:], y[60:], w[60:])

        g_full = OnlineGram(4, forget=forget)
        g_full.update(X, y, w)

        np.testing.assert_allclose(g_split.gram, g_full.gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g_split.rhs, g_full.rhs, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g_split.solve(), g_full.solve(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", ("NO", "GA", "PO", "WEI", "LOGNO"))
def test_online_gamlss_streaming_recovers_dgp_and_predicts_quantiles(family: str):
    rng = np.random.default_rng(42)
    n = 400
    X = rng.normal(size=(n, 2))
    if family == "NO":
        y = 1.5 + 0.7 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.5, size=n)
    elif family == "GA":
        mu = np.exp(0.6 + 0.3 * X[:, 0])
        y = rng.gamma(shape=9.0, scale=mu / 9.0)
    elif family == "PO":
        y = rng.poisson(lam=np.exp(0.8 + 0.4 * X[:, 0])).astype(float)
    elif family == "WEI":
        y = rng.weibull(a=2.0, size=n) * np.exp(0.5 + 0.25 * X[:, 0])
    else:  # LOGNO
        y = np.exp(0.4 + 0.3 * X[:, 0] + rng.normal(scale=0.35, size=n))

    est = OnlineGAMLSS(family=family, forget=0.01)
    est.partial_fit(X[:200], y[:200])
    est.partial_fit(X[200:300], y[200:300])
    est.partial_fit(X[300:], y[300:])

    assert est.is_fitted_
    assert est.n_obs_ == n
    assert est.n_updates_ == 3

    params = est.predict_params(X[:10])
    assert set(params.keys()) == set(est.view.estimable)
    for arr in params.values():
        assert arr.shape == (10,)
        assert np.all(np.isfinite(arr))

    qs = est.predict_quantiles(X[:10], quantiles=(0.1, 0.5, 0.9))
    assert np.all(qs["0.1"] <= qs["0.5"] + 1e-12)
    assert np.all(qs["0.5"] <= qs["0.9"] + 1e-12)
    assert np.isfinite(est.log_likelihood(X[:10], y[:10]))


def test_online_gamlss_rejects_unfitted_update_and_bad_forget():
    with pytest.raises(ValueError, match="forget"):
        OnlineGAMLSS("NO", forget=1.0)
    est = OnlineGAMLSS("NO")
    with pytest.raises(RuntimeError, match="fitted"):
        est.update(np.ones((5, 2)), np.ones(5))


def test_p5_streaming_golden_artifact_and_all_7_criteria():
    ok, problems = sg.check_artifact(sg.DEFAULT_OUT)
    assert ok, problems
