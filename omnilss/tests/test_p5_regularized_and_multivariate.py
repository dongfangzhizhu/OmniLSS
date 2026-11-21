# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for P5-2 online regularized paths and JAX multivariate MVN/MVT families."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from omnilss.multivariate import (
    MVN,
    MVT,
    CholeskyCovariance,
    LowRankCovariance,
    ModifiedCholeskyPrecision,
)
from omnilss.online import OnlineRegularizedPath, online_coordinate_descent


# ---------------------------------------------------------------------------
# Online regularized coordinate descent (LASSO / Ridge / Elastic-Net)
# ---------------------------------------------------------------------------


def test_coordinate_descent_ridge_matches_closed_form():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(120), rng.normal(size=(120, 5))])
    y = 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.3, size=120)
    G = X.T @ X
    h = X.T @ y
    l2 = 4.5
    beta_cd = online_coordinate_descent(G, h, l1_alpha=0.0, l2_alpha=l2, unpenalized_first=True)
    P = np.diag([0.0] + [l2] * 5)
    beta_exact = np.linalg.solve(G + P, h)
    np.testing.assert_allclose(beta_cd, beta_exact, rtol=1e-12, atol=1e-12)


def test_coordinate_descent_lasso_satisfies_kkt_and_orthogonal_soft_threshold():
    # 1. Exact closed-form soft-thresholding on orthonormal design
    p = 4
    G = np.eye(p)
    h = np.array([2.5, 1.2, -0.4, 0.15])
    l1 = 0.5
    beta_orth = online_coordinate_descent(G, h, l1_alpha=l1, l2_alpha=0.0, unpenalized_first=False)
    expected = np.sign(h) * np.maximum(np.abs(h) - l1, 0.0)
    np.testing.assert_allclose(beta_orth, expected, rtol=1e-12, atol=1e-12)

    # 2. General correlated design KKT optimality conditions for Elastic-Net
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(150), rng.normal(size=(150, 6))])
    X[:, 2] = 0.7 * X[:, 1] + 0.3 * rng.normal(size=150)
    y = 1.0 + 2.0 * X[:, 1] - 1.5 * X[:, 4] + rng.normal(scale=0.4, size=150)
    G = X.T @ X
    h = X.T @ y
    l1_a, l2_a = 18.0, 5.0
    beta = online_coordinate_descent(G, h, l1_alpha=l1_a, l2_alpha=l2_a, unpenalized_first=True)
    grad = G @ beta - h
    # Unpenalized intercept: grad[0] == 0
    assert abs(grad[0]) <= 1e-8
    for j in range(1, beta.size):
        g_l2 = grad[j] + l2_a * beta[j]
        if abs(beta[j]) > 1e-10:
            assert abs(g_l2 + l1_a * np.sign(beta[j])) <= 1e-7
        else:
            assert abs(g_l2) <= l1_a + 1e-7


def test_online_regularized_path_monotonicity_selection_and_batch_invariance():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 8))
    true_beta = np.array([1.5, 2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = true_beta[0] + X @ true_beta[1:] + rng.normal(scale=0.35, size=n)

    p_split = OnlineRegularizedPath(8, forget=0.01, l1_ratio=0.8, n_alphas=15)
    p_split.partial_fit(X[:100], y[:100])
    p_split.partial_fit(X[100:220], y[100:220])
    p_split.partial_fit(X[220:], y[220:])

    p_full = OnlineRegularizedPath(8, forget=0.01, l1_ratio=0.8, n_alphas=15)
    p_full.partial_fit(X, y)

    res_split = p_split.path(criterion="bic")
    res_full = p_full.path(criterion="bic")

    # Streaming batch invariance
    np.testing.assert_allclose(res_split.coefs, res_full.coefs, rtol=1e-10, atol=1e-10)
    # L1 norm increases as alpha decreases (alphas sorted descending)
    l1_norms = np.sum(np.abs(res_full.coefs[:, 1:]), axis=1)
    assert l1_norms[-1] > l1_norms[0]
    assert l1_norms[0] <= 1e-6
    # BIC recovers active support (true nonzero coords 1, 2)
    best = res_full.best_coef
    assert abs(best[1] - 2.0) < 0.25
    assert abs(best[2] + 1.5) < 0.25


# ---------------------------------------------------------------------------
# Multivariate MVN / MVT across all 3 covariance / precision parameterizations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", ("cholesky", "modified_cholesky", "low_rank"))
def test_mvn_logpdf_score_hessian_match_scipy_and_analytic(param: str):
    rng = np.random.default_rng(99)
    d = 4
    mvn = MVN(d, parameterization=param, rank=2)
    mu = jnp.array([0.5, -1.0, 2.0, 0.2], dtype=jnp.float64)
    cov_eta = jnp.asarray(rng.normal(scale=0.3, size=mvn.n_cov_params), dtype=jnp.float64)
    Y = jnp.asarray(rng.normal(size=(12, d)), dtype=jnp.float64)

    cov_np = np.asarray(mvn.covariance(cov_eta))
    # Strictly symmetric positive definite
    np.testing.assert_allclose(cov_np, cov_np.T, atol=1e-13)
    eigvals = np.linalg.eigvalsh(cov_np)
    assert np.all(eigvals > 0.0)

    # Exact logpdf match against scipy.stats.multivariate_normal
    py_ll = np.asarray(mvn.logpdf(Y, mu, cov_eta))
    sp_ll = stats.multivariate_normal.logpdf(np.asarray(Y), mean=np.asarray(mu), cov=cov_np)
    np.testing.assert_allclose(py_ll, sp_ll, rtol=1e-11, atol=1e-11)

    # Exact analytical mu score: sum_i Sigma^{-1} (y_i - mu)
    scores = mvn.score(Y, mu, cov_eta)
    prec_np = np.linalg.inv(cov_np)
    analytic_mu_score = np.sum((np.asarray(Y) - np.asarray(mu)) @ prec_np.T, axis=0)
    np.testing.assert_allclose(np.asarray(scores["mu"]), analytic_mu_score, rtol=1e-10, atol=1e-10)

    # Finite-difference check on cov_eta score
    eps = 1e-6
    num_grad = np.zeros(mvn.n_cov_params)
    eta_np = np.asarray(cov_eta)
    for k in range(mvn.n_cov_params):
        e_p = eta_np.copy(); e_p[k] += eps
        e_m = eta_np.copy(); e_m[k] -= eps
        num_grad[k] = (
            float(jnp.sum(mvn.logpdf(Y, mu, jnp.asarray(e_p))))
            - float(jnp.sum(mvn.logpdf(Y, mu, jnp.asarray(e_m))))
        ) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(scores["cov_eta"]), num_grad, rtol=1e-6, atol=1e-7)

    # Hessian symmetry + exact mu-mu block == -n * Sigma^{-1}
    H = np.asarray(mvn.hessian(Y, mu, cov_eta))
    np.testing.assert_allclose(H, H.T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(H[:d, :d], -Y.shape[0] * prec_np, rtol=1e-10, atol=1e-10)

    # Sampling moments
    samples = np.asarray(mvn.sample(jax.random.PRNGKey(0), 6000, mu, cov_eta))
    assert samples.shape == (6000, d)
    np.testing.assert_allclose(np.mean(samples, axis=0), np.asarray(mu), atol=0.06)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), cov_np, atol=0.12)


@pytest.mark.parametrize("param", ("cholesky", "modified_cholesky", "low_rank"))
def test_mvt_logpdf_score_hessian_match_scipy_and_finite_diff(param: str):
    rng = np.random.default_rng(123)
    d = 3
    mvt = MVT(d, parameterization=param, rank=2)
    mu = jnp.array([1.0, -0.5, 0.3], dtype=jnp.float64)
    cov_eta = jnp.asarray(rng.normal(scale=0.25, size=mvt.n_cov_params), dtype=jnp.float64)
    df = 5.5
    log_df = jnp.asarray(np.log(df), dtype=jnp.float64)
    Y = jnp.asarray(rng.normal(size=(10, d)), dtype=jnp.float64)

    scale_np = np.asarray(mvt.covariance(cov_eta))
    py_ll = np.asarray(mvt.logpdf(Y, mu, cov_eta, log_df))
    sp_ll = stats.multivariate_t.logpdf(np.asarray(Y), loc=np.asarray(mu), shape=scale_np, df=df)
    np.testing.assert_allclose(py_ll, sp_ll, rtol=1e-11, atol=1e-11)

    # Exact score check against central finite differences
    scores = mvt.score(Y, mu, cov_eta, log_df)
    eps = 1e-6
    ll_plus = float(jnp.sum(mvt.logpdf(Y, mu, cov_eta, log_df + eps)))
    ll_minus = float(jnp.sum(mvt.logpdf(Y, mu, cov_eta, log_df - eps)))
    assert np.isclose(float(scores["log_df"]), (ll_plus - ll_minus) / (2.0 * eps), rtol=1e-6, atol=1e-7)

    H = np.asarray(mvt.hessian(Y, mu, cov_eta, log_df))
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(H, H.T, rtol=1e-9, atol=1e-9)

    samples = np.asarray(mvt.sample(jax.random.PRNGKey(1), 5000, mu, cov_eta, log_df))
    assert samples.shape == (5000, d)
    np.testing.assert_allclose(np.mean(samples, axis=0), np.asarray(mu), atol=0.10)
    expected_cov = np.asarray(mvt.covariance(cov_eta, df=df))
    np.testing.assert_allclose(np.cov(samples, rowvar=False), expected_cov, atol=0.25)


def test_matrix_links_precision_and_covariance_are_inverses():
    rng = np.random.default_rng(77)
    for link in (
        CholeskyCovariance(4),
        ModifiedCholeskyPrecision(4),
        LowRankCovariance(4, rank=2),
    ):
        eta = jnp.asarray(rng.normal(scale=0.3, size=link.n_params), dtype=jnp.float64)
        cov = np.asarray(link.covariance(eta))
        prec = np.asarray(link.precision(eta))
        np.testing.assert_allclose(cov @ prec, np.eye(4), atol=1e-12)
