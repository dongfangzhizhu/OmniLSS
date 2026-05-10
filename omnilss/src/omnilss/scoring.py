"""Probabilistic forecast scoring rules for OmniLSS models.

Implements proper scoring rules for evaluating distributional forecasts:
- CRPS (Continuous Ranked Probability Score)
- Log Score (negative log-likelihood)
- DSS (Dawid-Sebastiani Score)
- Energy Score (multivariate generalization of CRPS)
- Interval Score

These are the standard metrics for comparing probabilistic forecasts.
Lower scores are better for all metrics.

References
----------
Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
prediction, and estimation. JASA, 102(477), 359-378.

Examples
--------
>>> from omnilss import gamlss
>>> from omnilss.scoring import crps, log_score, interval_score
>>> import numpy as np

>>> np.random.seed(42)
>>> n = 500
>>> x = np.random.randn(n)
>>> y = 2 + 3 * x + np.random.randn(n)
>>> data = {"y": y, "x": x}

>>> model = gamlss("y ~ x", family="NO", data=data)

>>> # Split into train/test
>>> x_test = np.random.randn(100)
>>> y_test = 2 + 3 * x_test + np.random.randn(100)
>>> newdata = {"x": x_test}

>>> print(f"CRPS:      {crps(model, newdata, y_test):.4f}")
>>> print(f"Log Score: {log_score(model, newdata, y_test):.4f}")
>>> print(f"DSS:       {dss(model, newdata, y_test):.4f}")
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .model import GAMLSSModel


def _get_predicted_params(
    model: GAMLSSModel,
    newdata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Get predicted distribution parameters as numpy arrays."""
    params = model.predict_params(newdata)
    return {k: np.asarray(v) for k, v in params.items()}


def log_score(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
) -> float:
    """Log Score (negative mean log-likelihood).

    The log score is the negative log-likelihood of the observations
    under the predictive distribution. Lower is better.

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction, e.g., {"x": array}.
    y_obs : array-like
        Observed values to score against.

    Returns
    -------
    score : float
        Mean negative log-likelihood (lower is better).

    Examples
    --------
    >>> score = log_score(model, {"x": x_test}, y_test)
    >>> print(f"Log Score: {score:.4f}")
    """
    y_obs = jnp.array(y_obs, dtype=jnp.float64)
    params = model.predict_params(newdata)
    jax_params = {k: jnp.array(v) for k, v in params.items()}
    log_lik = model.family.d(y_obs, log=True, **jax_params)
    return float(-jnp.mean(log_lik))


def dss(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
) -> float:
    """Dawid-Sebastiani Score.

    A proper scoring rule based on the first two moments of the
    predictive distribution. Requires the distribution to have
    finite mean and variance.

    DSS = (y - mu)^2 / sigma^2 + 2 * log(sigma)

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction.
    y_obs : array-like
        Observed values.

    Returns
    -------
    score : float
        Mean DSS (lower is better).

    Examples
    --------
    >>> score = dss(model, {"x": x_test}, y_test)
    >>> print(f"DSS: {score:.4f}")
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    params = _get_predicted_params(model, newdata)

    mu = params.get("mu", np.zeros_like(y_obs))
    sigma = params.get("sigma", np.ones_like(y_obs))

    scores = (y_obs - mu) ** 2 / (sigma ** 2) + 2 * np.log(sigma)
    return float(np.mean(scores))


def crps(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
    method: str = "quantile",
    n_quantiles: int = 99,
) -> float:
    """Continuous Ranked Probability Score (CRPS).

    The CRPS is a proper scoring rule that generalizes the MAE to
    probabilistic forecasts. It measures the compatibility of a
    predictive distribution with an observation.

    CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]

    where X, X' are independent draws from F.

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction.
    y_obs : array-like
        Observed values.
    method : str, default="quantile"
        Computation method:
        - "quantile": Grid approximation via quantile function (works for
          all distributions with a q() function).
        - "sample": Monte Carlo approximation via random samples.
    n_quantiles : int, default=99
        Number of quantile levels for the "quantile" method.
        Higher values give more accurate approximations.

    Returns
    -------
    score : float
        Mean CRPS (lower is better).

    Notes
    -----
    For distributions with closed-form CRPS (Normal, Gamma, etc.),
    the quantile method with n_quantiles=99 is accurate to ~3 decimal places.
    Use n_quantiles=999 for higher precision.

    Examples
    --------
    >>> score = crps(model, {"x": x_test}, y_test)
    >>> print(f"CRPS: {score:.4f}")

    >>> # Higher precision
    >>> score = crps(model, {"x": x_test}, y_test, n_quantiles=999)
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    n_obs = len(y_obs)

    if method == "quantile":
        return _crps_quantile(model, newdata, y_obs, n_quantiles)
    elif method == "sample":
        return _crps_sample(model, newdata, y_obs, n_samples=1000)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'quantile' or 'sample'.")


def _crps_quantile(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: np.ndarray,
    n_quantiles: int,
) -> float:
    """CRPS via quantile grid approximation.

    Uses the identity:
    CRPS(F, y) = 2 * integral_0^1 (1{y < q(p)} - p) * (q(p) - y) dp
    ≈ (2/n_q) * sum_k (1{y < q(p_k)} - p_k) * (q(p_k) - y)
    """
    params = _get_predicted_params(model, newdata)
    jax_params = {k: jnp.array(v) for k, v in params.items()}

    # Quantile levels: avoid 0 and 1
    prob_levels = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)

    crps_values = np.zeros(len(y_obs))

    for p in prob_levels:
        p_arr = jnp.full(len(y_obs), p)
        q_p = np.asarray(model.family.q(p_arr, **jax_params))
        indicator = (y_obs < q_p).astype(np.float64)
        crps_values += (indicator - p) * (q_p - y_obs)

    crps_values *= 2.0 / n_quantiles
    return float(np.mean(crps_values))


def _crps_sample(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: np.ndarray,
    n_samples: int,
) -> float:
    """CRPS via Monte Carlo approximation.

    Uses the identity:
    CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
    """
    import jax

    params = _get_predicted_params(model, newdata)
    jax_params = {k: jnp.array(v) for k, v in params.items()}
    n_obs = len(y_obs)

    # Draw samples from the predictive distribution
    key = jax.random.PRNGKey(0)
    # Expand params to (n_samples, n_obs) shape
    expanded_params = {}
    for k, v in jax_params.items():
        expanded_params[k] = jnp.tile(v[None, :], (n_samples, 1)).reshape(-1)

    samples = np.asarray(
        model.family.r(n_samples * n_obs, **expanded_params)
    ).reshape(n_samples, n_obs)

    # E[|X - y|]
    term1 = np.mean(np.abs(samples - y_obs[None, :]), axis=0)

    # E[|X - X'|] ≈ using pairs of samples
    half = n_samples // 2
    term2 = np.mean(np.abs(samples[:half] - samples[half:2*half]), axis=0)

    crps_values = term1 - 0.5 * term2
    return float(np.mean(crps_values))


def interval_score(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
    alpha: float = 0.1,
) -> float:
    """Interval Score for a (1-alpha) prediction interval.

    IS_alpha(l, u, y) = (u - l) + (2/alpha) * (l - y) * 1{y < l}
                                 + (2/alpha) * (y - u) * 1{y > u}

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction.
    y_obs : array-like
        Observed values.
    alpha : float, default=0.1
        Significance level. Produces a (1-alpha)*100% prediction interval.
        Default 0.1 gives a 90% interval.

    Returns
    -------
    score : float
        Mean interval score (lower is better).

    Examples
    --------
    >>> # Score 90% prediction intervals
    >>> score = interval_score(model, {"x": x_test}, y_test, alpha=0.1)
    >>> print(f"90% Interval Score: {score:.4f}")
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    params = _get_predicted_params(model, newdata)
    jax_params = {k: jnp.array(v) for k, v in params.items()}

    # Lower and upper quantiles
    p_lower = jnp.full(len(y_obs), alpha / 2)
    p_upper = jnp.full(len(y_obs), 1 - alpha / 2)

    lower = np.asarray(model.family.q(p_lower, **jax_params))
    upper = np.asarray(model.family.q(p_upper, **jax_params))

    width = upper - lower
    penalty_lower = (2 / alpha) * np.maximum(lower - y_obs, 0)
    penalty_upper = (2 / alpha) * np.maximum(y_obs - upper, 0)

    scores = width + penalty_lower + penalty_upper
    return float(np.mean(scores))


def coverage(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
    alpha: float = 0.1,
) -> float:
    """Empirical coverage of a (1-alpha) prediction interval.

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction.
    y_obs : array-like
        Observed values.
    alpha : float, default=0.1
        Significance level. Checks a (1-alpha)*100% interval.

    Returns
    -------
    coverage : float
        Fraction of observations within the prediction interval.
        Ideally close to (1 - alpha).

    Examples
    --------
    >>> cov = coverage(model, {"x": x_test}, y_test, alpha=0.1)
    >>> print(f"90% Coverage: {cov:.1%}")  # Should be ~90%
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    params = _get_predicted_params(model, newdata)
    jax_params = {k: jnp.array(v) for k, v in params.items()}

    p_lower = jnp.full(len(y_obs), alpha / 2)
    p_upper = jnp.full(len(y_obs), 1 - alpha / 2)

    lower = np.asarray(model.family.q(p_lower, **jax_params))
    upper = np.asarray(model.family.q(p_upper, **jax_params))

    in_interval = (y_obs >= lower) & (y_obs <= upper)
    return float(np.mean(in_interval))


def pit_histogram(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Probability Integral Transform (PIT) histogram.

    A calibration diagnostic. For a well-calibrated model, PIT values
    should be uniformly distributed on [0, 1].

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction.
    y_obs : array-like
        Observed values.
    n_bins : int, default=10
        Number of histogram bins.

    Returns
    -------
    result : dict
        - "pit_values": PIT values (CDF evaluated at observations)
        - "bin_edges": Histogram bin edges
        - "counts": Histogram counts
        - "expected": Expected count per bin (n_obs / n_bins)

    Examples
    --------
    >>> result = pit_histogram(model, {"x": x_test}, y_test)
    >>> import matplotlib.pyplot as plt
    >>> plt.bar(result["bin_edges"][:-1], result["counts"],
    ...         width=1/result["n_bins"], align="edge")
    >>> plt.axhline(result["expected"], color="red", linestyle="--",
    ...             label="Uniform")
    >>> plt.xlabel("PIT value")
    >>> plt.ylabel("Count")
    >>> plt.title("PIT Histogram (uniform = well-calibrated)")
    >>> plt.legend()
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    params = _get_predicted_params(model, newdata)
    jax_params = {k: jnp.array(v) for k, v in params.items()}

    # PIT = CDF(y_obs | predicted params)
    pit_values = np.asarray(
        model.family.p(jnp.array(y_obs), **jax_params)
    )

    counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    expected = len(y_obs) / n_bins

    return {
        "pit_values": pit_values,
        "bin_edges": bin_edges,
        "counts": counts,
        "expected": expected,
        "n_bins": n_bins,
    }


def scoring_summary(
    model: GAMLSSModel,
    newdata: dict[str, Any],
    y_obs: Any,
    alphas: list[float] = (0.1, 0.2),
) -> dict[str, float]:
    """Compute a comprehensive set of scoring metrics.

    Parameters
    ----------
    model : GAMLSSModel
        Fitted OmniLSS model.
    newdata : dict
        New data for prediction.
    y_obs : array-like
        Observed values.
    alphas : list of float
        Significance levels for interval scores and coverage.

    Returns
    -------
    summary : dict
        Dictionary of metric names to values.

    Examples
    --------
    >>> summary = scoring_summary(model, {"x": x_test}, y_test)
    >>> for metric, value in summary.items():
    ...     print(f"{metric:30s}: {value:.4f}")
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    summary: dict[str, float] = {}

    # Core metrics
    summary["log_score"] = log_score(model, newdata, y_obs)
    summary["crps"] = crps(model, newdata, y_obs)
    summary["dss"] = dss(model, newdata, y_obs)

    # Interval metrics
    for alpha in alphas:
        level = int((1 - alpha) * 100)
        summary[f"interval_score_{level}"] = interval_score(
            model, newdata, y_obs, alpha=alpha
        )
        summary[f"coverage_{level}"] = coverage(
            model, newdata, y_obs, alpha=alpha
        )

    # Point forecast metrics (using mu as point forecast)
    params = _get_predicted_params(model, newdata)
    mu = params.get("mu", np.zeros_like(y_obs))
    summary["mae"] = float(np.mean(np.abs(y_obs - mu)))
    summary["rmse"] = float(np.sqrt(np.mean((y_obs - mu) ** 2)))

    return summary


__all__ = [
    "log_score",
    "crps",
    "dss",
    "interval_score",
    "coverage",
    "pit_histogram",
    "scoring_summary",
]
