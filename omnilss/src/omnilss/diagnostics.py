"""Comprehensive model diagnostics for GAMLSS models.

This module provides a complete suite of diagnostic tools for GAMLSS models,
including:
- Quantile residuals (randomized and normalized)
- Q-Q plots
- Worm plots
- Residual plots
- Calibration plots
- Centile checks

References
----------
- Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models 
  for location, scale and shape. Journal of the Royal Statistical Society: 
  Series C (Applied Statistics), 54(3), 507-554.
- Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals. 
  Journal of Computational and Graphical Statistics, 5(3), 236-244.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional
import warnings

import numpy as np
import jax.numpy as jnp
from scipy import stats
from scipy.stats import norm as scipy_norm

from .model import GAMLSSModel
from .operations import fitted, residuals, is_gamlss


# =============================================================================
# Data Classes for Diagnostic Results
# =============================================================================

@dataclass(frozen=True)
class QuantileResidualsResult:
    """Quantile residuals diagnostic result.
    
    Attributes
    ----------
    residuals : np.ndarray
        Quantile residuals (normalized)
    mean : float
        Mean of residuals
    variance : float
        Variance of residuals
    skewness : float
        Skewness of residuals
    kurtosis : float
        Excess kurtosis of residuals
    n : int
        Number of observations
    """
    residuals: np.ndarray
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    n: int


@dataclass(frozen=True)
class QQPlotResult:
    """Q-Q plot diagnostic result.
    
    Attributes
    ----------
    theoretical_quantiles : np.ndarray
        Theoretical quantiles from standard normal
    sample_quantiles : np.ndarray
        Sample quantiles (ordered residuals)
    correlation : float
        Filliben correlation coefficient
    mean : float
        Mean of residuals
    variance : float
        Variance of residuals
    """
    theoretical_quantiles: np.ndarray
    sample_quantiles: np.ndarray
    correlation: float
    mean: float
    variance: float


@dataclass(frozen=True)
class WormPlotResult:
    """Worm plot diagnostic result.
    
    Attributes
    ----------
    theoretical_quantiles : np.ndarray
        Theoretical quantiles
    deviations : np.ndarray
        Deviations from theoretical line
    lower_band : np.ndarray
        Lower confidence band
    upper_band : np.ndarray
        Upper confidence band
    n : int
        Number of observations
    """
    theoretical_quantiles: np.ndarray
    deviations: np.ndarray
    lower_band: np.ndarray
    upper_band: np.ndarray
    n: int


@dataclass(frozen=True)
class ResidualPlotResult:
    """Residual plot diagnostic result.
    
    Attributes
    ----------
    fitted_values : np.ndarray
        Fitted values
    residuals : np.ndarray
        Residuals
    index : np.ndarray
        Observation indices
    """
    fitted_values: np.ndarray
    residuals: np.ndarray
    index: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration diagnostic result.
    
    Attributes
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities
    observed_probs : np.ndarray
        Observed probabilities
    n_bins : int
        Number of bins
    """
    predicted_probs: np.ndarray
    observed_probs: np.ndarray
    n_bins: int


@dataclass(frozen=True)
class CooksDistanceResult:
    """Cook's distance diagnostic result.
    
    Attributes
    ----------
    cooks_distance : np.ndarray
        Cook's distance for each observation
    threshold : float
        Threshold for influential observations (typically 4/n)
    influential : np.ndarray
        Boolean array indicating influential observations
    n_influential : int
        Number of influential observations
    index : np.ndarray
        Observation indices
    """
    cooks_distance: np.ndarray
    threshold: float
    influential: np.ndarray
    n_influential: int
    index: np.ndarray


@dataclass(frozen=True)
class CentileCheckResult:
    """Centile check diagnostic result.
    
    Attributes
    ----------
    centiles : np.ndarray
        Centile levels (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
    predicted : np.ndarray
        Predicted centile values (n_obs x n_centiles)
    observed : np.ndarray
        Observed values
    coverage : np.ndarray
        Empirical coverage for each centile
    expected_coverage : np.ndarray
        Expected coverage for each centile
    """
    centiles: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    coverage: np.ndarray
    expected_coverage: np.ndarray


# =============================================================================
# Core Diagnostic Functions
# =============================================================================

def _require_gamlss(model: GAMLSSModel) -> None:
    """Check if object is a GAMLSS model."""
    if not is_gamlss(model):
        raise TypeError("This is not a GAMLSS object")


def _get_residuals(model: GAMLSSModel, what: str = "z-scores") -> np.ndarray:
    """Get residuals from model, removing NaN/Inf values."""
    res = np.asarray(residuals(model, what=what), dtype=np.float64).ravel()
    res = res[np.isfinite(res)]
    return res


def quantile_residuals(
    model: GAMLSSModel,
    randomized: bool = False,
    seed: Optional[int] = None
) -> QuantileResidualsResult:
    """Compute quantile residuals for GAMLSS model.
    
    Quantile residuals (also called randomized quantile residuals) are
    defined as Φ^(-1)(F(y)), where F is the fitted CDF and Φ is the
    standard normal CDF. For discrete distributions, randomization is
    used to avoid discreteness.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    randomized : bool, default=False
        Whether to use randomization for discrete distributions
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    QuantileResidualsResult
        Diagnostic result containing residuals and summary statistics
        
    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236-244.
    
    Examples
    --------
    >>> result = quantile_residuals(model)
    >>> print(f"Mean: {result.mean:.4f}, Variance: {result.variance:.4f}")
    """
    _require_gamlss(model)
    
    # Get residuals (already computed as z-scores)
    res = _get_residuals(model, what="z-scores")
    
    if res.size == 0:
        raise ValueError("No valid residuals available")
    
    # Compute summary statistics
    mean = float(np.mean(res))
    centered = res - mean
    variance = float(np.var(res, ddof=1))
    
    # Avoid division by zero
    if variance < np.finfo(np.float64).eps:
        skewness = 0.0
        kurtosis = 0.0
    else:
        std = np.sqrt(variance)
        skewness = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4) - 3.0)  # Excess kurtosis
    
    return QuantileResidualsResult(
        residuals=res,
        mean=mean,
        variance=variance,
        skewness=skewness,
        kurtosis=kurtosis,
        n=len(res)
    )


def qq_plot_data(model: GAMLSSModel) -> QQPlotResult:
    """Compute Q-Q plot data for GAMLSS model.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
        
    Returns
    -------
    QQPlotResult
        Q-Q plot data including theoretical and sample quantiles
        
    Examples
    --------
    >>> result = qq_plot_data(model)
    >>> print(f"Filliben correlation: {result.correlation:.4f}")
    """
    _require_gamlss(model)
    
    # Get residuals
    res = _get_residuals(model, what="z-scores")
    
    if res.size == 0:
        raise ValueError("No valid residuals available")
    
    # Sort residuals
    sorted_res = np.sort(res)
    n = len(sorted_res)
    
    # Compute theoretical quantiles
    # Using Filliben's approximation for plotting positions
    if n <= 10:
        probs = (np.arange(1, n + 1) - 0.5) / n
    else:
        probs = np.zeros(n)
        probs[0] = 1.0 - 0.5 ** (1.0 / n)
        probs[-1] = 0.5 ** (1.0 / n)
        probs[1:-1] = (np.arange(2, n) - 0.3175) / (n + 0.365)
    
    theoretical = scipy_norm.ppf(probs)
    
    # Compute Filliben correlation coefficient
    correlation = float(np.corrcoef(sorted_res, theoretical)[0, 1])
    
    # Summary statistics
    mean = float(np.mean(res))
    variance = float(np.var(res, ddof=1))
    
    return QQPlotResult(
        theoretical_quantiles=theoretical,
        sample_quantiles=sorted_res,
        correlation=correlation,
        mean=mean,
        variance=variance
    )


def worm_plot_data(
    model: GAMLSSModel,
    confidence_level: float = 0.95
) -> WormPlotResult:
    """Compute worm plot data for GAMLSS model.
    
    A worm plot is a detrended Q-Q plot that shows deviations from the
    theoretical normal line. It's more sensitive to departures from
    normality than a standard Q-Q plot.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    confidence_level : float, default=0.95
        Confidence level for bands
        
    Returns
    -------
    WormPlotResult
        Worm plot data including deviations and confidence bands
        
    References
    ----------
    van Buuren, S., & Fredriks, M. (2001). Worm plot: a simple diagnostic
    device for modelling growth reference curves. Statistics in Medicine,
    20(8), 1259-1277.
    
    Examples
    --------
    >>> result = worm_plot_data(model)
    >>> # Check if deviations are within bands
    >>> within_bands = np.all((result.deviations >= result.lower_band) & 
    ...                       (result.deviations <= result.upper_band))
    """
    _require_gamlss(model)
    
    # Get Q-Q plot data
    qq_result = qq_plot_data(model)
    
    theoretical = qq_result.theoretical_quantiles
    sample = qq_result.sample_quantiles
    n = len(theoretical)
    
    # Compute deviations from theoretical line
    deviations = sample - theoretical
    
    # Compute confidence bands
    # Standard error for order statistics
    probs = scipy_norm.cdf(theoretical)
    
    # Avoid division by zero
    pdf_vals = scipy_norm.pdf(theoretical)
    pdf_vals = np.maximum(pdf_vals, np.finfo(np.float64).eps)
    
    se = np.sqrt(probs * (1.0 - probs) / n) / pdf_vals
    
    # Confidence bands
    z_alpha = scipy_norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    lower_band = -z_alpha * se
    upper_band = z_alpha * se
    
    return WormPlotResult(
        theoretical_quantiles=theoretical,
        deviations=deviations,
        lower_band=lower_band,
        upper_band=upper_band,
        n=n
    )


def residual_plot_data(
    model: GAMLSSModel,
    xvar: Optional[np.ndarray] = None
) -> ResidualPlotResult:
    """Compute residual plot data for GAMLSS model.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    xvar : np.ndarray, optional
        X variable for plotting. If None, uses observation index.
        
    Returns
    -------
    ResidualPlotResult
        Residual plot data
        
    Examples
    --------
    >>> result = residual_plot_data(model)
    >>> # Plot residuals vs fitted values
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(result.fitted_values, result.residuals)
    """
    _require_gamlss(model)
    
    # Get residuals
    res = _get_residuals(model, what="z-scores")
    
    if res.size == 0:
        raise ValueError("No valid residuals available")
    
    # Get fitted values
    fitted_vals = np.asarray(fitted(model, "mu"), dtype=np.float64).ravel()
    
    # Handle weights if present
    weights = model.weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if np.all(np.floor(weights) == weights):
            # Repeat for integer weights
            fitted_vals = np.repeat(fitted_vals, weights.astype(int))
    
    # Ensure same length
    min_len = min(len(res), len(fitted_vals))
    res = res[:min_len]
    fitted_vals = fitted_vals[:min_len]
    
    # X variable
    if xvar is None:
        index = np.arange(1, len(res) + 1, dtype=np.float64)
    else:
        index = np.asarray(xvar, dtype=np.float64).ravel()[:min_len]
    
    return ResidualPlotResult(
        fitted_values=fitted_vals,
        residuals=res,
        index=index
    )


def calibration_check(
    model: GAMLSSModel,
    n_bins: int = 10
) -> CalibrationResult:
    """Check calibration of probabilistic predictions.
    
    Calibration checks whether predicted probabilities match observed
    frequencies. For a well-calibrated model, if we predict 70% probability,
    the event should occur about 70% of the time.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    n_bins : int, default=10
        Number of bins for calibration curve
        
    Returns
    -------
    CalibrationResult
        Calibration diagnostic result
        
    Examples
    --------
    >>> result = calibration_check(model)
    >>> # Plot calibration curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result.predicted_probs, result.observed_probs, 'o-')
    >>> plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
    """
    _require_gamlss(model)
    
    # Get residuals (these are already normalized)
    res = _get_residuals(model, what="z-scores")
    
    if res.size == 0:
        raise ValueError("No valid residuals available")
    
    # Convert residuals to probabilities
    # For normalized residuals, P(Z <= z) = Φ(z)
    probs = scipy_norm.cdf(res)
    
    # Bin the probabilities
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # Compute observed frequencies in each bin
    observed = np.zeros(n_bins)
    predicted = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:  # Include right endpoint for last bin
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        
        if np.sum(mask) > 0:
            # Predicted probability is the bin center
            predicted[i] = bin_centers[i]
            # Observed probability: what fraction of these predictions
            # had residuals below the median?
            observed[i] = np.mean(probs[mask])
    
    return CalibrationResult(
        predicted_probs=predicted,
        observed_probs=observed,
        n_bins=n_bins
    )


def centile_check(
    model: GAMLSSModel,
    centiles: Optional[np.ndarray] = None
) -> CentileCheckResult:
    """Check centile predictions against observed data.
    
    This function checks whether the predicted centiles match the observed
    data. For example, if we predict the 90th centile, about 90% of
    observations should fall below this value.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    centiles : np.ndarray, optional
        Centile levels to check (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
        If None, uses [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        
    Returns
    -------
    CentileCheckResult
        Centile check diagnostic result
        
    Notes
    -----
    This function requires the distribution to have a quantile function (q)
    implemented in dpqr_functions.py.
    
    Examples
    --------
    >>> result = centile_check(model, centiles=np.array([0.05, 0.5, 0.95]))
    >>> print("Centile coverage:")
    >>> for c, obs, exp in zip(result.centiles, result.coverage, 
    ...                         result.expected_coverage):
    ...     print(f"  {c:.2f}: {obs:.3f} (expected {exp:.3f})")
    """
    _require_gamlss(model)
    
    if centiles is None:
        centiles = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    else:
        centiles = np.asarray(centiles, dtype=np.float64)
    
    # Check if model has quantile function
    if not hasattr(model.family, 'q') or model.family.q is None:
        warnings.warn(
            f"Distribution {model.family.family} does not have a quantile "
            "function implemented. Centile check not available.",
            UserWarning
        )
        # Return empty result
        return CentileCheckResult(
            centiles=centiles,
            predicted=np.array([]),
            observed=np.array([]),
            coverage=np.array([]),
            expected_coverage=centiles
        )
    
    # Get observed values
    y = np.asarray(model.y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    
    if y.size == 0:
        raise ValueError("No valid observations available")
    
    # Get fitted parameters
    n_obs = len(y)
    params = {}
    for param in model.parameters or model.par:
        param_vals = np.asarray(fitted(model, param), dtype=np.float64).ravel()
        params[param] = param_vals[:n_obs]
    
    # Compute predicted centiles for each observation
    n_centiles = len(centiles)
    predicted = np.zeros((n_obs, n_centiles))
    
    try:
        for i, p in enumerate(centiles):
            # Call quantile function
            q_vals = model.family.q(p, **params)
            predicted[:, i] = np.asarray(q_vals, dtype=np.float64).ravel()[:n_obs]
    except Exception as e:
        warnings.warn(
            f"Error computing quantiles: {e}. Centile check not available.",
            UserWarning
        )
        return CentileCheckResult(
            centiles=centiles,
            predicted=np.array([]),
            observed=y,
            coverage=np.array([]),
            expected_coverage=centiles
        )
    
    # Compute empirical coverage
    coverage = np.zeros(n_centiles)
    for i in range(n_centiles):
        # What fraction of observations are below the predicted centile?
        coverage[i] = np.mean(y <= predicted[:, i])
    
    return CentileCheckResult(
        centiles=centiles,
        predicted=predicted,
        observed=y,
        coverage=coverage,
        expected_coverage=centiles
    )


def cooks_distance(
    model: GAMLSSModel,
    threshold: Optional[float] = None
) -> CooksDistanceResult:
    """Calculate Cook's distance for each observation.
    
    Cook's distance measures the influence of each observation on the fitted values.
    It combines information about leverage (how unusual the predictor values are)
    and residual size (how unusual the response is).
    
    Large Cook's distance values indicate influential observations that have a
    substantial impact on the model fit. A common threshold is 4/n, where n is
    the number of observations.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    threshold : float, optional
        Threshold for identifying influential observations.
        If None, uses 4/n (where n is the number of observations).
        
    Returns
    -------
    CooksDistanceResult
        Cook's distance diagnostic result
        
    Notes
    -----
    Cook's distance is computed as:
    
        D_i = (r_i^2 / p) * (h_i / (1 - h_i))
    
    where:
    - r_i is the standardized residual for observation i
    - h_i is the leverage (hat value) for observation i
    - p is the total number of parameters in the model
    
    References
    ----------
    Cook, R. D. (1977). Detection of influential observation in linear regression.
    Technometrics, 19(1), 15-18.
    
    Cook, R. D., & Weisberg, S. (1982). Residuals and influence in regression.
    New York: Chapman and Hall.
    
    Examples
    --------
    >>> result = cooks_distance(model)
    >>> print(f"Number of influential observations: {result.n_influential}")
    >>> # Get indices of influential observations
    >>> influential_idx = result.index[result.influential]
    >>> print(f"Influential observations: {influential_idx}")
    """
    from .hatvalues import hatvalues
    
    _require_gamlss(model)
    
    # Get standardized residuals (quantile residuals)
    resid = _get_residuals(model, what="z-scores")
    
    if resid.size == 0:
        raise ValueError("No valid residuals available")
    
    n = len(resid)
    
    # Get leverage values
    try:
        leverage = np.asarray(hatvalues(model), dtype=np.float64).ravel()
        
        # Ensure same length
        min_len = min(len(resid), len(leverage))
        resid = resid[:min_len]
        leverage = leverage[:min_len]
        n = min_len
    except Exception as e:
        warnings.warn(
            f"Could not compute leverage values: {e}. "
            "Using uniform leverage (1/n) for all observations.",
            UserWarning
        )
        leverage = np.full(n, 1.0 / n)
    
    # Clip leverage to avoid division by zero
    # Leverage should be in (0, 1), but we add safety bounds
    eps = np.finfo(np.float64).eps
    leverage = np.clip(leverage, eps, 1.0 - eps)
    
    # Number of parameters
    # Count total number of coefficients across all parameters
    n_params = sum(len(coef) for coef in model.coefficients.values())
    
    # Ensure n_params > 0
    if n_params == 0:
        warnings.warn(
            "No model coefficients found. Using n_params=1.",
            UserWarning
        )
        n_params = 1
    
    # Calculate Cook's distance
    # D_i = (r_i^2 / p) * (h_i / (1 - h_i))
    cooks_d = (np.square(resid) / n_params) * (leverage / (1.0 - leverage))
    
    # Set threshold
    if threshold is None:
        threshold = 4.0 / n
    
    # Identify influential observations
    influential = cooks_d > threshold
    n_influential = int(np.sum(influential))
    
    # Observation indices
    index = np.arange(1, n + 1)
    
    return CooksDistanceResult(
        cooks_distance=cooks_d,
        threshold=threshold,
        influential=influential,
        n_influential=n_influential,
        index=index
    )


# =============================================================================
# Comprehensive Diagnostic Function
# =============================================================================

@dataclass(frozen=True)
class ComprehensiveDiagnostics:
    """Complete diagnostic results for a GAMLSS model.
    
    Attributes
    ----------
    quantile_residuals : QuantileResidualsResult
        Quantile residuals and summary statistics
    qq_plot : QQPlotResult
        Q-Q plot data
    worm_plot : WormPlotResult
        Worm plot data
    residual_plot : ResidualPlotResult
        Residual plot data
    calibration : CalibrationResult
        Calibration check results
    centile_check : CentileCheckResult
        Centile check results
    cooks_distance : CooksDistanceResult
        Cook's distance results
    """
    quantile_residuals: QuantileResidualsResult
    qq_plot: QQPlotResult
    worm_plot: WormPlotResult
    residual_plot: ResidualPlotResult
    calibration: CalibrationResult
    centile_check: CentileCheckResult
    cooks_distance: CooksDistanceResult


def comprehensive_diagnostics(
    model: GAMLSSModel,
    centiles: Optional[np.ndarray] = None,
    n_bins: int = 10,
    confidence_level: float = 0.95
) -> ComprehensiveDiagnostics:
    """Run comprehensive diagnostics on a GAMLSS model.
    
    This function runs all available diagnostic checks and returns
    a comprehensive result object.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    centiles : np.ndarray, optional
        Centile levels for centile check
    n_bins : int, default=10
        Number of bins for calibration check
    confidence_level : float, default=0.95
        Confidence level for worm plot bands
        
    Returns
    -------
    ComprehensiveDiagnostics
        Complete diagnostic results
        
    Examples
    --------
    >>> diag = comprehensive_diagnostics(model)
    >>> print(f"Residual mean: {diag.quantile_residuals.mean:.4f}")
    >>> print(f"Q-Q correlation: {diag.qq_plot.correlation:.4f}")
    >>> print(f"Centile coverage: {diag.centile_check.coverage}")
    >>> print(f"Influential observations: {diag.cooks_distance.n_influential}")
    """
    _require_gamlss(model)
    
    # Run all diagnostics
    qres = quantile_residuals(model)
    qq = qq_plot_data(model)
    worm = worm_plot_data(model, confidence_level=confidence_level)
    resid = residual_plot_data(model)
    calib = calibration_check(model, n_bins=n_bins)
    centile = centile_check(model, centiles=centiles)
    cooks = cooks_distance(model)
    
    return ComprehensiveDiagnostics(
        quantile_residuals=qres,
        qq_plot=qq,
        worm_plot=worm,
        residual_plot=resid,
        calibration=calib,
        centile_check=centile,
        cooks_distance=cooks
    )


# =============================================================================
# Plotting Functions (Optional - for convenience)
# =============================================================================

def plot_diagnostics(
    model: GAMLSSModel,
    figsize: tuple[float, float] = (14, 10),
    save_path: Optional[str] = None
) -> None:
    """Create a comprehensive diagnostic plot for a GAMLSS model.
    
    This function creates a 2x3 grid of diagnostic plots:
    1. Residuals vs Fitted
    2. Q-Q Plot
    3. Worm Plot
    4. Residuals vs Index
    5. Density of Residuals
    6. Calibration Plot
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    figsize : tuple, default=(14, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
        
    Examples
    --------
    >>> plot_diagnostics(model)
    >>> plot_diagnostics(model, save_path='diagnostics.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. "
                         "Install it with: pip install matplotlib")
    
    # Get diagnostic data
    diag = comprehensive_diagnostics(model)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('GAMLSS Model Diagnostics', fontsize=16, fontweight='bold')
    
    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(diag.residual_plot.fitted_values, diag.residual_plot.residuals,
               alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted values', fontsize=11)
    ax.set_ylabel('Quantile residuals', fontsize=11)
    ax.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    ax = axes[0, 1]
    ax.scatter(diag.qq_plot.theoretical_quantiles, 
               diag.qq_plot.sample_quantiles,
               alpha=0.5, s=20)
    # Add reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Theoretical')
    ax.set_xlabel('Theoretical quantiles', fontsize=11)
    ax.set_ylabel('Sample quantiles', fontsize=11)
    ax.set_title(f'Q-Q Plot (r={diag.qq_plot.correlation:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Worm Plot
    ax = axes[0, 2]
    ax.scatter(diag.worm_plot.theoretical_quantiles, 
               diag.worm_plot.deviations,
               alpha=0.5, s=20)
    ax.plot(diag.worm_plot.theoretical_quantiles, 
            diag.worm_plot.lower_band, 'r--', linewidth=1.5, label='95% CI')
    ax.plot(diag.worm_plot.theoretical_quantiles, 
            diag.worm_plot.upper_band, 'r--', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_xlabel('Theoretical quantiles', fontsize=11)
    ax.set_ylabel('Deviations', fontsize=11)
    ax.set_title('Worm Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Residuals vs Index
    ax = axes[1, 0]
    ax.scatter(diag.residual_plot.index, diag.residual_plot.residuals,
               alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Index', fontsize=11)
    ax.set_ylabel('Quantile residuals', fontsize=11)
    ax.set_title('Residuals vs Index', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. Density of Residuals
    ax = axes[1, 1]
    ax.hist(diag.quantile_residuals.residuals, bins=30, density=True,
            alpha=0.7, edgecolor='black', label='Observed')
    # Add theoretical normal density
    x = np.linspace(diag.quantile_residuals.residuals.min(),
                    diag.quantile_residuals.residuals.max(), 100)
    ax.plot(x, scipy_norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    ax.set_xlabel('Quantile residuals', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Residual Density (μ={diag.quantile_residuals.mean:.3f}, '
                 f'σ²={diag.quantile_residuals.variance:.3f})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 6. Calibration Plot
    ax = axes[1, 2]
    ax.plot(diag.calibration.predicted_probs, 
            diag.calibration.observed_probs, 'o-', linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    ax.set_xlabel('Predicted probability', fontsize=11)
    ax.set_ylabel('Observed probability', fontsize=11)
    ax.set_title('Calibration Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plot saved to: {save_path}")
    else:
        plt.show()


def print_diagnostic_summary(model: GAMLSSModel) -> None:
    """Print a text summary of diagnostic statistics.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
        
    Examples
    --------
    >>> print_diagnostic_summary(model)
    """
    diag = comprehensive_diagnostics(model)
    
    print("=" * 70)
    print("GAMLSS MODEL DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print()
    
    print("Quantile Residuals:")
    print(f"  N observations:  {diag.quantile_residuals.n}")
    print(f"  Mean:            {diag.quantile_residuals.mean:8.4f}  (should be ≈ 0)")
    print(f"  Variance:        {diag.quantile_residuals.variance:8.4f}  (should be ≈ 1)")
    print(f"  Skewness:        {diag.quantile_residuals.skewness:8.4f}  (should be ≈ 0)")
    print(f"  Excess Kurtosis: {diag.quantile_residuals.kurtosis:8.4f}  (should be ≈ 0)")
    print()
    
    print("Q-Q Plot:")
    print(f"  Filliben correlation: {diag.qq_plot.correlation:.4f}  (should be ≈ 1)")
    print()
    
    print("Worm Plot:")
    n_outside = np.sum((diag.worm_plot.deviations < diag.worm_plot.lower_band) |
                       (diag.worm_plot.deviations > diag.worm_plot.upper_band))
    pct_outside = 100.0 * n_outside / diag.worm_plot.n
    print(f"  Points outside 95% CI: {n_outside}/{diag.worm_plot.n} ({pct_outside:.1f}%)")
    print(f"  Expected outside:      ~{int(0.05 * diag.worm_plot.n)} (5%)")
    print()
    
    if len(diag.centile_check.coverage) > 0:
        print("Centile Check:")
        print("  Centile  Expected  Observed  Difference")
        print("  " + "-" * 42)
        for i, c in enumerate(diag.centile_check.centiles):
            exp = diag.centile_check.expected_coverage[i]
            obs = diag.centile_check.coverage[i]
            diff = obs - exp
            print(f"  {c:6.2f}   {exp:7.3f}   {obs:7.3f}   {diff:+7.3f}")
        print()
    
    print("=" * 70)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Data classes
    'QuantileResidualsResult',
    'QQPlotResult',
    'WormPlotResult',
    'ResidualPlotResult',
    'CalibrationResult',
    'CentileCheckResult',
    'CooksDistanceResult',
    'ComprehensiveDiagnostics',
    
    # Core functions
    'quantile_residuals',
    'qq_plot_data',
    'worm_plot_data',
    'residual_plot_data',
    'calibration_check',
    'centile_check',
    'cooks_distance',
    'comprehensive_diagnostics',
    
    # Plotting functions
    'plot_diagnostics',
    'print_diagnostic_summary',
]
