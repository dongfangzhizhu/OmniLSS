"""Worm Plot diagnostics for GAMLSS models.

The worm plot is a detrended Q-Q plot that shows deviations from the
theoretical normal line. It's particularly useful for detecting local
departures from the assumed distribution.

This module provides:
- Data computation for worm plots
- Static plotting with matplotlib
- Interactive plotting with plotly
- Multiple panel worm plots for different groups

References
----------
- van Buuren, S., & Fredriks, M. (2001). Worm plot: a simple diagnostic
  device for modelling growth reference curves. Statistics in Medicine,
  20(8), 1259-1277.
- Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
  for location, scale and shape. Journal of the Royal Statistical Society:
  Series C (Applied Statistics), 54(3), 507-554.

Examples
--------
Basic worm plot:

>>> from omnilss import gamlss
>>> from omnilss.worm_plot import wp, wp_interactive
>>> 
>>> # Fit model
>>> model = gamlss("y ~ x", family="NO", data=data)
>>> 
>>> # Create worm plot
>>> wp(model)
>>> 
>>> # Interactive version
>>> fig = wp_interactive(model)
>>> fig.show()

Multiple panel worm plot:

>>> # Split by groups
>>> wp(model, xvar=data["group"], n_inter=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Literal
import warnings

import numpy as np
import jax.numpy as jnp
from scipy import stats
from scipy.stats import norm as scipy_norm

from .model import GAMLSSModel
from .operations import residuals, is_gamlss


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class WormPlotData:
    """Worm plot data for a single panel.
    
    Attributes
    ----------
    theoretical_quantiles : np.ndarray
        Theoretical quantiles from standard normal
    deviations : np.ndarray
        Deviations from theoretical line (sample - theoretical)
    lower_band : np.ndarray
        Lower confidence band
    upper_band : np.ndarray
        Upper confidence band
    n : int
        Number of observations
    group_label : str, optional
        Label for this group (if using multiple panels)
    """
    theoretical_quantiles: np.ndarray
    deviations: np.ndarray
    lower_band: np.ndarray
    upper_band: np.ndarray
    n: int
    group_label: Optional[str] = None


@dataclass(frozen=True)
class MultiPanelWormPlotData:
    """Worm plot data for multiple panels.
    
    Attributes
    ----------
    panels : list[WormPlotData]
        List of worm plot data for each panel
    n_panels : int
        Number of panels
    """
    panels: list[WormPlotData]
    n_panels: int


# =============================================================================
# Core Functions
# =============================================================================

def _require_gamlss(model: GAMLSSModel) -> None:
    """Check if object is a GAMLSS model."""
    if not is_gamlss(model):
        raise TypeError("This is not a GAMLSS object")


def _get_residuals(model: GAMLSSModel) -> np.ndarray:
    """Get quantile residuals from model, removing NaN/Inf values."""
    res = np.asarray(residuals(model, what="z-scores"), dtype=np.float64).ravel()
    res = res[np.isfinite(res)]
    return res


def _compute_worm_plot_data(
    residuals: np.ndarray,
    confidence_level: float = 0.95,
    group_label: Optional[str] = None
) -> WormPlotData:
    """Compute worm plot data from residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Quantile residuals (z-scores)
    confidence_level : float, default=0.95
        Confidence level for bands
    group_label : str, optional
        Label for this group
        
    Returns
    -------
    WormPlotData
        Worm plot data
    """
    if len(residuals) == 0:
        raise ValueError("No valid residuals available")
    
    # Sort residuals
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    
    # Compute theoretical quantiles using Filliben's approximation
    if n <= 10:
        probs = (np.arange(1, n + 1) - 0.5) / n
    else:
        probs = np.zeros(n)
        probs[0] = 1.0 - 0.5 ** (1.0 / n)
        probs[-1] = 0.5 ** (1.0 / n)
        probs[1:-1] = (np.arange(2, n) - 0.3175) / (n + 0.365)
    
    theoretical = scipy_norm.ppf(probs)
    
    # Compute deviations from theoretical line
    deviations = sorted_res - theoretical
    
    # Compute confidence bands
    # Standard error for order statistics
    pdf_vals = scipy_norm.pdf(theoretical)
    pdf_vals = np.maximum(pdf_vals, np.finfo(np.float64).eps)
    
    se = np.sqrt(probs * (1.0 - probs) / n) / pdf_vals
    
    # Confidence bands
    z_alpha = scipy_norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    lower_band = -z_alpha * se
    upper_band = z_alpha * se
    
    return WormPlotData(
        theoretical_quantiles=theoretical,
        deviations=deviations,
        lower_band=lower_band,
        upper_band=upper_band,
        n=n,
        group_label=group_label
    )


def wp_data(
    model: GAMLSSModel,
    xvar: Optional[np.ndarray] = None,
    n_inter: int = 4,
    confidence_level: float = 0.95
) -> WormPlotData | MultiPanelWormPlotData:
    """Compute worm plot data for GAMLSS model.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    xvar : np.ndarray, optional
        Variable for grouping (creates multiple panels)
    n_inter : int, default=4
        Number of intervals for grouping (if xvar is provided)
    confidence_level : float, default=0.95
        Confidence level for bands
        
    Returns
    -------
    WormPlotData or MultiPanelWormPlotData
        Worm plot data (single or multiple panels)
        
    Examples
    --------
    Single panel:
    
    >>> data = wp_data(model)
    >>> print(f"N observations: {data.n}")
    
    Multiple panels:
    
    >>> data = wp_data(model, xvar=x, n_inter=4)
    >>> print(f"N panels: {data.n_panels}")
    """
    _require_gamlss(model)
    
    # Get residuals
    res = _get_residuals(model)
    
    if res.size == 0:
        raise ValueError("No valid residuals available")
    
    # Single panel case
    if xvar is None:
        return _compute_worm_plot_data(res, confidence_level)
    
    # Multiple panel case
    xvar = np.asarray(xvar, dtype=np.float64).ravel()
    
    # Ensure same length
    min_len = min(len(res), len(xvar))
    res = res[:min_len]
    xvar = xvar[:min_len]
    
    # Create intervals
    if n_inter < 2:
        n_inter = 2
    if n_inter > 16:
        warnings.warn("n_inter > 16 may create too many panels", UserWarning)
    
    # Compute quantiles for intervals
    quantiles = np.linspace(0, 1, n_inter + 1)
    breaks = np.quantile(xvar, quantiles)
    breaks[0] = -np.inf  # Include all values
    breaks[-1] = np.inf
    
    # Create panels
    panels = []
    for i in range(n_inter):
        mask = (xvar >= breaks[i]) & (xvar < breaks[i + 1])
        if np.sum(mask) < 10:
            warnings.warn(
                f"Panel {i+1} has fewer than 10 observations ({np.sum(mask)})",
                UserWarning
            )
            continue
        
        panel_res = res[mask]
        panel_label = f"[{breaks[i]:.2f}, {breaks[i+1]:.2f})"
        
        panel_data = _compute_worm_plot_data(
            panel_res,
            confidence_level,
            group_label=panel_label
        )
        panels.append(panel_data)
    
    return MultiPanelWormPlotData(panels=panels, n_panels=len(panels))


# =============================================================================
# Plotting Functions
# =============================================================================

def wp(
    model: GAMLSSModel,
    xvar: Optional[np.ndarray] = None,
    n_inter: int = 4,
    confidence_level: float = 0.95,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True
) -> Any:
    """Create worm plot for GAMLSS model.
    
    A worm plot is a detrended Q-Q plot that shows deviations from the
    theoretical normal line. Points should lie within the confidence bands
    if the model is well-specified.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    xvar : np.ndarray, optional
        Variable for grouping (creates multiple panels)
    n_inter : int, default=4
        Number of intervals for grouping (if xvar is provided)
    confidence_level : float, default=0.95
        Confidence level for bands
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Plot title
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
        
    Examples
    --------
    >>> from omnilss import gamlss
    >>> from omnilss.worm_plot import wp
    >>> 
    >>> model = gamlss("y ~ x", family="NO", data=data)
    >>> wp(model)
    >>> 
    >>> # Multiple panels
    >>> wp(model, xvar=data["x"], n_inter=4)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )
    
    # Get data
    data = wp_data(model, xvar, n_inter, confidence_level)
    
    # Determine layout
    if isinstance(data, WormPlotData):
        # Single panel
        n_rows, n_cols = 1, 1
        panels = [data]
    else:
        # Multiple panels
        n_panels = data.n_panels
        n_cols = min(2, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        panels = data.panels
    
    # Create figure
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    
    # Plot each panel
    for idx, panel in enumerate(panels):
        ax = axes[idx]
        
        # Plot confidence bands
        ax.fill_between(
            panel.theoretical_quantiles,
            panel.lower_band,
            panel.upper_band,
            alpha=0.2,
            color='gray',
            label=f'{int(confidence_level*100)}% CI'
        )
        
        # Plot zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Plot deviations
        ax.plot(
            panel.theoretical_quantiles,
            panel.deviations,
            'o',
            markersize=3,
            color='blue',
            alpha=0.6
        )
        
        # Labels
        ax.set_xlabel('Unit normal quantile')
        ax.set_ylabel('Deviation')
        
        # Title
        if panel.group_label:
            ax.set_title(f'{panel.group_label} (n={panel.n})')
        elif title:
            ax.set_title(title)
        else:
            ax.set_title(f'Worm Plot (n={panel.n})')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        if idx == 0:
            ax.legend(loc='best', fontsize='small')
    
    # Hide unused subplots
    for idx in range(len(panels), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def wp_interactive(
    model: GAMLSSModel,
    xvar: Optional[np.ndarray] = None,
    n_inter: int = 4,
    confidence_level: float = 0.95,
    title: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None
) -> Any:
    """Create interactive worm plot using Plotly.
    
    Parameters
    ----------
    model : GAMLSSModel
        Fitted GAMLSS model
    xvar : np.ndarray, optional
        Variable for grouping (creates multiple panels)
    n_inter : int, default=4
        Number of intervals for grouping (if xvar is provided)
    confidence_level : float, default=0.95
        Confidence level for bands
    title : str, optional
        Plot title
    height : int, default=500
        Plot height in pixels
    width : int, optional
        Plot width in pixels
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
        
    Examples
    --------
    >>> from omnilss import gamlss
    >>> from omnilss.worm_plot import wp_interactive
    >>> 
    >>> model = gamlss("y ~ x", family="NO", data=data)
    >>> fig = wp_interactive(model)
    >>> fig.show()
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plotting. "
            "Install it with: pip install plotly"
        )
    
    # Get data
    data = wp_data(model, xvar, n_inter, confidence_level)
    
    # Determine layout
    if isinstance(data, WormPlotData):
        # Single panel
        n_rows, n_cols = 1, 1
        panels = [data]
    else:
        # Multiple panels
        n_panels = data.n_panels
        n_cols = min(2, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        panels = data.panels
    
    # Create subplots
    subplot_titles = [
        panel.group_label if panel.group_label else f'n={panel.n}'
        for panel in panels
    ]
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Plot each panel
    for idx, panel in enumerate(panels):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([panel.theoretical_quantiles, panel.theoretical_quantiles[::-1]]),
                y=np.concatenate([panel.upper_band, panel.lower_band[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int(confidence_level*100)}% CI',
                showlegend=(idx == 0),
                hoverinfo='skip'
            ),
            row=row,
            col=col
        )
        
        # Zero line
        fig.add_trace(
            go.Scatter(
                x=panel.theoretical_quantiles,
                y=np.zeros_like(panel.theoretical_quantiles),
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                name='Zero line',
                showlegend=(idx == 0),
                hoverinfo='skip'
            ),
            row=row,
            col=col
        )
        
        # Deviations
        fig.add_trace(
            go.Scatter(
                x=panel.theoretical_quantiles,
                y=panel.deviations,
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.6),
                name='Deviations',
                showlegend=(idx == 0),
                hovertemplate='<b>Quantile</b>: %{x:.3f}<br>' +
                              '<b>Deviation</b>: %{y:.3f}<br>' +
                              '<extra></extra>'
            ),
            row=row,
            col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text='Unit normal quantile', row=row, col=col)
        fig.update_yaxes(title_text='Deviation', row=row, col=col)
    
    # Update layout
    if width is None:
        width = 600 * n_cols
    
    fig.update_layout(
        title=title if title else 'Worm Plot',
        height=height * n_rows,
        width=width,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


# Aliases for R-style naming
worm_plot = wp
worm_plot_interactive = wp_interactive


__all__ = [
    "WormPlotData",
    "MultiPanelWormPlotData",
    "wp_data",
    "wp",
    "wp_interactive",
    "worm_plot",
    "worm_plot_interactive",
]
