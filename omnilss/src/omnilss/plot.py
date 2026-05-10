"""R-aligned diagnostic plot data interfaces.

R source reference:
- file: `gamlss/R/plot.R`
- functions: `plot.gamlss`
- file: `gamlss/R/rqresplot_new.R`
- functions: `get.rqres`, `rqres.plot`
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from .model import GAMLSSModel
from .operations import fitted, is_gamlss, residuals


@dataclass(frozen=True)
class QQStatsResult:
    """Compact normal QQ diagnostic summary for staged diagnostics."""

    theoretical: np.ndarray
    ordered: np.ndarray
    mean: float
    variance: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class RQResSamplesResult:
    """Structured randomised-quantile residual sample matrix for diagnostics."""

    samples: np.ndarray
    ordered: bool
    howmany: int


@dataclass(frozen=True)
class WormPlotResult:
    """Structured worm-plot statistics based on quantile residual samples."""

    theoretical: np.ndarray
    median_deviation: np.ndarray
    lower_band: np.ndarray
    upper_band: np.ndarray
    sample_deviations: np.ndarray


@dataclass(frozen=True)
class PlotDiagnosticsResult:
    """Structured data bundle matching the main `plot.gamlss` panels."""

    fitted_x: np.ndarray
    residual_y: np.ndarray
    index_x: np.ndarray
    density_x: np.ndarray
    density_y: np.ndarray
    qq_theoretical: np.ndarray
    qq_ordered: np.ndarray
    summary_stats: dict[str, float]


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _ordered_res(object: GAMLSSModel) -> np.ndarray:
    values = np.asarray(residuals(object, what="z-scores"), dtype=np.float64).ravel()
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def _kernel_density(values: np.ndarray, points: int = 128) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("need at least one value for density estimation")
    if values.size == 1:
        x = np.array([values[0]], dtype=np.float64)
        y = np.array([1.0], dtype=np.float64)
        return x, y
    std = float(np.std(values, ddof=1))
    spread = max(std, np.finfo(np.float64).eps)
    bandwidth = max(1.06 * spread * values.size ** (-1.0 / 5.0), 1e-3)
    x = np.linspace(values.min() - 3.0 * bandwidth, values.max() + 3.0 * bandwidth, points, dtype=np.float64)
    z = (x[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)
    y = np.mean(kernel, axis=1) / bandwidth
    return x, y


def qq_stats(
    object: GAMLSSModel,
) -> QQStatsResult:
    """R references:
    - `gamlss/R/plot.R::plot.gamlss`
    - `gamlss/R/qstats.R::qstats`

    Staged behavior:
    - Returns QQ inputs and residual moment summaries instead of plotting.
    """

    _require_gamlss(object)
    z = np.sort(_ordered_res(object))
    n = z.size
    if n == 0:
        raise ValueError("need at least one residual for QQ diagnostics")
    probs = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / n
    theoretical = np.array([NormalDist().inv_cdf(float(p)) for p in probs], dtype=np.float64)
    mean = float(np.mean(z))
    centered = z - mean
    variance = float(np.mean(np.square(centered)))
    scale = max(variance, np.finfo(np.float64).eps)
    skewness = float(np.mean(np.power(centered, 3.0)) / (scale ** 1.5))
    kurtosis = float(np.mean(np.power(centered, 4.0)) / (scale ** 2.0))
    return QQStatsResult(
        theoretical=theoretical,
        ordered=z,
        mean=mean,
        variance=variance,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def get_rqres_samples(
    object: GAMLSSModel,
    howmany: int = 10,
    order: bool = False,
) -> RQResSamplesResult:
    """R reference: `gamlss/R/rqresplot_new.R::get.rqres`.

    Staged behavior:
    - Uses the stored `rqres` callable when available.
    - Current discrete implementation is deterministic, so repeated samples are
      identical copies. This still provides a stable matrix API for later plots.
    """

    _require_gamlss(object)
    if howmany < 1:
        raise ValueError("howmany must be at least 1")
    if object.rqres is None:
        raise ValueError("object does not contain an rqres callable")

    weights = object.weights
    if weights is None:
        weights = np.ones(object.n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    y = np.asarray(object.y, dtype=np.float64)
    active_parameters = object.parameters or object.par

    if np.all(np.floor(weights) == weights):
        repeated_y = np.repeat(y, weights.astype(int))
        kwargs: dict[str, Any] = {"y": repeated_y}
        for parameter in active_parameters:
            kwargs[parameter] = np.repeat(
                np.asarray(fitted(object, parameter), dtype=np.float64),
                weights.astype(int),
            )
    else:
        kwargs = {"y": y}
        for parameter in active_parameters:
            kwargs[parameter] = np.asarray(fitted(object, parameter), dtype=np.float64)

    base = np.asarray(object.rqres(**kwargs), dtype=np.float64)
    if order:
        base = np.sort(base)
    samples = np.repeat(base[:, None], int(howmany), axis=1)
    return RQResSamplesResult(samples=samples, ordered=bool(order), howmany=int(howmany))


def worm_plot_data(
    object: GAMLSSModel,
    howmany: int = 6,
) -> WormPlotResult:
    """R references:
    - `gamlss/R/rqresplot_new.R::rqres.plot`
    - `gamlss/R/wp.R::wp`

    Staged behavior:
    - Returns worm-plot summary statistics instead of plotting.
    - Uses median residual sample deviation and normal-reference bands.
    """

    _require_gamlss(object)
    rqres_samples = get_rqres_samples(object, howmany=howmany, order=True)
    sample_matrix = np.asarray(rqres_samples.samples, dtype=np.float64)
    if sample_matrix.shape[0] == 0:
        raise ValueError("need at least one residual sample for worm plot diagnostics")
    theoretical = qq_stats(object).theoretical
    sample_deviations = sample_matrix - theoretical[:, None]
    median_deviation = np.median(sample_deviations, axis=1)

    n = theoretical.size
    level = 0.95
    probs = np.array([NormalDist().cdf(float(value)) for value in theoretical], dtype=np.float64)
    se = (1.0 / np.maximum(np.exp(-0.5 * theoretical**2) / np.sqrt(2.0 * np.pi), np.finfo(np.float64).eps)) * np.sqrt(
        probs * (1.0 - probs) / max(n, 1)
    )
    z_alpha = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    lower = -z_alpha * se
    upper = z_alpha * se
    return WormPlotResult(
        theoretical=theoretical,
        median_deviation=median_deviation,
        lower_band=lower,
        upper_band=upper,
        sample_deviations=sample_deviations,
    )


def plot_diagnostics(
    object: GAMLSSModel,
    xvar: Any | None = None,
) -> PlotDiagnosticsResult:
    """R reference: `gamlss/R/plot.R::plot.gamlss`.

    Staged behavior:
    - Returns the data needed for the four standard panels.
    - Does not perform plotting directly.
    """

    _require_gamlss(object)
    residx = _ordered_res(object)
    if residx.size == 0:
        raise ValueError("there are no quantile residuals in the object")

    weights = object.weights
    if weights is None:
        weights = np.ones(object.n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    base_fitted = np.asarray(fitted(object, "mu"), dtype=np.float64)
    if residx.size == base_fitted.size:
        fittedvalues = base_fitted
        index_x = np.arange(1, residx.size + 1, dtype=np.float64) if xvar is None else np.asarray(xvar, dtype=np.float64)
    else:
        fittedvalues = np.repeat(base_fitted, weights.astype(int))
        index_x = (
            np.arange(1, residx.size + 1, dtype=np.float64)
            if xvar is None
            else np.repeat(np.asarray(xvar, dtype=np.float64), weights.astype(int))
        )

    density_x, density_y = _kernel_density(residx)
    qq = qq_stats(object)
    filliben = float(np.corrcoef(qq.ordered, qq.theoretical)[0, 1]) if qq.ordered.size > 1 else float("nan")

    summary_stats = {
        "mean": float(qq.mean),
        "variance": float(qq.variance),
        "skewness": float(qq.skewness),
        "kurtosis": float(qq.kurtosis),
        "filliben": filliben,
    }
    return PlotDiagnosticsResult(
        fitted_x=fittedvalues,
        residual_y=residx,
        index_x=index_x,
        density_x=density_x,
        density_y=density_y,
        qq_theoretical=qq.theoretical,
        qq_ordered=qq.ordered,
        summary_stats=summary_stats,
    )


# R-style exact-name alias
plot_gamlss = plot_diagnostics

__all__ = [
    "PlotDiagnosticsResult",
    "QQStatsResult",
    "RQResSamplesResult",
    "WormPlotResult",
    "get_rqres_samples",
    "plot_diagnostics",
    "plot_gamlss",
    "qq_stats",
    "worm_plot_data",
]
