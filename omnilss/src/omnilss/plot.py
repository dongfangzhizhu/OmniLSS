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
class RQResPlotResult:
    """Structured result for R `gamlss::rqres.plot` (QQ or worm plot)."""

    samples: np.ndarray
    rmedian: np.ndarray
    theoretical: np.ndarray
    sample_deviations: np.ndarray
    median_deviation: np.ndarray
    band_z: np.ndarray
    band_low: np.ndarray
    band_high: np.ndarray
    plot_type: str
    type: str
    howmany: int
    figure: Any = None


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


def _kernel_density(
    values: np.ndarray, points: int = 128
) -> tuple[np.ndarray, np.ndarray]:
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
    x = np.linspace(
        values.min() - 3.0 * bandwidth,
        values.max() + 3.0 * bandwidth,
        points,
        dtype=np.float64,
    )
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
    theoretical = np.array(
        [NormalDist().inv_cdf(float(p)) for p in probs], dtype=np.float64
    )
    mean = float(np.mean(z))
    centered = z - mean
    variance = float(np.mean(np.square(centered)))
    scale = max(variance, np.finfo(np.float64).eps)
    skewness = float(np.mean(np.power(centered, 3.0)) / (scale**1.5))
    kurtosis = float(np.mean(np.power(centered, 4.0)) / (scale**2.0))
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
    return RQResSamplesResult(
        samples=samples, ordered=bool(order), howmany=int(howmany)
    )


def _r_ppoints(n: int) -> np.ndarray:
    """R's ``stats::ppoints(n)``: ``(1:n - a) / (n + 1 - 2*a)``."""
    a = 3.0 / 8.0 if n <= 10 else 0.5
    return (np.arange(1, n + 1, dtype=np.float64) - a) / (n + 1.0 - 2.0 * a)


def _expand_rqres_inputs(obj: GAMLSSModel) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Expand response and fitted parameter vectors for integer frequency weights.

    Mirrors ``R/gamlss/R/rqresplot_new.R::get.rqres`` lines 12-30.
    """
    weights = obj.weights
    if weights is None:
        weights = np.ones(obj.n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    y = np.asarray(obj.y, dtype=np.float64).ravel()
    active_parameters = obj.parameters or obj.par

    is_freq = bool(
        weights.size == y.size
        and np.all(np.trunc(weights) == weights)
        and np.all(weights >= 0)
    )
    rep_counts = weights.astype(int) if is_freq else None

    y_out = np.repeat(y, rep_counts) if rep_counts is not None else y
    params: dict[str, np.ndarray] = {}
    for parameter in active_parameters:
        vals = np.asarray(fitted(obj, parameter), dtype=np.float64).ravel()
        if vals.size == 1 and y.size > 1:
            vals = np.full(y.shape, vals.item(), dtype=np.float64)
        params[parameter] = (
            np.repeat(vals, rep_counts) if rep_counts is not None else vals
        )

    # Binomial / fixed parameter (bd) if present in call data
    fixed_names = tuple(getattr(obj.family, "fixed_parameters", ()) or ())
    call_data = None if obj.call is None else obj.call.get("data")
    if call_data is not None:
        for fname in fixed_names:
            if fname in call_data and fname not in params:
                fvals = np.asarray(call_data[fname], dtype=np.float64).ravel()
                if fvals.size == 1 and y.size > 1:
                    fvals = np.full(y.shape, fvals.item(), dtype=np.float64)
                params[fname] = (
                    np.repeat(fvals, rep_counts) if rep_counts is not None else fvals
                )
    return y_out, params


def _draw_one_rqres(
    obj: GAMLSSModel,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one realization of (randomized) quantile residuals."""
    family = obj.family
    fam_type = str(getattr(family, "type", "")).lower()
    cdf_fn = getattr(family, "p", None)

    if fam_type == "discrete" and callable(cdf_fn):
        cdf_hi = np.asarray(cdf_fn(y, **params), dtype=np.float64)
        y_prev = y - 1.0
        cdf_lo = np.where(
            y > 0.0,
            np.asarray(cdf_fn(np.maximum(y_prev, 0.0), **params), dtype=np.float64),
            0.0,
        )
        cdf_lo = np.clip(cdf_lo, 0.0, 1.0)
        cdf_hi = np.clip(np.maximum(cdf_hi, cdf_lo), 0.0, 1.0)
        u = rng.uniform(cdf_lo, cdf_hi)
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        norm = NormalDist()
        return np.array([norm.inv_cdf(float(p)) for p in u], dtype=np.float64)

    if fam_type == "mixed" and callable(cdf_fn):
        cdf_hi = np.asarray(cdf_fn(y, **params), dtype=np.float64)
        at_zero = np.isclose(y, 0.0)
        u = np.where(at_zero, rng.uniform(0.0, np.clip(cdf_hi, 0.0, 1.0)), cdf_hi)
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        norm = NormalDist()
        return np.array([norm.inv_cdf(float(p)) for p in u], dtype=np.float64)

    if obj.rqres is not None:
        return np.asarray(obj.rqres(y=y, **params), dtype=np.float64).ravel()

    if callable(cdf_fn):
        u = np.clip(
            np.asarray(cdf_fn(y, **params), dtype=np.float64), 1e-12, 1.0 - 1e-12
        )
        norm = NormalDist()
        return np.array([norm.inv_cdf(float(p)) for p in u], dtype=np.float64)

    raise ValueError("object does not provide rqres or family.p for quantile residuals")


def get_rqres(
    obj: GAMLSSModel,
    howmany: int = 10,
    order: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """R reference: ``gamlss/R/rqresplot_new.R::get.rqres``.

    Returns an ``(n_eff, howmany)`` matrix of randomized quantile residuals,
    expanding integer frequency weights as R does.
    """
    _require_gamlss(obj)
    howmany_int = int(howmany)
    if howmany_int < 1:
        raise ValueError("howmany must be at least 1")

    y, params = _expand_rqres_inputs(obj)
    rng = np.random.default_rng(seed)
    rs = np.empty((y.size, howmany_int), dtype=np.float64)
    for col in range(howmany_int):
        draw = _draw_one_rqres(obj, y, params, rng)
        rs[:, col] = np.sort(draw) if order else draw
    return rs


def rqres_plot(
    obj: GAMLSSModel,
    howmany: int = 6,
    plot_type: str = "few",
    type: str = "wp",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    seed: int | None = None,
    plot: bool = False,
) -> RQResPlotResult:
    """R reference: ``gamlss/R/rqresplot_new.R::rqres.plot``.

    Computes multiple realizations of randomized quantile residuals and builds
    QQ or worm-plot diagnostic data (with optional matplotlib rendering).
    """
    _require_gamlss(obj)
    mode = str(plot_type)
    if mode not in {"few", "all"}:
        raise ValueError("plot_type must be 'few' or 'all'")
    kind = str(type)
    if kind not in {"wp", "QQ"}:
        raise ValueError("type must be 'wp' or 'QQ'")
    howmany_int = int(howmany)
    if howmany_int < 1:
        raise ValueError("howmany must be at least 1")
    if mode == "few" and howmany_int > 8:
        raise ValueError("howmany for 'few' should be less than or equal to 8")

    rs = get_rqres(obj, howmany=howmany_int, order=True, seed=seed)
    n = rs.shape[0]
    if n == 0:
        raise ValueError("need at least one observation for rqres_plot")

    rmedian = np.median(rs, axis=1)
    norm = NormalDist()
    theoretical = np.array(
        [norm.inv_cdf(float(p)) for p in _r_ppoints(n)], dtype=np.float64
    )
    sample_deviations = rs - theoretical[:, None]
    median_deviation = rmedian - theoretical

    # R lines 113-118: 95% confidence band on grid z = seq(-5, 5, 0.25).
    # Use cdf * sf to avoid 1 - p cancellation in the extreme tails (z = +/-5).
    from scipy.stats import norm as scipy_norm

    band_z = np.arange(-5.0, 5.0 + 1e-9, 0.25, dtype=np.float64)
    p_band = scipy_norm.cdf(band_z)
    q_band = scipy_norm.sf(band_z)
    d_band = scipy_norm.pdf(band_z)
    se_band = (1.0 / np.maximum(d_band, np.finfo(np.float64).eps)) * np.sqrt(
        p_band * q_band / float(n)
    )
    z_crit = float(scipy_norm.ppf(0.025))
    band_low = z_crit * se_band
    band_high = -band_low

    figure = None
    if plot:
        try:
            import matplotlib.pyplot as plt

            if mode == "all":
                fig, ax = plt.subplots()
                if kind == "QQ":
                    for col in range(howmany_int):
                        ax.scatter(theoretical, rs[:, col], color="lightgray", s=10)
                    ax.scatter(theoretical, rmedian, color="black", s=12)
                    ax.plot(rmedian, rmedian, color="red", linewidth=0.8)
                    ax.set_xlabel("Theoretical Quantiles")
                    ax.set_ylabel("Sample Quantiles")
                else:
                    for col in range(howmany_int):
                        ax.scatter(
                            theoretical,
                            sample_deviations[:, col],
                            color="lightgray",
                            s=10,
                        )
                    ax.scatter(theoretical, median_deviation, color="black", s=12)
                    ax.plot(band_z, band_low, linestyle="--", color="black")
                    ax.plot(band_z, band_high, linestyle="--", color="black")
                    ax.axhline(0.0, linestyle="--", color="gray")
                    ax.set_xlabel("Unit normal quantile")
                    ax.set_ylabel("Deviation")
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                figure = fig
            else:
                nrows = int(np.ceil(howmany_int / 2.0))
                fig, axes = plt.subplots(nrows, 2, squeeze=False)
                for col in range(howmany_int):
                    ax = axes[col // 2][col % 2]
                    if kind == "QQ":
                        ax.scatter(theoretical, rs[:, col], color="darkgreen", s=10)
                        ax.plot(rs[:, col], rs[:, col], color="red", linewidth=0.8)
                    else:
                        ax.scatter(
                            theoretical,
                            sample_deviations[:, col],
                            color="darkgreen",
                            s=10,
                        )
                        ax.plot(band_z, band_low, linestyle="--", color="black")
                        ax.plot(band_z, band_high, linestyle="--", color="black")
                        ax.axhline(0.0, linestyle="--", color="gray")
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    if ylim is not None:
                        ax.set_ylim(ylim)
                figure = fig
        except ImportError:
            figure = None

    return RQResPlotResult(
        samples=rs,
        rmedian=rmedian,
        theoretical=theoretical,
        sample_deviations=sample_deviations,
        median_deviation=median_deviation,
        band_z=band_z,
        band_low=band_low,
        band_high=band_high,
        plot_type=mode,
        type=kind,
        howmany=howmany_int,
        figure=figure,
    )


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
    probs = np.array(
        [NormalDist().cdf(float(value)) for value in theoretical], dtype=np.float64
    )
    se = (
        1.0
        / np.maximum(
            np.exp(-0.5 * theoretical**2) / np.sqrt(2.0 * np.pi),
            np.finfo(np.float64).eps,
        )
    ) * np.sqrt(probs * (1.0 - probs) / max(n, 1))
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
        index_x = (
            np.arange(1, residx.size + 1, dtype=np.float64)
            if xvar is None
            else np.asarray(xvar, dtype=np.float64)
        )
    else:
        fittedvalues = np.repeat(base_fitted, weights.astype(int))
        index_x = (
            np.arange(1, residx.size + 1, dtype=np.float64)
            if xvar is None
            else np.repeat(np.asarray(xvar, dtype=np.float64), weights.astype(int))
        )

    density_x, density_y = _kernel_density(residx)
    qq = qq_stats(object)
    filliben = (
        float(np.corrcoef(qq.ordered, qq.theoretical)[0, 1])
        if qq.ordered.size > 1
        else float("nan")
    )

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
    "RQResPlotResult",
    "RQResSamplesResult",
    "WormPlotResult",
    "get_rqres",
    "get_rqres_samples",
    "plot_diagnostics",
    "plot_gamlss",
    "qq_stats",
    "rqres_plot",
    "worm_plot_data",
]
