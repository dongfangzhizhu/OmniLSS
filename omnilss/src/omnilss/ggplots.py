# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots, first batch (D2.255).

R source: R/gamlss.ggplots/R/ (y_hist.R, fitted_pdf.R, fitted_cdf.R,
model_qqplot.R, model_density.R, resid_*.R, Bucket_plot.R).

The package is a ggplot2 presentation layer.  The port returns a GGPlot object
carrying the plot's data (histogram edges/counts, QQ coordinates, density grid,
bucket moments, ...) plus a matplotlib figure when matplotlib is installed.
Pixel parity with ggplot2 is explicitly out of scope (DEVELOPMENT_PLAN P4); the
contract is the data layer and the outlier/diagnostic rules, which match R:

    y_hist / y_dots   median +/- value * mad (mad scaled by 1.4826)
    qq plots          qnorm(ppoints(n)) against sorted residuals, qqline through
                      the quartiles
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class GGPlot:
    """A plot reduced to its data plus an optional matplotlib figure."""

    kind: str
    data: Mapping[str, Any]
    figure: Any = None


def _figure():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        return figure, figure.add_subplot(111)
    except Exception:  # pragma: no cover - matplotlib is optional
        return None, None


def _require_model(obj: Any, name: str) -> None:
    if obj is None or not hasattr(obj, "family") or not hasattr(obj, "residuals"):
        raise ValueError(f"{name}: the model is not a gamlss model")


def _residuals(obj: Any) -> np.ndarray:
    values = np.asarray(obj.residuals, dtype=np.float64)
    weights = getattr(obj, "weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        values = values[weights != 0]
    return values[np.isfinite(values)]


def _ppoints(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) - 0.5) / n


def _quantile_line(values: np.ndarray) -> tuple[float, float]:
    quartiles = np.quantile(values, [0.25, 0.75])
    theoretical = stats.norm.ppf([0.25, 0.75])
    slope = float((quartiles[1] - quartiles[0]) / (theoretical[1] - theoretical[0]))
    intercept = float(quartiles[0] - slope * theoretical[0])
    return slope, intercept


def y_hist(y: Any = None, value: float = 3.0, bins: int | None = None, plot: bool = True) -> GGPlot:
    """R y_hist: histogram with median and median +/- value * mad lines."""

    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("y_hist needs finite values")
    if bins is None:
        bins = min(30, max(5, int(np.sqrt(values.size))))
    counts, edges = np.histogram(values, bins=bins, density=False)
    median = float(np.median(values))
    spread = float(stats.median_abs_deviation(values, scale="normal"))
    low = median - value * spread
    high = median + value * spread
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.hist(values, bins=bins, color="white", edgecolor="black")
        for line, style in ((median, "-"), (low, ":"), (high, ":")):
            axes.axvline(line, color="red" if line == median else "black", linestyle=style)
    return GGPlot(
        "hist",
        {
            "counts": counts,
            "edges": edges,
            "median": median,
            "mad": spread,
            "lower": low,
            "upper": high,
            "value": float(value),
        },
        figure,
    )


def y_dots(y: Any = None, value: float = 3.0, plot: bool = True) -> GGPlot:
    """R y_dots: dot plot with the same outlier rule as y_hist."""

    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("y_dots needs finite values")
    median = float(np.median(values))
    spread = float(stats.median_abs_deviation(values, scale="normal"))
    low = median - value * spread
    high = median + value * spread
    outlier = (values <= low) | (values >= high)
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.scatter(values, np.zeros(values.size), s=8, color="gray")
        axes.scatter(values[outlier], np.zeros(int(outlier.sum())), s=14, color="red")
        axes.axvline(median, color="red")
        for line in (low, high):
            axes.axvline(line, color="black", linestyle=":")
    return GGPlot(
        "dots",
        {
            "values": values,
            "outlier": outlier,
            "median": median,
            "mad": spread,
            "lower": low,
            "upper": high,
        },
        figure,
    )


def fitted_pdf(
    model: Any = None,
    obs: Any = 1,
    from_: float = 0.0,
    to: float = 10.0,
    no_points: int = 201,
    plot: bool = True,
) -> GGPlot:
    """R fitted_pdf: fitted density curves at selected observations."""

    _require_model(model, "fitted_pdf")
    grid = np.linspace(from_, to, int(no_points))
    density = {}
    curves = []
    observations = np.atleast_1d(np.asarray(obs, dtype=int))
    for index in range(observations.size):
        args = []
        for name in model.family.parameters:
            if name in model.fitted_values:
                args.append(np.full(grid.size, np.asarray(model.fitted_values[name], float)[observations[index]]))
        curves.append(np.asarray(model.family.d(grid, *args), dtype=np.float64))
        density[str(observations[index] + 1)] = curves[-1]
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for curve in curves:
            axes.plot(grid, curve)
    return GGPlot("fitted_pdf", {"grid": grid, "density": density, "obs": observations + 1}, figure)


def fitted_pdf_data(
    model: Any = None,
    obs: Any = 1,
    from_: float = 0.0,
    to: float = 10.0,
    no_points: int = 201,
    plot: bool = True,
) -> GGPlot:
    """R fitted_pdf_data: the density curves plus the observed values."""

    base = fitted_pdf(model, obs, from_, to, no_points, plot=False)
    observations = np.atleast_1d(np.asarray(obs, dtype=int))
    observed = np.asarray(model.y, dtype=np.float64)[observations]
    data = dict(base.data)
    data["observed"] = observed
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for curve in data["density"].values():
            axes.plot(data["grid"], curve)
        for value in observed:
            axes.axvline(float(value), color="darkred", linestyle="--")
    return GGPlot("fitted_pdf_data", data, figure)


def fitted_cdf(
    model: Any = None,
    obs: Any = 1,
    from_: float = 0.0,
    to: float = 10.0,
    no_points: int = 201,
    plot: bool = True,
) -> GGPlot:
    """R fitted_cdf: fitted CDF curves at selected observations."""

    _require_model(model, "fitted_cdf")
    grid = np.linspace(from_, to, int(no_points))
    observations = np.atleast_1d(np.asarray(obs, dtype=int))
    curves = {}
    for index in range(observations.size):
        args = []
        for name in model.family.parameters:
            if name in model.fitted_values:
                args.append(np.full(grid.size, np.asarray(model.fitted_values[name], float)[observations[index]]))
        curves[str(observations[index] + 1)] = np.asarray(model.family.p(grid, *args), dtype=np.float64)
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for curve in curves.values():
            axes.plot(grid, curve)
    return GGPlot("fitted_cdf", {"grid": grid, "cdf": curves, "obs": observations + 1}, figure)


def fitted_cdf_data(
    model: Any = None,
    obs: Any = 1,
    from_: float = 0.0,
    to: float = 10.0,
    no_points: int = 201,
    plot: bool = True,
) -> GGPlot:
    """R fitted_cdf_data: the CDF curves plus the observed values."""

    base = fitted_cdf(model, obs, from_, to, no_points, plot=False)
    observations = np.atleast_1d(np.asarray(obs, dtype=int))
    observed = np.asarray(model.y, dtype=np.float64)[observations]
    data = dict(base.data)
    data["observed"] = observed
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for curve in data["cdf"].values():
            axes.plot(data["grid"], curve)
        for value in observed:
            axes.axvline(float(value), color="darkred", linestyle="--")
    return GGPlot("fitted_cdf_data", data, figure)


def model_qqplot(obj: Any = None, value: float = 3.0, plot: bool = True) -> GGPlot:
    """R model_qqplot: normal QQ plot of the quantile residuals."""

    _require_model(obj, "model_qqplot")
    residuals = np.sort(_residuals(obj))
    n = residuals.size
    if n < 3:
        raise ValueError("model_qqplot needs at least three residuals")
    theoretical = stats.norm.ppf(_ppoints(n))
    slope, intercept = _quantile_line(residuals)
    outlier = np.abs(residuals) >= value
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.scatter(theoretical, residuals, s=8, color="steelblue")
        axes.plot(theoretical, intercept + slope * theoretical, color="darkgray")
        axes.scatter(theoretical[outlier], residuals[outlier], s=14, color="sienna")
    return GGPlot(
        "qqplot",
        {
            "theoretical": theoretical,
            "sample": residuals,
            "slope": slope,
            "intercept": intercept,
            "outlier": outlier,
            "value": float(value),
        },
        figure,
    )


def resid_qqplot(obj: Any = None, value: float = 3.0, plot: bool = True) -> GGPlot:
    """R resid_qqplot: model_qqplot with the outlier index preserved."""

    base = model_qqplot(obj, value=value, plot=False)
    residuals = np.asarray(obj.residuals, dtype=np.float64)
    weights = getattr(obj, "weights", None)
    observations = np.arange(1, residuals.size + 1)
    if weights is not None:
        keep = np.asarray(weights, dtype=np.float64) != 0
        residuals = residuals[keep]
        observations = observations[keep]
    finite = np.isfinite(residuals)
    data = dict(base.data)
    data["observation"] = observations[finite]
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.scatter(data["theoretical"], data["sample"], s=8, color="steelblue")
        axes.plot(data["theoretical"], data["intercept"] + data["slope"] * data["theoretical"], color="darkgray")
        axes.scatter(
            np.asarray(data["theoretical"])[data["outlier"]],
            np.asarray(data["sample"])[data["outlier"]],
            s=14, color="sienna",
        )
    return GGPlot("resid_qqplot", data, figure)


def add_resid_qqplot(gg: Any = None, obj: Any = None, value: float = 3.0, plot: bool = True) -> GGPlot:
    """R add_resid_qqplot: the QQ layer data for an existing plot."""

    base = resid_qqplot(obj, value=value, plot=plot)
    return GGPlot("add_resid_qqplot", base.data, base.figure)


def model_density(obj: Any = None, *others: Any, plot: bool = True) -> GGPlot:
    """R model_density: residual densities of one or more models."""

    _require_model(obj, "model_density")
    models = [obj, *others]
    samples = [_residuals(model) for model in models]
    grid = np.linspace(
        min(float(np.min(sample)) for sample in samples),
        max(float(np.max(sample)) for sample in samples),
        256,
    )
    densities = {}
    for index, sample in enumerate(samples):
        if sample.size < 3 or float(np.var(sample)) == 0.0:
            densities[str(index + 1)] = np.zeros_like(grid)
            continue
        densities[str(index + 1)] = stats.gaussian_kde(sample)(grid)
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for curve in densities.values():
            axes.plot(grid, curve)
    return GGPlot("model_density", {"grid": grid, "density": densities}, figure)


def resid_density(obj: Any = None, plot: bool = True) -> GGPlot:
    """R resid_density: kernel density of the residuals."""

    _require_model(obj, "resid_density")
    sample = _residuals(obj)
    if sample.size < 3:
        raise ValueError("resid_density needs at least three residuals")
    grid = np.linspace(float(sample.min()), float(sample.max()), 256)
    density = stats.gaussian_kde(sample)(grid) if float(np.var(sample)) > 0 else np.zeros_like(grid)
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.plot(grid, density)
        axes.plot(grid, stats.norm.pdf(grid, sample.mean(), sample.std(ddof=1)), linestyle="--")
    return GGPlot("resid_density", {"grid": grid, "density": density}, figure)


def resid_ecdf(obj: Any = None, plot: bool = True) -> GGPlot:
    """R resid_ecdf: empirical CDF of the residuals."""

    _require_model(obj, "resid_ecdf")
    sample = np.sort(_residuals(obj))
    if sample.size == 0:
        raise ValueError("resid_ecdf needs residuals")
    ecdf = np.arange(1, sample.size + 1) / sample.size
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.step(sample, ecdf, where="post")
    return GGPlot("resid_ecdf", {"values": sample, "ecdf": ecdf}, figure)


def resid_plots(obj: Any = None) -> dict[str, GGPlot]:
    """R resid_plots: the QQ, density and ECDF diagnostics in one call."""

    return {
        "qqplot": resid_qqplot(obj, plot=False),
        "density": resid_density(obj, plot=False),
        "ecdf": resid_ecdf(obj, plot=False),
    }


def moment_bucket(
    *objects: Any,
    no_bootstrap: int = 0,
    seed: int = 123,
    plot: bool = True,
    **kwargs: Any,
) -> GGPlot:
    """R moment_bucket: skewness/kurtosis bucket with the 95% normal band.

    R bootstraps by default (no_bootstrap = 99); the port keeps the parameter
    but defaults to 0 so the call stays cheap, and returns the bootstrap cloud
    when a positive count is requested.
    """

    if not objects:
        raise ValueError("moment_bucket needs at least one numeric vector or model")
    samples = []
    for item in objects:
        if hasattr(item, "residuals"):
            samples.append(_residuals(item))
        else:
            values = np.asarray(item, dtype=np.float64)
            samples.append(values[np.isfinite(values)])
    rows = []
    rng = np.random.default_rng(seed)
    for sample in samples:
        n = sample.size
        if n < 4:
            raise ValueError("moment_bucket needs at least four values")
        skew = float(stats.skew(sample))
        kurt = float(stats.kurtosis(sample, fisher=True))
        band = _bucket_band(skew, n)
        cloud = []
        for _ in range(int(no_bootstrap)):
            draw = rng.choice(sample, size=n, replace=True)
            cloud.append(
                (float(stats.skew(draw)), float(stats.kurtosis(draw, fisher=True)))
            )
        rows.append(
            {
                "skewness": skew,
                "kurtosis": kurt,
                "band": band,
                "bootstrap": np.asarray(cloud) if cloud else np.zeros((0, 2)),
            }
        )
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for row in rows:
            axes.scatter(row["skewness"], row["kurtosis"], color="red")
            if row["band"]:
                axes.fill_between(*zip(*row["band"]), alpha=0.2, color="gray")
    return GGPlot("moment_bucket", {"rows": rows, "no_bootstrap": int(no_bootstrap)}, figure)


def _bucket_band(skew: float, n: int) -> tuple[tuple[float, float], ...]:
    """R CI95/CI95.2 curve: kurtosis band as a function of skewness."""

    xi = np.linspace(0.0, abs(skew) if skew else 0.999, 50)
    gamma1 = xi / np.clip(1.0 - xi, 1e-12, None)
    gamma2 = np.sqrt(
        np.clip((24.0 / n) * (stats.chi2.ppf(0.95, 2) - (n / 6.0) * gamma1**2), 0.0, None)
    )
    gamma2 = gamma2 / (1.0 + np.abs(gamma2))
    return tuple((float(a), float(b)) for a, b in zip(xi, gamma2))


__all__ = [
    "GGPlot",
    "add_resid_qqplot",
    "fitted_cdf",
    "fitted_cdf_data",
    "fitted_pdf",
    "fitted_pdf_data",
    "model_density",
    "model_qqplot",
    "moment_bucket",
    "resid_density",
    "resid_ecdf",
    "resid_plots",
    "resid_qqplot",
    "y_dots",
    "y_hist",
]
