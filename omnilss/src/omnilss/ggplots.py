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
import pandas as pd
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




# ---------------------------------------------------------------------------
# pe_* partial effects (D2.256)
# ---------------------------------------------------------------------------


def _model_data(model: Any, data: Any, name: str) -> dict:
    if data is not None:
        return dict(data)
    stored = (model.call or {}).get("data")
    if stored is None:
        raise ValueError(f"{name}: data are not defined for the model")
    return dict(stored)


def _scenario_frame(
    model: Any,
    values: Mapping[str, Any],
    term: str,
    grid: np.ndarray,
    scenario: Mapping[str, Any] | None,
    how_scenario: str,
) -> dict[str, np.ndarray]:
    """Hold every other covariate at its scenario value while term varies."""

    scenario = dict(scenario or {})
    frame = {}
    for name, column in values.items():
        array = np.asarray(column)
        if name == term:
            frame[name] = grid
            continue
        if name in scenario:
            frame[name] = np.full(grid.size, scenario[name])
            continue
        if array.dtype.kind in "iuf":
            numeric = np.asarray(array, dtype=np.float64)
            if how_scenario == "last":
                frame[name] = np.full(grid.size, float(numeric[-1]))
            elif how_scenario == "median":
                frame[name] = np.full(grid.size, float(np.median(numeric)))
            else:
                frame[name] = np.full(grid.size, float(np.mean(numeric)))
        else:
            levels = sorted({str(value) for value in array})
            frame[name] = np.full(grid.size, levels[0], dtype=object)
    return frame


def _center_value(grid: np.ndarray, effect: np.ndarray, centered_from: str) -> float:
    if centered_from == "none":
        return 0.0
    if centered_from == "median":
        target = float(np.median(grid))
    else:
        target = float(np.mean(grid))
    closest = int(np.argmin(np.abs(grid - target)))
    return float(effect[closest])


def pe_1_parameter(
    obj: Any = None,
    term: str = None,
    data: Any = None,
    n_points: int = 100,
    parameter: str = "mu",
    type: str = "parameter",
    scenario: Mapping[str, Any] | None = None,
    how_scenario: str = "median",
    centered_from: str = "mean",
    plot: bool = True,
    **kwargs: Any,
) -> GGPlot:
    """R pe_1_parameter: one term's partial effect on a parameter or its eta.

    The covariate is varied over its observed range while every other covariate
    is held at the scenario value (median / last / mean), then the chosen
    parameter is predicted and centred at its mean (default).
    """

    _require_model(obj, "pe_1_parameter")
    if term is None:
        raise ValueError("pe_1_parameter needs a term")
    values = _model_data(obj, data, "pe_1_parameter")
    if term not in values:
        raise ValueError(f"term {term!r} is not in the data")
    if parameter not in obj.family.parameters:
        raise ValueError(f"parameter {parameter!r} is not in the family")
    if str(type) not in {"parameter", "eta"}:
        raise ValueError('type must be "parameter" or "eta"')
    column = np.asarray(values[term])
    if column.dtype.kind in "iuf":
        numeric = np.asarray(column, dtype=np.float64)
        grid = np.linspace(float(numeric.min()), float(numeric.max()), int(n_points))
    else:
        grid = np.asarray(sorted({str(value) for value in column}))
    frame = _scenario_frame(obj, values, term, grid, scenario, how_scenario)
    from .prediction import predict_params

    predicted = np.asarray(predict_params(obj, frame)[parameter], dtype=np.float64)
    if str(type) == "eta":
        link = obj.family.link_functions[parameter]
        import jax.numpy as jnp

        raw = np.asarray(link(jnp.asarray(predicted, dtype=jnp.float64)), dtype=np.float64)
    else:
        raw = predicted
    center = _center_value(grid, raw, str(centered_from))
    effect = raw - center
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        axes.plot(grid, effect, color="darkblue")
    return GGPlot(
        "pe_1_parameter",
        {
            "term": term,
            "x": grid,
            "effect": effect,
            "raw": raw,
            "parameter": parameter,
            "type": str(type),
            "centered_at": center,
            "centered_from": str(centered_from),
        },
        figure,
    )


def pe_param(obj: Any = None, term: str = None, **kwargs: Any) -> GGPlot:
    """R pe_param: partial effect on the parameter scale (type = "parameter")."""

    kwargs.pop("type", None)
    return pe_1_parameter(obj, term, type="parameter", **kwargs)


def pe_eta(obj: Any = None, term: str = None, **kwargs: Any) -> GGPlot:
    """R pe_eta: partial effect on the linear-predictor scale."""

    kwargs.pop("type", None)
    return pe_1_parameter(obj, term, type="eta", **kwargs)


def pe_2_parameter(obj: Any = None, terms: Any = None, **kwargs: Any) -> dict:
    """R pe_2_parameter: partial effects for two (or more) terms."""

    if terms is None:
        raise ValueError("pe_2_parameter needs the terms")
    terms = [terms] if isinstance(terms, str) else list(terms)
    return {
        str(term): pe_1_parameter(obj, str(term), **kwargs) for term in terms
    }


def pe_param_grid(obj: Any = None, terms: Any = None, **kwargs: Any) -> dict:
    """R pe_param_grid: parameter-scale partial effects for several terms."""

    kwargs.pop("type", None)
    return pe_2_parameter(obj, terms, type="parameter", **kwargs)


def pe_eta_grid(obj: Any = None, terms: Any = None, **kwargs: Any) -> dict:
    """R pe_eta_grid: eta-scale partial effects for several terms."""

    kwargs.pop("type", None)
    return pe_2_parameter(obj, terms, type="eta", **kwargs)


def pe_pdf(
    obj: Any = None,
    term: str = None,
    data: Any = None,
    n_points: int = 100,
    parameter: str = "mu",
    points: int = 5,
    no_points: int = 201,
    scenario: Mapping[str, Any] | None = None,
    how_scenario: str = "median",
    plot: bool = True,
    **kwargs: Any,
) -> GGPlot:
    """R pe_pdf: fitted densities at several values of one term."""

    _require_model(obj, "pe_pdf")
    if term is None:
        raise ValueError("pe_pdf needs a term")
    values = _model_data(obj, data, "pe_pdf")
    if term not in values:
        raise ValueError(f"term {term!r} is not in the data")
    column = np.asarray(values[term])
    if column.dtype.kind not in "iuf":
        grid = np.asarray(sorted({str(value) for value in column}))
    else:
        numeric = np.asarray(column, dtype=np.float64)
        grid = np.linspace(float(numeric.min()), float(numeric.max()), int(n_points))
    count = min(int(points), grid.size)
    picks = np.unique(np.linspace(0, grid.size - 1, count).round().astype(int))
    frame = _scenario_frame(obj, values, term, grid[picks], scenario, how_scenario)
    from .prediction import predict_params

    params = predict_params(obj, frame)
    # The density grid spans the observed response range.
    formula = (obj.call or {}).get("formula", "")
    response_name = formula.split("~")[0].strip() if "~" in formula else list(values)[0]
    response = np.asarray(values[response_name], dtype=np.float64)
    density_grid = np.linspace(float(response.min()), float(response.max()), int(no_points))
    curves = {}
    for index, pick in enumerate(picks):
        args = [
            np.full(density_grid.size, np.asarray(params[name], dtype=np.float64)[index])
            for name in obj.family.parameters
        ]
        curves[str(float(grid[pick]) if grid.dtype.kind in "iuf" else grid[pick])] = np.asarray(
            obj.family.d(density_grid, *args), dtype=np.float64
        )
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for curve in curves.values():
            axes.plot(density_grid, curve)
    return GGPlot(
        "pe_pdf",
        {"term": term, "grid": density_grid, "values": grid[picks], "density": curves},
        figure,
    )


def pe_pdf_grid(obj: Any = None, terms: Any = None, **kwargs: Any) -> dict:
    """R pe_pdf_grid: densities along several terms."""

    if terms is None:
        raise ValueError("pe_pdf_grid needs the terms")
    terms = [terms] if isinstance(terms, str) else list(terms)
    return {str(term): pe_pdf(obj, str(term), **kwargs) for term in terms}


def pe_terms(obj: Any = None, terms: Any = None, data: Any = None, **kwargs: Any) -> dict:
    """R pe_terms: partial effects for the model's terms (or a given subset)."""

    _require_model(obj, "pe_terms")
    values = _model_data(obj, data, "pe_terms")
    if terms is None:
        formula = (obj.call or {}).get("formula", "")
        rhs = formula.split("~", 1)[-1] if "~" in formula else ""
        terms = [
            part.strip() for part in rhs.split("+")
            if part.strip() not in {"", "1", "-1"} and part.strip() in values
        ]
    terms = [terms] if isinstance(terms, str) else list(terms)
    if not terms:
        raise ValueError("pe_terms found no covariates to vary")
    return {str(term): pe_1_parameter(obj, str(term), data=values, **kwargs) for term in terms}




# ---------------------------------------------------------------------------
# model_* comparisons and residual diagnostics (D2.257)
# ---------------------------------------------------------------------------


def _gaic_value(model: Any, k: float, corrected: bool) -> float:
    from .operations import _gaic_single

    return _gaic_single(model, k=float(k), c=bool(corrected))


def _model_names(count: int, names: Any) -> list[str]:
    if names is None:
        return [f"model_{index + 1}" for index in range(count)]
    listed = [names] if isinstance(names, str) else list(names)
    if len(listed) != count:
        raise ValueError("names must have one entry per model")
    return [str(name) for name in listed]


def _scaled_table(
    values: np.ndarray,
    labels: list[str],
    diff_limit: float,
    column_name: str = "value",
) -> pd.DataFrame:
    """R's delta/scaled construction: delta relative to the max, cut at diff.dev."""

    values = np.asarray(values, dtype=np.float64)
    # R filters on the distance from the minimum, then reports delta as
    # max - value (so the worst model gets 0 and the best gets the full span).
    distance = values - float(np.nanmin(values))
    kept = distance < float(diff_limit)
    span = float(np.nanmax(values[kept])) - float(np.nanmin(values[kept]))
    if span <= 0:
        span = 1.0
    delta = float(np.nanmax(values[kept])) - values
    scaled = delta / span
    return pd.DataFrame(
        {
            column_name: values[kept],
            "delta": np.round(delta[kept], 3),
            "scaled": np.round(scaled[kept], 4),
        },
        index=[label for label, keep in zip(labels, kept) if keep],
    )


def _comparison_plot(
    kind: str,
    table: pd.DataFrame,
    scale: str,
    col: str,
    horiz: bool,
    title: str,
    plot: bool,
) -> Any:
    if not plot:
        return table
    figure, axes = _figure()
    if axes is not None:
        column = "scaled" if scale == "[0,1]" else "delta"
        positions = np.arange(len(table))
        axes.barh(positions, table[column].to_numpy(), color=col)
        axes.set_yticks(positions)
        axes.set_yticklabels(list(table.index))
        axes.set_title(title)
    return GGPlot(
        kind,
        {"table": table, "scale": scale, "column": "scaled" if scale == "[0,1]" else "delta"},
        figure,
    )


def model_GAIC(
    object: Any = None,
    *others: Any,
    k: float = 2,
    c: bool = False,
    plot: bool = True,
    which: int = 1,
    diff_dev: float = 1000,
    scale: str = "[0,1]",
    col: str = "rosybrown",
    horiz: bool = True,
    names: Any = None,
    title: str | None = None,
) -> Any:
    """R model_GAIC: GAIC comparison table (matrix, model list, or one model).

    A single model returns its GAIC value, exactly as R does; several models (or
    a matrix of candidate GAICs) return the delta/scaled table, and plot=True
    wraps it in a GGPlot.
    """

    if str(scale) not in {"[0,1]", "[max,min]"}:
        raise ValueError('scale must be "[0,1]" or "[max,min]"')
    models = [object, *others]
    is_model = [hasattr(item, "g_dev") and hasattr(item, "df_fit") for item in models]
    if all(is_model):
        if len(models) == 1:
            return _gaic_value(models[0], k, c)
        values = np.asarray([_gaic_value(model, k, c) for model in models], dtype=np.float64)
        labels = _model_names(len(models), names)
        table = _scaled_table(values, labels, diff_dev, column_name="AIC")
        df_by_label = {
            label: float(model.df_fit) for label, model in zip(labels, models)
        }
        table.insert(0, "df", [df_by_label[label] for label in table.index])
        text = title or f"GAIC's in {scale} scale"
        return _comparison_plot("model_GAIC", table, scale, col, horiz, text, plot)

    matrix = np.asarray(object, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("the matrix of GAIC values must be two-dimensional")
    column = int(which) - 1
    if column < 0 or column >= matrix.shape[1]:
        raise ValueError("which is outside the matrix columns")
    selected = matrix[:, column]
    finite = ~np.isnan(selected)
    labels = [str(value) for value in np.asarray(
        getattr(object, "index", np.arange(matrix.shape[0]))
    )[finite]]
    table = _scaled_table(selected[finite], labels, diff_dev, column_name="GAIC")
    text = title or f"GAIC's ordered from {scale}"
    return _comparison_plot("model_GAIC", table, scale, col, horiz, text, plot)


def model_GAIC_lollipop(object: Any = None, *others: Any, **kwargs: Any) -> Any:
    """R model_GAIC_lollipop: the GAIC table drawn as lollipops."""

    kwargs.setdefault("col", "skyblue")
    result = model_GAIC(object, *others, **kwargs)
    if isinstance(result, GGPlot):
        return GGPlot("model_GAIC_lollipop", result.data, result.figure)
    return result


def model_TD(
    *models: Any,
    newdata: Mapping[str, Any] | None = None,
    plot: bool = True,
    diff_TD: float = 1000,
    scale: str = "[0,1]",
    col: str = "rosybrown",
    horiz: bool = True,
    names: Any = None,
    title: str | None = None,
) -> Any:
    """R model_TD: test global deviance on newdata for each model."""

    from .gamlssVGD_23_12_21 import get_tgd

    if str(scale) not in {"[0,1]", "[min,max]"}:
        raise ValueError('scale must be "[0,1]" or "[min,max]"')
    if len(models) == 1:
        raise ValueError("The function needs more than one model")
    if newdata is None:
        raise ValueError("The newdata option is needed (you can use the training set)")
    for model in models:
        if not (hasattr(model, "family") and hasattr(model, "df_fit")):
            raise ValueError("some of the objects are not gamlss")
    values = np.asarray(
        [float(get_tgd(model, newdata=dict(newdata)).test_global_deviance) for model in models],
        dtype=np.float64,
    )
    table = _scaled_table(values, _model_names(len(models), names), diff_TD, column_name="TD")
    text = title or f"Test Deviance's ordered from {scale}"
    return _comparison_plot("model_TD", table, scale, col, horiz, text, plot)


def model_TD_lollipop(*models: Any, **kwargs: Any) -> Any:
    """R model_TD_lollipop: the test-deviance table as lollipops."""

    kwargs.setdefault("col", "skyblue")
    result = model_TD(*models, **kwargs)
    if isinstance(result, GGPlot):
        return GGPlot("model_TD_lollipop", result.data, result.figure)
    return result


def _response_name(model: Any) -> str:
    from .formula_parser import parse_formula

    formula = (model.call or {}).get("formula")
    if not formula:
        raise ValueError("the model call does not record a formula")
    return parse_formula(str(formula)).response


def _covariate_names(model: Any) -> list[str]:
    formula = (model.call or {}).get("formula")
    if not formula:
        return []
    rhs = str(formula).split("~", 1)[-1]
    return [
        part.strip()
        for part in rhs.split("+")
        if part.strip() not in {"", "1", "-1"}
    ]


def model_centiles(
    obj: Any = None,
    *others: Any,
    cent: Any = (97, 90, 75, 50, 25, 10, 3),
    xvar: Any = None,
    in_one: bool = False,
    plot: bool = True,
    names: Any = None,
    title: str | None = None,
) -> Any:
    """R model_centiles: centile curves for one or more models.

    The x covariate is taken from xvar (or from the model's formula/data) and the
    curves are evaluated at the fitted parameters of each observation, ordered by
    x, exactly as fitted_centiles_legend does with save.data = TRUE.
    """

    models = [obj, *others]
    for model in models:
        _require_model(model, "model_centiles")
    percentages = np.asarray(cent, dtype=np.float64)
    labels = _model_names(len(models), names)
    x_name = None
    if xvar is not None:
        x_name = xvar if isinstance(xvar, str) else None
    frames = []
    for label, model in zip(labels, models):
        values = dict((model.call or {}).get("data") or {})
        if not values:
            raise ValueError("data are required in the original fit or the xvar")
        response = _response_name(model)
        if x_name is None:
            covariates = _covariate_names(model)
            if len(covariates) != 1:
                raise ValueError("fitted_centiles is design for one x only")
            x_name = covariates[0]
        x_values = np.asarray(values[x_name], dtype=np.float64)
        order = np.argsort(x_values)
        params = {
            name: np.asarray(model.fitted_values[name], dtype=np.float64)[order]
            for name in model.family.parameters
            if name in model.fitted_values
        }
        for percentage in percentages:
            args = [params[name] for name in model.family.parameters if name in params]
            frame = pd.DataFrame(
                {
                    "c": np.asarray(model.family.q(percentage / 100.0, *args), dtype=np.float64),
                    "x": x_values[order],
                    "y": np.asarray(values[response], dtype=np.float64)[order],
                    "centiles": str(int(percentage)) if float(percentage).is_integer() else str(percentage),
                    "Y": label,
                }
            )
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if not plot:
        return data
    figure, axes = _figure()
    if axes is not None:
        for (label, centile), group in data.groupby(["Y", "centiles"], sort=False):
            axes.plot(group["x"], group["c"], linewidth=0.7)
    return GGPlot(
        "model_centiles",
        {"data": data, "cent": percentages, "x": x_name, "in_one": bool(in_one)},
        figure,
    )


def model_wp(
    obj: Any = None,
    *others: Any,
    level: float = 0.95,
    plot: bool = True,
    names: Any = None,
    title: str | None = None,
) -> Any:
    """R model_wp: worm plot (deviation against the unit normal quantile)."""

    models = [obj, *others]
    for model in models:
        _require_model(model, "model_wp")
    if len(models) < 2:
        raise ValueError("you need more than two models")
    labels = _model_names(len(models), names)
    frames = []
    for label, model in zip(labels, models):
        residuals = np.asarray(model.residuals, dtype=np.float64)
        weights = getattr(model, "weights", None)
        observations = np.arange(1, residuals.size + 1)
        if weights is not None:
            keep = np.asarray(weights, dtype=np.float64) != 0
            residuals = residuals[keep]
            observations = observations[keep]
        finite = np.isfinite(residuals)
        residuals = residuals[finite]
        observations = observations[finite]
        x = stats.norm.ppf(_ppoints(residuals.size))[np.argsort(np.argsort(residuals))]
        frames.append(
            pd.DataFrame(
                {"obs": observations, "rqres": residuals - x, "x": x, "model": label}
            )
        )
    data = pd.concat(frames, ignore_index=True)
    band = _worm_band(float(np.max(np.abs(data["x"]))) + 0.5, data.shape[0], level)
    text = title or "worm-plots of residuals from different models"
    if not plot:
        return {"data": data, "band": band}
    figure, axes = _figure()
    if axes is not None:
        axes.fill_between(band["z"], band["low"], band["high"], alpha=0.1)
        for label, group in data.groupby("model", sort=False):
            axes.scatter(group["x"], group["rqres"], s=8, alpha=0.8, label=label)
        axes.axhline(0.0, color="gray")
        axes.axvline(0.0, color="gray")
        axes.set_title(text)
    return GGPlot("model_wp", {"data": data, "band": band}, figure)


def _worm_band(xlim: float, n: int, level: float) -> pd.DataFrame:
    dz = 0.25
    z = np.arange(-xlim, xlim + dz / 2, dz)
    p = stats.norm.cdf(z)
    se = (1.0 / stats.norm.pdf(z)) * np.sqrt(p * (1.0 - p) / max(n, 1))
    quantile = stats.norm.ppf((1.0 - level) / 2.0)
    low = quantile * se
    return pd.DataFrame({"high": -low, "low": low, "z": z})


def model_wp_wrap(
    obj: Any = None,
    *others: Any,
    xvar: Any = None,
    n_inter: int = 4,
    level: float = 0.95,
    plot: bool = True,
    names: Any = None,
    title: str | None = None,
) -> Any:
    """R model_wp_wrap: worm plots faceted by intervals of one covariate."""

    models = [obj, *others]
    for model in models:
        _require_model(model, "model_wp_wrap")
    if len(models) < 2:
        raise ValueError("you need more than two models")
    if xvar is None:
        raise ValueError("worm plot wrap() expects one xvar")
    values = dict((models[0].call or {}).get("data") or {})
    if values and isinstance(xvar, str) and xvar in values:
        column = np.asarray(values[xvar])
    else:
        column = np.asarray(xvar)
    if column.size == 0:
        raise ValueError("xvar must not be empty")
    if column.dtype.kind in "iuf":
        bins = pd.qcut(np.asarray(column, dtype=np.float64), int(n_inter), duplicates="drop")
        groups = np.asarray(bins.astype(str))
    else:
        groups = np.asarray(column).astype(str)
    labels = _model_names(len(models), names)
    frames = []
    for level_value in sorted(set(groups.tolist())):
        mask = groups == level_value
        if not np.any(mask):
            continue
        subset_size = int(mask.sum())
        for label, model in zip(labels, models):
            residuals = np.asarray(model.residuals, dtype=np.float64)[mask]
            weights = getattr(model, "weights", None)
            keep = np.ones(residuals.size, dtype=bool)
            if weights is not None:
                keep = np.asarray(weights, dtype=np.float64)[mask] != 0
            residuals = residuals[keep]
            residuals = residuals[np.isfinite(residuals)]
            x = stats.norm.ppf(_ppoints(residuals.size))[np.argsort(np.argsort(residuals))]
            frames.append(
                pd.DataFrame(
                    {"rqres": residuals - x, "x": x, "model": label, "Z": level_value}
                )
            )
        band = _worm_band(float(np.max(np.abs(frames[-1]["x"]))) + 0.5, subset_size, level)
        band["Z"] = level_value
        frames.append(band.assign(kind="band"))
    data = pd.concat([frame for frame in frames if "model" in frame], ignore_index=True)
    bands = pd.concat([frame for frame in frames if "kind" in frame], ignore_index=True)
    text = title or "worm-plots of residuals from different models"
    if not plot:
        return {"data": data, "band": bands}
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(data["x"], data["rqres"], s=8, alpha=0.8)
        axes.axhline(0.0, color="gray")
        axes.axvline(0.0, color="gray")
        axes.set_title(text)
    return GGPlot("model_wp_wrap", {"data": data, "band": bands}, figure)


def model_pca(obj: Any = None, *others: Any, scale: bool = True, plot: bool = True,
              names: Any = None) -> Any:
    """R model_pca: principal components of the residual matrix of several models."""

    models = [obj, *others]
    for model in models:
        _require_model(model, "model_pca")
    if len(models) < 2:
        raise ValueError("you need more than two models")
    labels = _model_names(len(models), names)
    columns = []
    for model in models:
        residuals = np.asarray(model.residuals, dtype=np.float64)
        weights = getattr(model, "weights", None)
        if weights is not None:
            residuals = residuals[np.asarray(weights, dtype=np.float64) != 0]
        columns.append(residuals[np.isfinite(residuals)])
    size = min(column.size for column in columns)
    matrix = np.column_stack([column[:size] for column in columns])
    centered = matrix - matrix.mean(axis=0)
    if scale:
        spread = centered.std(axis=0, ddof=1)
        spread[spread == 0] = 1.0
        centered = centered / spread
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u * s
    rotation = vt.T
    sdev = s / np.sqrt(max(size - 1, 1))
    arrows = pd.DataFrame(
        {
            "PC1": rotation[:, 0] * sdev[0],
            "PC2": rotation[:, 1] * sdev[1],
            "label": labels,
        }
    )
    if not plot:
        return {"scores": scores[:, :2], "arrows": arrows, "sdev": sdev}
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(scores[:, 0], scores[:, 1], s=6, color="grey", alpha=0.75)
        axes.axhline(0.0, linewidth=0.25)
        axes.axvline(0.0, linewidth=0.25)
        for row in arrows.itertuples(index=False):
            axes.arrow(0.0, 0.0, row.PC1, row.PC2, length_includes_head=True,
                       head_width=0.02, color="steelblue")
            axes.annotate(row.label, (row.PC1, row.PC2))
        axes.set_title("Residuals PCA")
    return GGPlot(
        "model_pca", {"scores": scores[:, :2], "arrows": arrows, "sdev": sdev}, figure
    )


def model_devianceIncr_diff(
    model1: Any = None,
    model2: Any = None,
    against: str = "index",
    tol: float = 20,
    data: Any = None,
    newdata: Any = None,
    plot: bool = True,
) -> Any:
    """R model_devianceIncr_diff: per-observation deviance-increment difference."""

    for model in (model1, model2):
        _require_model(model, "model_devianceIncr_diff")
    from .DevianceIncr import devianceIncr

    first = np.asarray(devianceIncr(model1, newdata=newdata), dtype=np.float64)
    second = np.asarray(devianceIncr(model2, newdata=newdata), dtype=np.float64)
    if first.size != second.size:
        raise ValueError("The two models should have the same no of obsrvations")
    difference = first - second
    is_outlier = np.abs(difference) >= float(tol)
    index = np.arange(1, first.size + 1)
    values = dict(data) if data is not None else (
        dict((model1.call or {}).get("data") or {}) if (model1.call or {}).get("data") else None
    )
    if values is None:
        raise ValueError("The data argument is needed in obj")
    frame = pd.DataFrame({key: np.asarray(value)[: first.size] for key, value in values.items()})
    frame["diff"] = difference
    frame["col"] = np.where(is_outlier, "outlier", "normal")
    frame["index"] = index
    frame["txt"] = np.where(is_outlier, index, np.nan)
    x_values = frame["index"] if str(against) == "index" else frame[str(against)]
    if not plot:
        return frame
    figure, axes = _figure()
    if axes is not None:
        # R's "steelblue4" has no matplotlib equivalent; use the closest name.
        axes.vlines(x_values, 0.0, difference, color="steelblue")
        axes.scatter(x_values, difference, facecolors="none", edgecolors="steelblue")
        axes.set_xlabel(str(against))
    return GGPlot(
        "model_devianceIncr_diff",
        {"data": frame, "x": np.asarray(x_values), "against": str(against)},
        figure,
    )


def model_mom_bucket(*args: Any, **kwargs: Any) -> Any:
    """R model_mom_bucket: alias of moment_bucket for model residuals."""

    result = moment_bucket(*args, **kwargs)
    return GGPlot("model_mom_bucket", result.data, result.figure)


__all__ = [
    "GGPlot",
    "add_resid_qqplot",
    "fitted_cdf",
    "fitted_cdf_data",
    "fitted_pdf",
    "fitted_pdf_data",
    "model_density",
    "model_qqplot",
    "model_GAIC",
    "model_GAIC_lollipop",
    "model_TD",
    "model_TD_lollipop",
    "model_centiles",
    "model_devianceIncr_diff",
    "model_mom_bucket",
    "model_pca",
    "model_wp",
    "model_wp_wrap",
    "moment_bucket",
    "pe_1_parameter",
    "pe_2_parameter",
    "pe_eta",
    "pe_eta_grid",
    "pe_param",
    "pe_param_grid",
    "pe_pdf",
    "pe_pdf_grid",
    "pe_terms",
    "resid_density",
    "resid_ecdf",
    "resid_plots",
    "resid_qqplot",
    "y_dots",
    "y_hist",
]
