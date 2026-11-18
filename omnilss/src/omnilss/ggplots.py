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
    from .momentSK import momentSK

    rows = []
    rng = np.random.default_rng(seed)
    for sample in samples:
        n = sample.size
        if n < 4:
            raise ValueError("moment_bucket needs at least four values")
        moments = momentSK(sample)
        skew = moments.skewness
        kurt = moments.excess_kurtosis
        band = _bucket_band(skew, n)
        polygon = _ci95_polygon(n)
        cloud = []
        for _ in range(int(no_bootstrap)):
            draw = rng.choice(sample, size=n, replace=True)
            boot = momentSK(draw)
            # R's bucket cloud lives on the transformed SK coordinates.
            cloud.append((boot.trans_skew, boot.trans_kurt))
        rows.append(
            {
                "skewness": skew,
                "kurtosis": kurt,
                "trans_skew": moments.trans_skew,
                "trans_kurt": moments.trans_kurt,
                "jarque_bera": moments.jarque_bera,
                "band": band,
                "polygon": polygon,
                "bootstrap": np.asarray(cloud) if cloud else np.zeros((0, 2)),
            }
        )
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        for row in rows:
            axes.scatter(row["trans_skew"], row["trans_kurt"], color="red")
            axes.fill(row["polygon"]["X"], row["polygon"]["Y"], alpha=0.2, color="gray")
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




# ---------------------------------------------------------------------------
# Parameter/residual diagnostics (D2.258)
# ---------------------------------------------------------------------------


def _weight_mask(model: Any) -> np.ndarray:
    residuals = np.asarray(model.residuals, dtype=np.float64)
    weights = getattr(model, "weights", None)
    if weights is None:
        return np.ones(residuals.size, dtype=bool)
    return np.asarray(weights, dtype=np.float64) != 0


def _masked_residuals(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """R get_residuals()/get_weights(): observations keep their 1-based index."""

    residuals = np.asarray(model.residuals, dtype=np.float64)
    observations = np.arange(1, residuals.size + 1)
    keep = _weight_mask(model)
    return observations[keep], residuals[keep]


def _fitted_parameter(model: Any, name: str) -> np.ndarray:
    if name not in model.fitted_values:
        raise ValueError(f"parameter {name!r} is not fitted in this model")
    return np.asarray(model.fitted_values[name], dtype=np.float64)


def _fitted_quantile(model: Any, quantile: float, newdata: Any = None) -> np.ndarray:
    from .prediction import predict_params

    if newdata is None:
        params = {
            name: np.asarray(model.fitted_values[name], dtype=np.float64)
            for name in model.family.parameters
            if name in model.fitted_values
        }
    else:
        params = predict_params(model, dict(newdata))
    args = [np.asarray(params[name], dtype=np.float64) for name in model.family.parameters
            if name in params]
    return np.asarray(model.family.q(float(quantile), *args), dtype=np.float64)


def _response_vector(model: Any) -> np.ndarray:
    return np.asarray(model.y, dtype=np.float64)


def _outlier_frame(observations: np.ndarray, values: np.ndarray, value: float,
                   x: np.ndarray) -> pd.DataFrame:
    flagged = (values >= value) | (values <= -value)
    return pd.DataFrame(
        {
            "obs": observations,
            "sdres": values,
            "fv": x,
            "color": np.where(flagged, "outlier", "normal"),
            "txt": np.where(flagged, observations, np.nan),
        }
    )


def _residual_diagnostic(
    kind: str,
    model: Any,
    values: np.ndarray,
    fitted: np.ndarray,
    value: float,
    plot: bool,
    title: str | None,
    xlab: str,
    ylab: str = "Quantile Residuals",
) -> Any:
    observations, residuals = _masked_residuals(model)
    fitted_values = fitted[_weight_mask(model)]
    frame = _outlier_frame(observations, residuals, value, fitted_values)
    outliers = frame.loc[frame["color"] == "outlier", ["obs", "sdres"]].rename(
        columns={"obs": "observation", "sdres": "quan_resid"}
    )
    title = title or f"{kind} of model"
    if not plot:
        return {"data": frame, "outliers": outliers, "threshold": float(value)}
    figure, axes = _figure()
    if axes is not None:
        axes.vlines(frame["fv"], 0.0, frame["sdres"], color="steelblue")
        axes.scatter(frame["fv"], frame["sdres"], facecolors="none", edgecolors="steelblue")
        axes.axhline(0.0, color="gray")
        for line in (value, -value):
            axes.axhline(line, color="red")
        axes.set_xlabel(xlab)
        axes.set_ylabel(ylab)
        axes.set_title(title)
    return GGPlot(
        kind,
        {"data": frame, "outliers": outliers, "threshold": float(value)},
        figure,
    )


def resid_mu(
    obj: Any = None,
    resid: Any = None,
    plot: bool = True,
    value: float = 2,
    title: str | None = None,
    annotate: bool = True,
) -> Any:
    """R resid_mu: quantile residuals against the fitted mu."""

    if obj is None:
        raise ValueError("A GAMLSS fitted object should be used")
    _require_model(obj, "resid_mu")
    return _residual_diagnostic(
        "resid_mu", obj, np.asarray(obj.residuals, float),
        _fitted_parameter(obj, "mu"), value, plot, title, "mu fitted values",
    )


def resid_median(
    obj: Any = None,
    resid: Any = None,
    plot: bool = True,
    value: float = 3,
    title: str | None = None,
    annotate: bool = True,
) -> Any:
    """R resid_median: quantile residuals against the fitted median."""

    if obj is None:
        raise ValueError("A GAMLSS fitted object should be used")
    _require_model(obj, "resid_median")
    return _residual_diagnostic(
        "resid_median", obj, np.asarray(obj.residuals, float),
        _fitted_quantile(obj, 0.5), value, plot, title, "mu fitted values",
    )


def resid_param(
    obj: Any = None,
    param: str = "mu",
    title: str | None = None,
    line_col: str = "darkred",
    point_col: str = "steelblue",
    point_shape: int = 20,
) -> Any:
    """R resid_param: quantile residuals against one fitted parameter."""

    _require_model(obj, "resid_param")
    if param not in obj.family.parameters:
        raise ValueError(f"parameter {param!r} is not in the family")
    observations, residuals = _masked_residuals(obj)
    fitted = _fitted_parameter(obj, param)[_weight_mask(obj)]
    frame = pd.DataFrame({"obs": observations, "resid": residuals, "fv": fitted})
    correlation = float(np.corrcoef(frame["resid"], frame["fv"])[0, 1])
    text = title or f"r = {correlation:.3f}"
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(frame["fv"], frame["resid"], s=12, color=point_col)
        axes.set_xlabel(param)
        axes.set_ylabel("residuals")
        axes.set_title(text)
    return GGPlot(
        "resid_param", {"data": frame, "correlation": correlation, "param": param}, figure
    )


def resid_quantile(
    obj: Any = None,
    quantile: float = 0.5,
    title: str | None = None,
    newdata: Any = None,
    line_col: str = "darkred",
    point_col: str = "steelblue",
    point_shape: int = 20,
) -> Any:
    """R resid_quantile: residuals against one fitted quantile."""

    _require_model(obj, "resid_quantile")
    observations, residuals = _masked_residuals(obj)
    fitted = _fitted_quantile(obj, quantile, newdata)
    if newdata is None:
        fitted = fitted[_weight_mask(obj)]
    frame = pd.DataFrame({"obs": observations, "y": residuals, "fv": fitted})
    correlation = float(np.corrcoef(frame["y"], frame["fv"])[0, 1])
    text = title or f"r = {correlation:.3f}"
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(frame["fv"], frame["y"], s=12, color=point_col)
        axes.set_xlabel(f"{quantile} quantile")
        axes.set_ylabel("response")
        axes.set_title(text)
    return GGPlot(
        "resid_quantile",
        {"data": frame, "correlation": correlation, "quantile": float(quantile)},
        figure,
    )


def resid_xvar(
    obj: Any = None,
    xvar: Any = None,
    plot: bool = True,
    value: float = 2,
    title: str | None = None,
    annotate: bool = True,
) -> Any:
    """R resid_xvar: quantile residuals against one covariate."""

    _require_model(obj, "resid_xvar")
    if xvar is None:
        raise ValueError("the xvar has to be set")
    values = dict((obj.call or {}).get("data") or {})
    if isinstance(xvar, str):
        if xvar not in values:
            raise ValueError(f"covariate {xvar!r} is not in the data")
        column = np.asarray(values[xvar])
        xname = xvar
    else:
        column = np.asarray(xvar)
        xname = "xvar"
    observations, residuals = _masked_residuals(obj)
    keep = _weight_mask(obj)
    column = column[keep]
    frame = _outlier_frame(observations, residuals, value, _fitted_parameter(obj, "mu")[keep])
    frame[xname] = column
    is_factor = column.dtype.kind not in "iuf"
    outliers = frame.loc[frame["color"] == "outlier", ["obs", "sdres"]].rename(
        columns={"obs": "observation", "sdres": "quan_resid"}
    )
    text = title or f"Residuals against {xname}"
    if not plot:
        return {"data": frame, "outliers": outliers, "threshold": float(value),
                "xname": xname, "is_factor": bool(is_factor)}
    figure, axes = _figure()
    if axes is not None:
        if is_factor:
            categories = sorted({str(item) for item in column})
            positions = [categories.index(str(item)) for item in column]
            axes.scatter(positions, frame["sdres"], s=6, alpha=0.9)
            axes.set_xticks(range(len(categories)))
            axes.set_xticklabels(categories)
        else:
            axes.scatter(column, frame["sdres"], s=6, color="steelblue")
        axes.axhline(0.0, color="gray")
        axes.axhline(value, color="red")
        axes.axhline(-value, color="red")
        axes.set_xlabel(xname)
        axes.set_ylabel("Quantile Residuals")
        axes.set_title(text)
    return GGPlot(
        "resid_xvar",
        {"data": frame, "outliers": outliers, "threshold": float(value),
         "xname": xname, "is_factor": bool(is_factor)},
        figure,
    )


def resid_wp(
    obj: Any = None,
    resid: Any = None,
    value: float = 3,
    points_col: str = "steelblue",
    poly_col: str = "darkred",
    check_overlap: bool = True,
    title: str | None = None,
    ylim: float | None = None,
) -> Any:
    """R resid_wp: worm plot with a cubic polynomial through the ordinates."""

    if obj is None and resid is None:
        raise ValueError("A GAMLSS fitted object or the argument resid should be used")
    if obj is not None:
        _require_model(obj, "resid_wp")
        observations, residuals = _masked_residuals(obj)
    else:
        residuals = np.asarray(resid, dtype=np.float64)
        residuals = residuals[np.isfinite(residuals)]
        observations = np.arange(1, residuals.size + 1)
    x = stats.norm.ppf(_ppoints(residuals.size))[np.argsort(np.argsort(residuals))]
    deviation = residuals - x
    coefficients = np.polyfit(x, deviation, 3)
    fitted = np.polyval(coefficients, x)
    frame = pd.DataFrame(
        {
            "obs": observations,
            "rqres": deviation,
            "x": x,
            "fv": fitted,
            "color": np.where(np.abs(residuals) >= value, "outlier", "normal"),
            "txt": np.where(np.abs(residuals) >= value, observations, np.nan),
        }
    )
    band = _worm_band(float(np.max(np.abs(x))) + 0.5, frame.shape[0], 0.95)
    text = title or "Worm-plot for model"
    figure, axes = _figure()
    if axes is not None:
        axes.fill_between(band["z"], band["low"], band["high"], alpha=0.1)
        axes.scatter(frame["x"], frame["rqres"], s=8, color=points_col, alpha=0.8)
        axes.plot(frame["x"], frame["fv"], color=poly_col)
        for line in (band["low"], band["high"]):
            axes.plot(band["z"], line, linestyle="--", color="gray")
        axes.axhline(0.0, color="gray")
        axes.axvline(0.0, color="gray")
        axes.set_xlabel("Unit normal quantile")
        axes.set_ylabel("Deviation")
        axes.set_title(text)
    return GGPlot("resid_wp", {"data": frame, "band": band}, figure)


def resid_wp_wrap(
    obj: Any = None,
    resid: Any = None,
    value: float = 3,
    xvar: Any = None,
    n_inter: int = 4,
    points_col: str = "steelblue",
    poly_col: str = "darkred",
    alpha_bound: float = 0.1,
    check_overlap: bool = True,
    title: str | None = None,
    ylim: float | None = None,
) -> Any:
    """R resid_wp_wrap: worm plots faceted by intervals of one covariate."""

    if xvar is None:
        raise ValueError("resid_wp_wrap() expects one xvar")
    if obj is not None:
        _require_model(obj, "resid_wp_wrap")
        residuals = np.asarray(obj.residuals, dtype=np.float64)
        values = dict((obj.call or {}).get("data") or {})
        if isinstance(xvar, str):
            if xvar not in values:
                raise ValueError(f"covariate {xvar!r} is not in the data")
            column = np.asarray(values[xvar])
        else:
            column = np.asarray(xvar)
    else:
        residuals = np.asarray(resid, dtype=np.float64)
        column = np.asarray(xvar)
    if column.size != residuals.size:
        raise ValueError("xvar and the residuals must have the same length")
    if column.dtype.kind in "iuf":
        groups = np.asarray(
            pd.qcut(np.asarray(column, float), int(n_inter), duplicates="drop").astype(str)
        )
    else:
        groups = np.asarray(column).astype(str)
    frames = []
    bands = []
    for level in sorted(set(groups.tolist())):
        mask = groups == level
        if not np.any(mask):
            continue
        subset = residuals[mask]
        subset = subset[np.isfinite(subset)]
        x = stats.norm.ppf(_ppoints(subset.size))[np.argsort(np.argsort(subset))]
        deviation = subset - x
        fitted = np.polyval(np.polyfit(x, deviation, 3), x)
        frames.append(
            pd.DataFrame(
                {
                    "rqres": deviation,
                    "x": x,
                    "fv": fitted,
                    "Z": level,
                    "color": np.where(np.abs(subset) >= value, "outlier", "normal"),
                }
            )
        )
        band = _worm_band(float(np.max(np.abs(x))) + 0.5, int(mask.sum()), 0.95)
        band["Z"] = level
        bands.append(band)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    band = pd.concat(bands, ignore_index=True) if bands else pd.DataFrame()
    text = title or "worm-plots of residuals"
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(data["x"], data["rqres"], s=8, color=points_col, alpha=0.8)
        axes.axhline(0.0, color="gray")
        axes.axvline(0.0, color="gray")
        axes.set_title(text)
    return GGPlot("resid_wp_wrap", {"data": data, "band": band, "n_inter": int(n_inter)}, figure)


def _np_bands(values: np.ndarray, kind: str, conf_level: str) -> pd.DataFrame:
    """R's NPOwen.bands / NPJagerWellner.bands (SMIR NPL bands)."""

    counts = pd.Series(np.asarray(values, dtype=np.float64)).value_counts().sort_index()
    yi = counts.index.to_numpy(dtype=np.float64)
    cn = np.cumsum(counts.to_numpy(dtype=np.float64))
    nn = np.full(cn.size, float(np.sum(counts.to_numpy(dtype=np.float64)) + 1.0))
    p = cn / nn
    log_nn = np.log(nn)
    if kind == "Owen":
        if conf_level == "95":
            near = 3.0123 + 0.4835 * log_nn - 0.00957 * log_nn**2 - 0.001488 * log_nn**3
            far = 3.0806 + 0.4894 * log_nn - 0.02086 * log_nn**2
        else:
            near = -4.626 - 0.541 * log_nn + 0.0242 * log_nn**2
            far = -4.71 - 0.512 * log_nn + 0.0219 * log_nn**2
    else:
        if conf_level == "95":
            near = 3.6792 + 0.5720 * log_nn - 0.0567 * log_nn**2 + 0.0027 * log_nn**3
            far = 3.7752 + 0.5062 * log_nn - 0.0417 * log_nn**2 + 0.0016 * log_nn**3
        else:
            near = 5.3318 + 0.5539 * log_nn - 0.00370 * log_nn**2
            far = 5.6392 + 0.4018 * log_nn - 0.0183 * log_nn**2
    scale = np.sqrt(2.0 * np.where(nn <= 100, near, far))
    phi = stats.beta.cdf(p, 1.0 / 3.0, 1.0 / 3.0)
    se = 1.0 / (5.3 * np.sqrt(nn * np.power(p * (1.0 - p), 1.0 / 3.0)))
    upper = np.clip(phi + scale * se, 0.0, 1.0)
    lower = np.clip(phi - scale * se, 0.0, 1.0)
    return pd.DataFrame(
        {
            "x": yi,
            "lower": stats.beta.ppf(lower, 1.0 / 3.0, 1.0 / 3.0),
            "upper": stats.beta.ppf(upper, 1.0 / 3.0, 1.0 / 3.0),
        }
    )


def resid_dtop(
    obj: Any = None,
    resid: Any = None,
    type: str = "Owen",
    conf_level: str = "95",
    value: float = 2,
    points_col: str = "steelblue",
    check_overlap: bool = True,
    title: str | None = None,
) -> Any:
    """R resid_dtop: detrended Owen plot with SMIR NPL confidence bands."""

    if str(type) not in {"Owen", "JW"}:
        raise ValueError('type must be "Owen" or "JW"')
    if str(conf_level) not in {"95", "99"}:
        raise ValueError('conf.level must be "95" or "99"')
    if obj is None and resid is None:
        raise ValueError("A GAMLSS fitted object or the argument resid should be used")
    if obj is not None:
        _require_model(obj, "resid_dtop")
        observations, residuals = _masked_residuals(obj)
    else:
        residuals = np.asarray(resid, dtype=np.float64)
        keep = np.isfinite(residuals)
        residuals = residuals[keep]
        observations = np.arange(1, residuals.size + 1)
    # R ECDF(): fraction of the sample at or below each residual.
    ordered = np.sort(residuals)
    mm = np.searchsorted(ordered, residuals, side="right") / residuals.size
    mm = np.clip(mm, 1e-300, 1.0 - 1e-16)
    zscore = stats.norm.ppf(mm)
    deviation = zscore - residuals
    order = np.argsort(residuals)
    frame = pd.DataFrame(
        {
            "obs": observations[order],
            "rqres": residuals[order],
            "dzscores": deviation[order],
            "color": np.where(np.abs(residuals[order]) >= value, "outlier", "normal"),
            "txt": np.where(np.abs(residuals[order]) >= value, observations[order], np.nan),
        }
    )
    band = _np_bands(residuals, str(type), str(conf_level))
    band["lower"] = stats.norm.ppf(np.clip(band["lower"], 1e-12, 1 - 1e-12)) - band["x"]
    band["upper"] = stats.norm.ppf(np.clip(band["upper"], 1e-12, 1 - 1e-12)) - band["x"]
    text = title or "Owen-plot for model"
    figure, axes = _figure()
    if axes is not None:
        axes.fill_between(band["x"], band["lower"], band["upper"], alpha=0.2)
        axes.scatter(frame["rqres"], frame["dzscores"], s=8, color=points_col)
        axes.axhline(0.0, color="gray")
        axes.set_xlabel("ordered norm. quan. residuals")
        axes.set_ylabel(f"{conf_level}% confidence intervals")
        axes.set_title(text)
    return GGPlot(
        "resid_dtop",
        {"data": frame, "band": band, "type": str(type), "conf_level": str(conf_level)},
        figure,
    )


def _response_scatter(
    kind: str,
    model: Any,
    fitted: np.ndarray,
    xlab: str,
    title: str | None,
    diagonal: bool,
) -> GGPlot:
    observations, _ = _masked_residuals(model)
    keep = _weight_mask(model)
    response = _response_vector(model)[keep]
    frame = pd.DataFrame({"obs": observations, "y": response, "fv": fitted[keep]})
    correlation = float(np.corrcoef(frame["y"], frame["fv"])[0, 1])
    text = title or f"r = {correlation:.3f}"
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(frame["fv"], frame["y"], s=12, color="steelblue")
        if diagonal:
            limits = [float(np.min(frame[["y", "fv"]].min())), float(np.max(frame[["y", "fv"]].max()))]
            axes.plot(limits, limits, color="darkred")
        axes.set_xlabel(xlab)
        axes.set_ylabel("response")
        axes.set_title(text)
    return GGPlot(kind, {"data": frame, "correlation": correlation}, figure)


def resp_mu(obj: Any = None, title: str | None = None, **kwargs: Any) -> GGPlot:
    """R resp_mu: response against fitted mu with the identity line."""

    _require_model(obj, "resp_mu")
    return _response_scatter(
        "resp_mu", obj, _fitted_parameter(obj, "mu"), "mu fitted values", title, True
    )


def resp_param(obj: Any = None, param: str = "mu", title: str | None = None,
               **kwargs: Any) -> GGPlot:
    """R resp_param: response against one fitted parameter."""

    _require_model(obj, "resp_param")
    if param not in obj.family.parameters:
        raise ValueError(f"parameter {param!r} is not in the family")
    return _response_scatter(
        "resp_param", obj, _fitted_parameter(obj, param), param, title, False
    )


def resp_quantile(
    obj: Any = None,
    quantile: float = 0.5,
    title: str | None = None,
    newdata: Any = None,
    **kwargs: Any,
) -> GGPlot:
    """R resp_quantile: response against one fitted quantile."""

    _require_model(obj, "resp_quantile")
    fitted = _fitted_quantile(obj, quantile, newdata)
    result = _response_scatter(
        "resp_quantile", obj, fitted, f"{quantile} quantile", title, False
    )
    data = dict(result.data)
    data["quantile"] = float(quantile)
    return GGPlot("resp_quantile", data, result.figure)


def fitted_leverage(
    obj: Any = None,
    plot: bool = True,
    title: str | None = None,
    quan_val: float = 0.99,
    annotate: bool = True,
    line_col: str = "steelblue",
    point_col: str = "steelblue",
    annot_col: str = "darkred",
) -> Any:
    """R fitted_leverage: linear-model leverage of the mu design."""

    _require_model(obj, "fitted_leverage")
    from .formula_parser import build_design_matrix, parse_formula

    formula = (obj.call or {}).get("formula")
    values = dict((obj.call or {}).get("data") or {})
    if not formula or not values:
        raise ValueError("the model needs a stored formula and data")
    design, _smooth = build_design_matrix(parse_formula(str(formula)), values)
    design = np.asarray(design, dtype=np.float64)
    weights = np.asarray(obj.weights, dtype=np.float64)
    root_w = np.sqrt(weights)
    weighted = design * root_w[:, None]
    gram = weighted.T @ weighted
    leverage = np.einsum("ij,jk,ik->i", weighted, np.linalg.pinv(gram), weighted)
    threshold = float(np.quantile(leverage, float(quan_val)))
    observations = np.arange(1, leverage.size + 1)
    keep = weights != 0
    frame = pd.DataFrame(
        {
            "obs": observations[keep],
            "hat": leverage[keep],
            "color": np.where(leverage[keep] >= threshold, "outlier", "normal"),
        }
    )
    frame["txt"] = np.where(frame["color"] == "outlier", frame["obs"], np.nan)
    rn2 = 2.0 * design.shape[1] / design.shape[0]
    text = title or "Linear leverage of model"
    if not plot:
        return {
            "leverage": frame["hat"].to_numpy(),
            "index": frame["obs"].to_numpy(),
            "threshold": threshold,
            "rn2": float(rn2),
            "data": frame,
        }
    figure, axes = _figure()
    if axes is not None:
        axes.vlines(frame["obs"], 0.0, frame["hat"], color=line_col)
        axes.scatter(frame["obs"], frame["hat"], facecolors="none", edgecolors=point_col)
        axes.set_xlabel("Observation number")
        axes.set_ylabel("linear leverage")
        axes.set_title(text)
    return GGPlot(
        "fitted_leverage",
        {
            "data": frame,
            "leverage": frame["hat"].to_numpy(),
            "index": frame["obs"].to_numpy(),
            "threshold": threshold,
            "rn2": float(rn2),
        },
        figure,
    )


def fitted_devianceIncr(
    obj: Any = None,
    plot: bool = True,
    title: str | None = None,
    quan_val: float = 0.99,
    annotate: bool = True,
    line_col: str = "steelblue",
    point_col: str = "darkblue",
    annot_col: str = "white",
    newdata: Any = None,
) -> Any:
    """R fitted_devianceIncr: per-observation deviance increments."""

    _require_model(obj, "fitted_devianceIncr")
    from .DevianceIncr import devianceIncr

    increments = np.asarray(devianceIncr(obj, newdata=newdata), dtype=np.float64)
    threshold = float(np.quantile(increments, float(quan_val)))
    observations = np.arange(1, increments.size + 1)
    if newdata is None:
        keep = _weight_mask(obj)
        observations = observations[keep]
        increments = increments[keep]
    frame = pd.DataFrame(
        {
            "obs": observations,
            "DIncr": increments,
            "color": np.where(increments >= threshold, "outlier", "normal"),
        }
    )
    frame["txt"] = np.where(frame["color"] == "outlier", frame["obs"], np.nan)
    text = title or "Deviance increment of model"
    if not plot:
        return {"data": frame, "threshold": threshold,
                "outliers": frame.loc[frame["color"] == "outlier", ["obs", "DIncr"]]}
    figure, axes = _figure()
    if axes is not None:
        axes.vlines(frame["obs"], 0.0, frame["DIncr"], color=line_col)
        axes.scatter(frame["obs"], frame["DIncr"], s=10, color=point_col)
        axes.set_xlabel("Observation number")
        axes.set_ylabel("Deviance increment")
        axes.set_title(text)
    return GGPlot(
        "fitted_devianceIncr",
        {"data": frame, "threshold": threshold,
         "outliers": frame.loc[frame["color"] == "outlier", ["obs", "DIncr"]]},
        figure,
    )


def fitted_survival(
    model: Any = None,
    obs: Any = 1,
    title: str | None = None,
    from_: float = 0,
    to: float = 10,
    no_points: int = 201,
    alpha: float = 1,
    lower_tail: bool = True,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R fitted_survival: fitted survival curves at selected observations."""

    _require_model(model, "fitted_survival")
    observations = np.atleast_1d(np.asarray(obs, dtype=int)) - 1
    if np.any(observations < 0):
        raise ValueError("obs must be 1-based observation indices")
    fixed = [name for name in model.family.parameters if name not in model.fitted_values]
    if fixed:
        raise NotImplementedError(
            "fitted_survival's binomial branch (fixed bd) is not ported"
        )
    discrete = str(
        getattr(model.family, "type", getattr(model.family, "type_", ""))
    ).lower() == "discrete"
    grid = (
        np.arange(float(from_), float(to) + 1.0, 1.0)
        if discrete
        else np.linspace(float(from_), float(to), int(no_points))
    )
    curves = {}
    for index in observations:
        args = [
            np.full(grid.size, float(np.asarray(model.fitted_values[name], float)[index]))
            for name in model.family.parameters
        ]
        cdf = np.asarray(model.family.p(grid, *args), dtype=np.float64)
        curves[str(index + 1)] = 1.0 - cdf if lower_tail else cdf
    text = title or "Fitted survival functions from model"
    if not plot:
        return {"grid": grid, "survival": curves, "obs": observations + 1,
                "discrete": bool(discrete)}
    figure, axes = _figure()
    if axes is not None:
        for curve in curves.values():
            if discrete:
                axes.step(grid, curve, where="post")
            else:
                axes.plot(grid, curve)
        axes.set_xlabel("y")
        axes.set_ylabel(f"{model.family.name}(y)")
        axes.set_title(text)
    return GGPlot(
        "fitted_survival",
        {"grid": grid, "survival": curves, "obs": observations + 1,
         "discrete": bool(discrete)},
        figure,
    )


def fitted_survival_data(
    model: Any = None,
    obs: Any = 1,
    from_: float = 0,
    to: float = 10,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R fitted_survival_data: survival curves plus vertical response lines."""

    base = fitted_survival(model, obs=obs, from_=from_, to=to, plot=False, **kwargs)
    observations = np.atleast_1d(np.asarray(obs, dtype=int)) - 1
    observed = _response_vector(model)[observations]
    data = dict(base)
    data["observed"] = observed
    if not plot:
        return data
    figure, axes = _figure()
    if axes is not None:
        for curve in data["survival"].values():
            if data["discrete"]:
                axes.step(data["grid"], curve, where="post")
            else:
                axes.plot(data["grid"], curve)
        for value in observed:
            axes.axvline(float(value), linestyle="--")
        axes.set_xlabel("y")
        axes.set_ylabel(f"{model.family.name}(y)")
    return GGPlot("fitted_survival_data", data, figure)


def _centile_frame(
    model: Any,
    x_values: np.ndarray,
    percentages: np.ndarray,
    params: Mapping[str, np.ndarray],
    y_values: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for percentage in percentages:
        args = [np.asarray(params[name], dtype=np.float64)
                for name in model.family.parameters if name in params]
        label = str(int(percentage)) if float(percentage).is_integer() else str(percentage)
        frames.append(
            pd.DataFrame(
                {
                    "c": np.asarray(model.family.q(percentage / 100.0, *args), dtype=np.float64),
                    "x": x_values,
                    "y": y_values,
                    "centiles": label,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _centile_x_setup(model: Any, xvar: Any) -> tuple[str, np.ndarray]:
    values = dict((model.call or {}).get("data") or {})
    if not values:
        raise ValueError("data are required in the original fit or the xvar")
    if isinstance(xvar, str):
        if xvar not in values:
            raise ValueError(f"covariate {xvar!r} is not in the data")
        return xvar, np.asarray(values[xvar], dtype=np.float64)
    if xvar is not None:
        return "xvar", np.asarray(xvar, dtype=np.float64)
    covariates = _covariate_names(model)
    if len(covariates) != 1:
        raise ValueError("fitted_centiles is design for one x only")
    return covariates[0], np.asarray(values[covariates[0]], dtype=np.float64)


def fitted_centiles(
    obj: Any = None,
    cent: Any = (99.4, 98, 90, 75, 50, 25, 10, 2, 0.4),
    xvar: Any = None,
    plot: bool = True,
    title: str | None = None,
) -> Any:
    """R fitted_centiles: centile curves at the observed covariate values."""

    _require_model(obj, "fitted_centiles")
    percentages = np.asarray(cent, dtype=np.float64)
    x_name, x_values = _centile_x_setup(obj, xvar)
    order = np.argsort(x_values)
    params = {
        name: np.asarray(obj.fitted_values[name], dtype=np.float64)[order]
        for name in obj.family.parameters
        if name in obj.fitted_values
    }
    data = _centile_frame(
        obj, x_values[order], percentages, params,
        _response_vector(obj)[order],
    )
    text = title or f"Centile curves using {obj.family.name}"
    if not plot:
        return data
    figure, axes = _figure()
    if axes is not None:
        for label, group in data.groupby("centiles", sort=False):
            axes.plot(group["x"], group["c"], linewidth=0.8)
        axes.set_xlabel(x_name)
        axes.set_ylabel("centiles")
        axes.set_title(text)
    return GGPlot(
        "fitted_centiles",
        {"data": data, "cent": percentages, "x": x_name},
        figure,
    )


def fitted_centiles_legend(
    obj: Any = None,
    cent: Any = (99.4, 98, 90, 75, 50, 25, 10, 2, 0.4),
    xvar: Any = None,
    save_data: bool = False,
    plot: bool = True,
    title: str | None = None,
) -> Any:
    """R fitted_centiles_legend: fitted_centiles with save.data support."""

    result = fitted_centiles(obj, cent=cent, xvar=xvar, plot=plot, title=title)
    if isinstance(result, pd.DataFrame):
        return result
    if save_data:
        return result.data["data"]
    return GGPlot("fitted_centiles_legend", result.data, result.figure)


def fitted_centiles_gap(
    obj: Any = None,
    cent: Any = (98, 90, 75, 50, 25, 10, 2),
    xvar: Any = None,
    plot: bool = True,
    title: str | None = None,
) -> Any:
    """R fitted_centiles_gap: centile curves on an evenly spaced x grid.

    R predicts at data.frame(ga = ooxvar), so this branch only makes sense for
    a single-covariate model; other covariates would have to be invented.
    """

    _require_model(obj, "fitted_centiles_gap")
    from .prediction import predict_params

    percentages = np.asarray(cent, dtype=np.float64)
    x_name, x_values = _centile_x_setup(obj, xvar)
    covariates = _covariate_names(obj)
    if len(covariates) > 1:
        raise NotImplementedError(
            "fitted_centiles_gap only supports a single-covariate model"
        )
    grid = np.linspace(float(x_values.min()), float(x_values.max()), x_values.size)
    params = predict_params(obj, {x_name: grid})
    # R recycles the response ordered by x across the centile blocks.
    ordered_response = _response_vector(obj)[np.argsort(x_values)]
    data = _centile_frame(obj, grid, percentages, params, ordered_response)
    text = title or f"Centile curves using {obj.family.name}"
    if not plot:
        return data
    figure, axes = _figure()
    if axes is not None:
        for label, group in data.groupby("centiles", sort=False):
            axes.plot(group["x"], group["c"], linewidth=0.8)
        axes.set_xlabel(x_name)
        axes.set_ylabel("centiles")
        axes.set_title(text)
    return GGPlot(
        "fitted_centiles_gap", {"data": data, "cent": percentages, "x": x_name}, figure
    )


def fitted_terms(
    object: Any = None,
    what: str = "mu",
    parameter: str | None = None,
    data: Any = None,
    terms: Any = None,
    plot: bool = True,
    **kwargs: Any,
) -> dict:
    """R fitted_terms: term-level partial effects (an alias of pe_terms in R)."""

    chosen = parameter or (what if isinstance(what, str) else "mu")
    kwargs.pop("type", None)
    result = pe_terms(object, terms=terms, data=data, parameter=chosen, plot=plot, **kwargs)
    return {
        name: GGPlot("fitted_terms", chart.data, chart.figure)
        for name, chart in result.items()
    }




# ---------------------------------------------------------------------------
# Bucket-plot style variants (D2.259)
# ---------------------------------------------------------------------------


def _ci95_polygon(n: int) -> pd.DataFrame:
    """R's CI95/CI95.2/CI95.3/CI95.4 outline in transformed SK coordinates."""

    xx = np.linspace(0.0, 0.99, 101)

    def ci95(values: np.ndarray) -> np.ndarray:
        gamma1 = values / (1.0 - values)
        gamma2 = np.sqrt(
            (24.0 / n) * (stats.chi2.ppf(0.95, 2) - (n / 6.0) * gamma1**2)
        )
        return gamma2 / (1.0 + np.abs(gamma2))

    y1 = ci95(xx)
    y2 = ci95(-xx)
    y3 = -ci95(-xx)
    y4 = -ci95(xx)
    x_out = np.concatenate([xx, xx[::-1], -xx, -xx[::-1]])
    y_out = np.concatenate([y1, y4[::-1], y3, y2[::-1]])
    return pd.DataFrame({"X": x_out, "Y": y_out}).dropna()


def _moment_boundary() -> pd.DataFrame:
    """R boundaryf(): the theoretical skewness/kurtosis boundary."""

    t = np.linspace(0.0, 0.99999, 1000)
    skew = t / (1.0 - t)
    kurt = 1.0 + skew**2
    excess = kurt - 3.0
    return pd.DataFrame(
        {"tskew": t, "tkurt": excess / (1.0 + np.abs(excess))}
    )


@dataclass(frozen=True)
class BucketSpec:
    """A bucket-plot layer set (R returns ggplot layers here).

    The theoretical boundary is data and is returned; R's reference
    distribution curves come from the package's extdata (MomentSkewKurt1.RData /
    CentileSkewKurt.Rdata), which is not vendored in this repository, so the
    reference field stays None rather than being invented.
    """

    kind: str
    variant: str
    legend: bool
    line_width: float
    type: str | None
    boundary: pd.DataFrame | None
    reference: Any = None
    note: str = ""


def _bucket_spec(kind: str, variant: str, legend: bool, line_width: float,
                 type: str | None = None) -> BucketSpec:
    if kind == "moment":
        return BucketSpec(
            kind=kind,
            variant=variant,
            legend=bool(legend),
            line_width=float(line_width),
            type=None,
            boundary=_moment_boundary(),
            note=(
                "R loads the reference distribution curves from "
                "extdata/MomentSkewKurt1.RData, which is not vendored here; "
                "only the theoretical boundary is returned."
            ),
        )
    if type not in ("tail", "central"):
        raise ValueError('type must be "tail" or "central"')
    return BucketSpec(
        kind=kind,
        variant=variant,
        legend=bool(legend),
        line_width=float(line_width),
        type=str(type),
        boundary=None,
        note=(
            "R loads the centile reference curves from "
            "extdata/CentileSkewKurt.Rdata, which is not vendored here."
        ),
    )


def moment_colour_half(legend: bool = True) -> BucketSpec:
    """R moment_colour_half: coloured moment-bucket background, one half."""

    return _bucket_spec("moment", "colour_half", legend, 0.5)


def moment_gray_half(legend: bool = False) -> BucketSpec:
    """R moment_gray_half: grey moment-bucket background, one half."""

    return _bucket_spec("moment", "gray_half", legend, 0.5)


def moment_colour_both(legend: bool = True, line_width: float = 1) -> BucketSpec:
    """R moment_colour_both: coloured moment-bucket background, both halves."""

    return _bucket_spec("moment", "colour_both", legend, line_width)


def moment_gray_both(line_width: float = 1) -> BucketSpec:
    """R moment_gray_both: grey moment-bucket background, both halves."""

    return _bucket_spec("moment", "gray_both", False, line_width)


def centile_colour_half(
    type: str = "tail", legend: bool = True, line_width: float = 1
) -> BucketSpec:
    """R centile_colour_half: coloured centile-bucket background, one half."""

    return _bucket_spec("centile", "colour_half", legend, line_width, type)


def centile_colour_both(
    type: str = "tail", legend: bool = True, line_width: float = 1
) -> BucketSpec:
    """R centile_colour_both: coloured centile-bucket background, both halves."""

    return _bucket_spec("centile", "colour_both", legend, line_width, type)


def centile_gray_both(
    type: str = "tail", legend: bool = True, line_width: float = 0.5
) -> BucketSpec:
    """R centile_gray_both: grey centile-bucket background, both halves."""

    return _bucket_spec("centile", "gray_both", legend, line_width, type)


def _bucket_series(objects: Sequence[Any], text_to_show: Any) -> tuple[list[str], list[np.ndarray]]:
    if not objects:
        raise ValueError("at least one numeric vector or model is required")
    names = (
        [f"model_{index + 1}" for index in range(len(objects))]
        if text_to_show is None
        else ([text_to_show] if isinstance(text_to_show, str) else list(text_to_show))
    )
    if len(names) != len(objects):
        raise ValueError("text_to_show must have one entry per object")
    samples = []
    for item in objects:
        if hasattr(item, "residuals"):
            values = np.asarray(item.residuals, dtype=np.float64)
            weights = getattr(item, "weights", None)
            if weights is not None:
                values = values[np.asarray(weights, dtype=np.float64) != 0]
            samples.append(values[np.isfinite(values)])
        else:
            values = np.asarray(item, dtype=np.float64).ravel()
            samples.append(values[np.isfinite(values)])
    size = min(sample.size for sample in samples)
    return [str(name) for name in names], [sample[:size] for sample in samples]


def _bucket_groups(xvar: Any, n_inter: int) -> np.ndarray:
    column = np.asarray(xvar)
    if column.dtype.kind in "iuf":
        return np.asarray(
            pd.qcut(np.asarray(column, dtype=np.float64), int(n_inter), duplicates="drop").astype(str)
        )
    return column.astype(str)


def _palette(count: int) -> list[str]:
    base = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    return [base[index % len(base)] for index in range(count)]


def moment_bucket_wrap(
    x: Any = None,
    *others: Any,
    weights: Any = None,
    xvar: Any = None,
    n_inter: int = 4,
    no_bootstrap: int = 99,
    seed: int = 123,
    text_to_show: Any = None,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R moment_bucket_wrap: moment bucket per interval of one covariate."""

    from .momentSK import momentSK

    names, samples = _bucket_series((x, *others), text_to_show)
    if xvar is None:
        raise ValueError("moment_buckets_wrap() expects one xvar")
    groups = _bucket_groups(xvar, n_inter)
    if groups.size != samples[0].size:
        raise ValueError("xvar and the data must have the same length")
    colours = _palette(len(names))
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    point_rows = []
    band_rows = []
    for level in sorted(set(groups.tolist())):
        mask = groups == level
        subset = int(mask.sum())
        if subset < 4:
            continue
        for index, (label, sample) in enumerate(zip(names, samples)):
            values = sample[mask]
            for _ in range(int(no_bootstrap)):
                draw = values[rng.integers(0, subset, size=subset)]
                boot = momentSK(draw)
                bootstrap_rows.append(
                    (boot.trans_skew, boot.trans_kurt, label, colours[index], level)
                )
            point = momentSK(values)
            point_rows.append(
                (point.trans_skew, point.trans_kurt, label, level)
            )
        polygon = _ci95_polygon(subset)
        for row in polygon.itertuples(index=False):
            band_rows.append((row.X, row.Y, level))
    bootstrap = pd.DataFrame(
        bootstrap_rows, columns=["X", "Y", "Model", "Col", "Z"]
    )
    points = pd.DataFrame(point_rows, columns=["X", "Y", "Model", "Z"])
    band = pd.DataFrame(band_rows, columns=["X", "Y", "Z"])
    text = "moment buckets by covariate interval"
    figure, axes = _figure()
    if axes is not None:
        axes.scatter(points["X"], points["Y"], color="red")
    return GGPlot(
        "moment_bucket_wrap",
        {
            "bootstrap": bootstrap,
            "points": points,
            "band": band,
            "n_inter": int(n_inter),
            "spec": moment_gray_both(line_width=0.3),
        },
        figure,
    )


def _centile_simulation(n: int, n_sim: int = 1000, seed: int = 1234) -> pd.DataFrame:
    """R centSim: centileSK of n-sim normal samples."""

    from .centileSK import centileSK

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_sim)):
        stats_ = centileSK(rng.normal(size=int(n)))
        rows.append(
            (stats_.s025, stats_.s001, stats_.k001, stats_.excess_k001, stats_.trans_k001)
        )
    return pd.DataFrame(rows, columns=["Sc", "St", "K", "Ke", "Kt"])


def _confidence_ellipse(
    x_values: np.ndarray, y_values: np.ndarray, level: float = 0.95, n_points: int = 100
) -> pd.DataFrame:
    """Covariance ellipse scaled by sqrt(qchisq(level, 2)), like ellipse::ellipse."""

    matrix = np.vstack([np.asarray(x_values, float), np.asarray(y_values, float)])
    covariance = np.cov(matrix)
    centre = matrix.mean(axis=1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    radius = np.sqrt(stats.chi2.ppf(level, 2))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points))
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    points = vectors @ np.diag(np.sqrt(values)) @ circle * radius + centre[:, None]
    return pd.DataFrame({"x": points[0], "y": points[1]})


def _centile_bucket_frame(
    names: Sequence[str],
    samples: Sequence[np.ndarray],
    groups: np.ndarray | None,
    type: str,
    no_bootstrap: int,
    seed: int,
    sim_test: bool,
    no_sim_test: int,
    seed_test: int,
) -> dict:
    from .centileSK import centileSK

    colours = _palette(len(names))
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    point_rows = []
    simulation_rows = []
    categories = [None] if groups is None else sorted(set(groups.tolist()))
    for level in categories:
        mask = np.ones(samples[0].size, dtype=bool) if level is None else (groups == level)
        subset = int(mask.sum())
        if subset < 4:
            continue
        for index, (label, sample) in enumerate(zip(names, samples)):
            values = sample[mask]
            for _ in range(int(no_bootstrap)):
                draw = values[rng.integers(0, subset, size=subset)]
                boot = centileSK(draw)
                bootstrap_rows.append(
                    (boot.s025, boot.s001, boot.trans_k001, label, colours[index], level)
                )
            point = centileSK(values)
            point_rows.append(
                (point.s025, point.s001, point.trans_k001, label, level)
            )
        if sim_test:
            simulation = _centile_simulation(subset, no_sim_test, seed_test)
            column = "Sc" if str(type) == "central" else "St"
            ellipse = _confidence_ellipse(
                simulation[column].to_numpy(), simulation["Ke"].to_numpy()
            )
            # R passes Ke through the same (0,1) transform used for centiles.
            ellipse["y"] = ellipse["y"] / (1.0 + np.abs(ellipse["y"]))
            ellipse["Z"] = level
            simulation_rows.extend(
                (row.x, row.y, row.Z) for row in ellipse.itertuples(index=False)
            )
    return {
        "bootstrap": pd.DataFrame(
            bootstrap_rows, columns=["S0.25", "S0.01", "K0.01", "Model", "Col", "Z"]
        ),
        "points": pd.DataFrame(
            point_rows, columns=["S0.25", "S0.01", "K0.01", "Model", "Z"]
        ),
        "simulation": pd.DataFrame(simulation_rows, columns=["x", "y", "Z"]),
        "type": str(type),
    }


def centile_bucket(
    x: Any = None,
    *others: Any,
    type: str = "tail",
    weights: Any = None,
    no_bootstrap: int = 99,
    seed: int = 123,
    text_to_show: Any = None,
    colour_bucket: bool = False,
    line_width: float = 0.5,
    sim_test: bool = False,
    no_sim_test: int = 1000,
    seed_test: int = 1234,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R centile_bucket: centile-based SK bucket plot."""

    if str(type) not in {"tail", "central"}:
        raise ValueError('type must be "tail" or "central"')
    names, samples = _bucket_series((x, *others), text_to_show)
    data = _centile_bucket_frame(
        names, samples, None, type, no_bootstrap, seed, sim_test,
        no_sim_test, seed_test,
    )
    x_column = "S0.25" if str(type) == "central" else "S0.01"
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        points = data["points"]
        axes.scatter(points[x_column], points["K0.01"], color="red")
    return GGPlot(
        "centile_bucket",
        {
            **data,
            "x_column": x_column,
            "spec": (
                centile_colour_both(type=type, legend=False, line_width=line_width)
                if colour_bucket
                else centile_gray_both(type=type, line_width=line_width)
            ),
        },
        figure,
    )


def centile_bucket_wrap(
    x: Any = None,
    *others: Any,
    type: str = "tail",
    weights: Any = None,
    xvar: Any = None,
    n_inter: int = 4,
    no_bootstrap: int = 99,
    seed: int = 123,
    text_to_show: Any = None,
    colour_bucket: bool = False,
    line_width: float = 0.5,
    sim_test: bool = False,
    no_sim_test: int = 1000,
    seed_test: int = 1234,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R centile_bucket_wrap: centile bucket per interval of one covariate."""

    if str(type) not in {"tail", "central"}:
        raise ValueError('type must be "tail" or "central"')
    names, samples = _bucket_series((x, *others), text_to_show)
    if xvar is None:
        raise ValueError("centile_buckets_wrap() expects one xvar")
    groups = _bucket_groups(xvar, n_inter)
    if groups.size != samples[0].size:
        raise ValueError("xvar and the data must have the same length")
    data = _centile_bucket_frame(
        names, samples, groups, type, no_bootstrap, seed, sim_test,
        no_sim_test, seed_test,
    )
    x_column = "S0.25" if str(type) == "central" else "S0.01"
    figure, axes = _figure() if plot else (None, None)
    if axes is not None:
        points = data["points"]
        for level, group in points.groupby("Z", sort=False):
            axes.scatter(group[x_column], group["K0.01"], s=8)
    return GGPlot(
        "centile_bucket_wrap",
        {
            **data,
            "x_column": x_column,
            "n_inter": int(n_inter),
            "spec": (
                centile_colour_both(type=type, legend=False, line_width=line_width)
                if colour_bucket
                else centile_gray_both(type=type, line_width=line_width)
            ),
        },
        figure,
    )




# ---------------------------------------------------------------------------
# predict_pdf / predict_cdf / quantile_gamlss (D2.260)
# ---------------------------------------------------------------------------


def _newdata_parameters(model: Any, newdata: Any, name: str) -> dict[str, np.ndarray]:
    if newdata is None:
        raise ValueError(f"the argument newdata is required for {name}")
    from .prediction import predict_params

    params = predict_params(model, dict(newdata))
    return {key: np.asarray(value, dtype=np.float64).ravel() for key, value in params.items()}


def _prediction_grid(model: Any, from_: float, to: float, no_points: int) -> tuple[np.ndarray, bool]:
    discrete = str(
        getattr(model.family, "type", getattr(model.family, "type_", ""))
    ).lower() == "discrete"
    grid = (
        np.arange(float(from_), float(to) + 1.0, 1.0)
        if discrete
        else np.linspace(float(from_), float(to), int(no_points))
    )
    return grid, discrete


def _density_at_rows(
    model: Any, params: Mapping[str, np.ndarray], grid: np.ndarray, kind: str
) -> dict[str, np.ndarray]:
    """Evaluate d/p per newdata row.

    Fixed parameters (the binomial bd) are taken from the predictions as well:
    predict_params returns them when newdata carries the column, and R's
    predict_pdf passes them to the density in the same way.
    """

    fixed = list(getattr(model.family, "fixed_parameters", ()) or ())
    names = list(model.family.parameters) + fixed
    size = len(next(iter(params.values())))
    curves = {}
    for index in range(size):
        kwargs = {}
        for name in names:
            if name in params:
                kwargs[name] = float(np.asarray(params[name]).ravel()[index])
        for name in fixed:
            if name not in kwargs:
                raise ValueError(
                    f"{kind}: the fixed parameter {name!r} is not in newdata"
                )
        function = model.family.d if kind == "predict_pdf" else model.family.p
        curves[str(index + 1)] = np.asarray(function(grid, **kwargs), dtype=np.float64)
    return curves


def predict_pdf(
    model: Any = None,
    newdata: Any = None,
    title: str | None = None,
    from_: float = 0,
    to: float = 10,
    no_points: int = 201,
    alpha: float = 0.4,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R predict_pdf: fitted densities at the parameters of newdata rows."""

    _require_model(model, "predict_pdf")
    params = _newdata_parameters(model, newdata, "predict_pdf")
    grid, discrete = _prediction_grid(model, from_, to, no_points)
    curves = _density_at_rows(model, params, grid, "predict_pdf")
    text = title or "Fitted pdf's from model"
    data = {"grid": grid, "density": curves, "discrete": bool(discrete),
            "rows": len(curves)}
    if not plot:
        return data
    figure, axes = _figure()
    if axes is not None:
        for curve in curves.values():
            if discrete:
                axes.vlines(grid, 0.0, curve, alpha=alpha)
            else:
                axes.fill_between(grid, 0.0, curve, alpha=alpha)
        axes.set_xlabel("y")
        axes.set_ylabel(f"{model.family.name}(y)")
        axes.set_title(text)
    return GGPlot("predict_pdf", data, figure)


def predict_cdf(
    model: Any = None,
    newdata: Any = None,
    title: str | None = None,
    from_: float = 0,
    to: float = 10,
    no_points: int = 201,
    alpha: float = 0.4,
    lower_tail: bool = True,
    plot: bool = True,
    **kwargs: Any,
) -> Any:
    """R predict_cdf: fitted CDFs at the parameters of newdata rows."""

    _require_model(model, "predict_cdf")
    params = _newdata_parameters(model, newdata, "predict_cdf")
    grid, discrete = _prediction_grid(model, from_, to, no_points)
    curves = _density_at_rows(model, params, grid, "predict_cdf")
    if not lower_tail:
        curves = {key: 1.0 - value for key, value in curves.items()}
    text = title or "Predicted cdf's from model"
    data = {"grid": grid, "cdf": curves, "discrete": bool(discrete),
            "rows": len(curves), "lower_tail": bool(lower_tail)}
    if not plot:
        return data
    figure, axes = _figure()
    if axes is not None:
        for curve in curves.values():
            if discrete:
                axes.step(grid, curve, where="post", alpha=alpha)
            else:
                axes.plot(grid, curve, alpha=alpha)
        axes.set_xlabel("y")
        axes.set_ylabel(f"{model.family.name}(y)")
        axes.set_title(text)
    return GGPlot("predict_cdf", data, figure)


def quantile_gamlss(obj: Any = None, quantile: float = 0.5, newdata: Any = None) -> np.ndarray:
    """R quantile_gamlss: one fitted quantile per observation (or newdata row)."""

    _require_model(obj, "quantile_gamlss")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile should be between 0 and 1")
    return np.asarray(_fitted_quantile(obj, float(quantile), newdata), dtype=np.float64)




# ---------------------------------------------------------------------------
# Bivariate surface helpers (D2.261)
# ---------------------------------------------------------------------------


def _bivar_grid(
    xlim: Sequence[float], ylim: Sequence[float], n: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(n))
    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(n))
    mesh = np.meshgrid(xs, ys, indexing="xy")
    frame = pd.DataFrame({"x": mesh[0].ravel(), "y": mesh[1].ravel()})
    return xs, ys, frame


def _evaluate_bivariate(function: Any, xs: np.ndarray, ys: np.ndarray,
                        vectorised: bool, kwargs: Mapping[str, Any]) -> np.ndarray:
    if vectorised:
        try:
            values = np.asarray(function(xs, ys, **kwargs), dtype=np.float64).ravel()
            if values.size == xs.size:
                return values
        except Exception:  # noqa: BLE001 - fall back to the scalar (mapply) path
            pass
    return np.array(
        [float(function(xi, yi, **kwargs)) for xi, yi in zip(xs, ys)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BivarSurface:
    """Bivariate surface data: grids, the long frame, and an optional figure."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    frame: pd.DataFrame
    level: float | None = None
    mass: float | None = None
    figure: Any = None


def bivar_fun(
    fun: Any = None,
    xlim: Sequence[float] = (-3, 3),
    ylim: Sequence[float] = (-3, 3),
    n: int = 100,
    xlab: str = "x",
    ylab: str = "y",
    plot: bool = True,
    **kwargs: Any,
) -> BivarSurface:
    """R bivar_fun: contour data of a bivariate function on an n x n grid.

    R builds expand.grid(x, y) (x varying fastest) and evaluates fun(x, y, ...)
    vectorised; the port returns the same long frame plus the Z matrix with rows
    by x and columns by y (R's matrix(z, nrow = n)).
    """

    if fun is None or not callable(fun):
        raise ValueError("bivar_fun needs a bivariate function")
    xs, ys, frame = _bivar_grid(xlim, ylim, n)
    values = _evaluate_bivariate(fun, frame["x"].to_numpy(), frame["y"].to_numpy(),
                                 True, kwargs)
    frame["z"] = values
    z = values.reshape(int(n), int(n)).T
    figure = None
    if plot:
        figure, axes = _figure()
        if axes is not None:
            mesh = np.meshgrid(xs, ys, indexing="xy")
            axes.contourf(mesh[0], mesh[1], z.T, levels=12)
            axes.set_xlabel(xlab)
            axes.set_ylabel(ylab)
    return BivarSurface(x=xs, y=ys, z=z, frame=frame, figure=figure)


def bivar_fun_plotly(
    f: Any = None,
    xlim: Sequence[float] = (-3, 3),
    ylim: Sequence[float] = (-3, 3),
    n: int = 100,
    plot: bool = True,
    **kwargs: Any,
) -> BivarSurface:
    """R bivar_fun_plotly: surface data for plotly::add_surface.

    plotly is not a dependency here, so the surface (x, y, Z) is returned and,
    when plot and matplotlib are available, rendered as a 3-D surface instead.
    """

    if f is None or not callable(f):
        raise ValueError("bivar_fun_plotly needs a bivariate function")
    xs, ys, frame = _bivar_grid(xlim, ylim, n)
    values = _evaluate_bivariate(f, frame["x"].to_numpy(), frame["y"].to_numpy(),
                                 True, kwargs)
    frame["z"] = values
    z = values.reshape(int(n), int(n)).T
    figure = None
    if plot:
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt

            figure = plt.figure()
            axes = figure.add_subplot(111, projection="3d")
            mesh = np.meshgrid(xs, ys, indexing="xy")
            axes.plot_surface(mesh[0], mesh[1], z.T, cmap="viridis")
        except Exception:  # pragma: no cover - matplotlib is optional
            figure = None
    return BivarSurface(x=xs, y=ys, z=z, frame=frame, figure=figure)


def bivar_pdf(
    pdf: Any = None,
    xlim: Sequence[float] = (-3, 3),
    ylim: Sequence[float] = (-3, 3),
    n: int = 200,
    xlab: str = "x",
    ylab: str = "y",
    hdv: bool = True,
    plot: bool = True,
    **kwargs: Any,
) -> BivarSurface:
    """R bivar_pdf: bivariate density surface with the 50% HDR contour.

    R evaluates the density with mapply (scalar calls), sorts the grid by
    density, accumulates z * dx * dy, normalises by the maximum (not the total)
    and takes the first cell reaching 0.5 as the median-region threshold.
    """

    if pdf is None or not callable(pdf):
        raise ValueError("bivar_pdf needs a bivariate density function")
    xs, ys, frame = _bivar_grid(xlim, ylim, n)
    values = _evaluate_bivariate(pdf, frame["x"].to_numpy(), frame["y"].to_numpy(),
                                 False, kwargs)
    frame["z"] = values
    z = values.reshape(int(n), int(n)).T
    dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    order = np.argsort(-values, kind="mergesort")
    cumulative = np.cumsum(values[order]) * dx * dy
    maximum = float(cumulative[-1]) if cumulative.size else 0.0
    if maximum > 0:
        cumulative = cumulative / maximum
    index = int(np.argmax(cumulative >= 0.5)) if cumulative.size else 0
    level = float(values[order][index]) if values.size else 0.0
    mass = float(np.sum(values)) * dx * dy
    figure = None
    if plot:
        figure, axes = _figure()
        if axes is not None:
            mesh = np.meshgrid(xs, ys, indexing="xy")
            axes.contourf(mesh[0], mesh[1], z.T, levels=12, alpha=0.8)
            axes.contour(mesh[0], mesh[1], z.T, levels=[level], colors="red")
            axes.set_xlabel(xlab)
            axes.set_ylabel(ylab)
            axes.set_title("50% Probability Region (Median Region)")
    return BivarSurface(
        x=xs, y=ys, z=z, frame=frame, level=level, mass=mass, figure=figure
    )




# ---------------------------------------------------------------------------
# Bootstrap coefficient plots (D2.262)
# ---------------------------------------------------------------------------


def _bootstrap_inputs(x: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not (hasattr(x, "boot") and hasattr(x, "orig_coef")):
        raise ValueError(
            "the function needs NonParametric.Boot or Bayesian.Boot objects"
        )
    boot = np.asarray(x.boot, dtype=np.float64)
    if boot.ndim != 2:
        raise ValueError("x$boot should be a matrix of bootstrap coefficients")
    original = np.asarray(x.orig_coef, dtype=np.float64).ravel()
    names = list(getattr(x, "coef_names", ()) or ())
    if len(names) != boot.shape[1]:
        names = [f"coef_{index + 1}" for index in range(boot.shape[1])]
    return boot, original, names


def _bootstrap_plot_data(
    values: np.ndarray, original: float, name: str
) -> dict[str, Any]:
    quantiles = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "values": values,
        "orig": float(original),
        "quantiles": np.asarray(quantiles, dtype=np.float64),
        "name": name,
    }


def _bootstrap_figure(data: Mapping[str, Any], alpha: float, dens_fill: str,
                      line_col: str) -> Any:
    figure, axes = _figure()
    if axes is None:
        return None
    values = np.asarray(data["values"], dtype=np.float64)
    if values.size >= 3 and float(np.var(values)) > 0.0:
        grid = np.linspace(float(values.min()), float(values.max()), 256)
        axes.plot(grid, stats.gaussian_kde(values)(grid), color=dens_fill)
        axes.fill_between(grid, 0.0, stats.gaussian_kde(values)(grid),
                          alpha=alpha, color=dens_fill)
    else:
        axes.hist(values, bins=min(20, max(3, values.size)), alpha=alpha,
                  color=dens_fill)
    axes.axvline(data["orig"], color="black")
    for value in np.asarray(data["quantiles"]):
        axes.axvline(float(value), color=line_col)
    axes.set_xlabel(data["name"])
    axes.set_ylabel("density")
    return figure


def boot_coef_one(
    x: Any = None,
    par: int = 1,
    rug: bool = True,
    alpha: float = 0.2,
    hist_col: str = "black",
    hist_fill: str = "white",
    line_col: str = "gray",
    dens_fill: str = "#FF6666",
    title: str | None = None,
    plot: bool = True,
) -> Any:
    """R boot_coef_one: density of one bootstrap coefficient."""

    boot, original, names = _bootstrap_inputs(x)
    index = int(par) - 1
    if not 0 <= index < boot.shape[1]:
        raise ValueError("par is outside the bootstrap coefficient columns")
    data = _bootstrap_plot_data(boot[:, index], original[index], names[index])
    data["title"] = title or names[index]
    if not plot:
        return data
    figure = _bootstrap_figure(data, alpha, dens_fill, line_col)
    return GGPlot("boot_coef_one", data, figure)


def boot_coef(
    x: Any = None,
    terms: Any = None,
    hist_col: str = "black",
    hist_fill: str = "white",
    dens_fill: str = "#FF6666",
    alpha: float = 0.2,
    line_col: str = "gray",
    nrow: int | None = None,
    ncol: int | None = None,
    plots_per_page: int = 9,
    one_by_one: bool = False,
    title: str | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """R boot_coef: one density (with the original estimate and 95% limits) per coefficient."""

    boot, original, names = _bootstrap_inputs(x)
    if terms is None:
        selected = list(range(boot.shape[1]))
    else:
        listed = [terms] if isinstance(terms, (str, int)) else list(terms)
        selected = []
        for item in listed:
            if isinstance(item, str):
                if item not in names:
                    raise ValueError(f"term {item!r} is not a bootstrap coefficient")
                selected.append(names.index(item))
            else:
                index = int(item) - 1
                if not 0 <= index < boot.shape[1]:
                    raise ValueError(f"term index {item} is outside the columns")
                selected.append(index)
    charts = {}
    for index in selected:
        data = _bootstrap_plot_data(boot[:, index], original[index], names[index])
        data["title"] = title or names[index]
        figure = _bootstrap_figure(data, alpha, dens_fill, line_col) if plot else None
        charts[names[index]] = GGPlot("boot_coef", data, figure)
    return charts


__all__ = [
    "GGPlot",
    "add_resid_qqplot",
    "fitted_cdf",
    "fitted_cdf_data",
    "fitted_centiles",
    "fitted_centiles_gap",
    "fitted_centiles_legend",
    "fitted_devianceIncr",
    "fitted_leverage",
    "fitted_survival",
    "fitted_survival_data",
    "fitted_terms",
    "fitted_pdf",
    "fitted_pdf_data",
    "model_density",
    "model_qqplot",
    "BivarSurface",
    "BucketSpec",
    "bivar_fun",
    "bivar_fun_plotly",
    "bivar_pdf",
    "boot_coef",
    "boot_coef_one",
    "centile_bucket",
    "centile_bucket_wrap",
    "centile_colour_both",
    "centile_colour_half",
    "centile_gray_both",
    "moment_bucket_wrap",
    "moment_colour_both",
    "moment_colour_half",
    "moment_gray_both",
    "moment_gray_half",
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
    "predict_cdf",
    "predict_pdf",
    "quantile_gamlss",
    "pe_terms",
    "resid_density",
    "resid_dtop",
    "resid_ecdf",
    "resid_median",
    "resid_mu",
    "resid_param",
    "resid_quantile",
    "resid_wp",
    "resid_wp_wrap",
    "resid_xvar",
    "resp_mu",
    "resp_param",
    "resp_quantile",
    "resid_plots",
    "resid_qqplot",
    "y_dots",
    "y_hist",
]
