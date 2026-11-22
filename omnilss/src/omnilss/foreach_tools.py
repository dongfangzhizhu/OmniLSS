# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.foreach: quantile regression, rolling fits and bootstrap helpers
(D2.253).

R source: R/gamlss.foreach/R/fitQR.R, FitRollingGamlss.R,
NonParametricBoot_version-5.R, BaysianBoot_version-5.R, centiles.boot.R.

  fitQR        fits SEP3 with sigma/nu/tau fixed (sigma = tau = 1,
               nu = sqrt((1 - q) / q)), which is R's quantile-regression trick.
  fitRolling   refits the model on every rolling window and predicts the next
               observation, recording the deviance increment.
  NonParametricBoot / BayesianBoot
               resample rows (or draw Dirichlet weights) and refit, collecting
               coefficients and fitted parameters.
  centiles.boot
               resample, refit and recompute the centile curves.

The bootstraps are statistical procedures: R's foreach loop is random and
order-dependent, so parity is functional (shapes, reproducibility with a seed,
bootstrap means near the original fit), not bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .prediction import predict_params


def _require_model(obj: Any, name: str) -> None:
    if obj is None or not hasattr(obj, "family") or not hasattr(obj, "call"):
        raise ValueError(f"the object passed to {name} should be a gamlss object")


def _call_data(obj: Any, data: Any, name: str) -> dict:
    if data is not None:
        return dict(data)
    stored = (obj.call or {}).get("data")
    if stored is None:
        raise ValueError(f"data are not defined for {name}")
    return dict(stored)


def _flatten_coefficients(model: Any) -> np.ndarray:
    parts = []
    for name in model.family.estimable_parameters:
        if name in model.coefficients:
            parts.append(
                np.ravel(np.asarray(model.coefficients[name], dtype=np.float64))
            )
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _parameter_formula_kwargs(call: Mapping[str, Any]) -> dict[str, str]:
    """Keep the model's non-mu formulas (sigma/nu/tau) when refitting.

    gamlss(formula) alone refits a mean-only model, so a bootstrap/rolling/centile
    refit of a model with sigma_formula="~ x" silently loses the sigma model and
    returns fewer coefficients than the original fit.
    """

    formulas = call.get("parameter_formulas") or {}
    kwargs = {}
    for parameter, formula in dict(formulas).items():
        if parameter != "mu" and isinstance(formula, str) and formula.strip():
            kwargs[f"{parameter}_formula"] = formula
    return kwargs


def _coefficient_names(model: Any) -> tuple[str, ...]:
    """R names(unlist(coef(model))): "mu.(Intercept)", "mu.x", "sigma.x", ..."""

    names: list[str] = []
    for parameter in model.family.estimable_parameters:
        if parameter not in model.coefficients:
            continue
        info = model.terms.get(parameter) or {}
        labels = [str(label) for label in info.get("term_labels", [])]
        count = int(np.ravel(np.asarray(model.coefficients[parameter])).size)
        prefix = ["(Intercept)"] if info.get("intercept", True) else []
        candidate = prefix + labels
        if len(candidate) != count:
            candidate = [f"X{index + 1}" for index in range(count)]
        names.extend(f"{parameter}.{label}" for label in candidate)
    return tuple(names)


def fitQR(
    formula: str = None,
    data: Mapping[str, Any] = None,
    quantile: Any = (0.5,),
    **kwargs: Any,
) -> list:
    """R fitQR: quantile regression through SEP3 with three fixed parameters.

    Returns one fitted model per requested quantile, in the same order.
    """

    from . import gamlss

    if formula is None:
        raise ValueError("fitQR needs a formula")
    if data is None:
        raise ValueError("fitQR needs data")
    quantiles = np.atleast_1d(np.asarray(quantile, dtype=np.float64))
    if np.any(quantiles <= 0.0) or np.any(quantiles >= 1.0):
        raise ValueError("quantile should be between 0 and 1")
    models = []
    for value in quantiles:
        nu_start = float(np.sqrt((1.0 - value) / value))
        models.append(
            gamlss(
                formula,
                family="SEP3",
                data=dict(data),
                sigma_fix=True,
                sigma_start=1.0,
                nu_fix=True,
                nu_start=nu_start,
                tau_fix=True,
                tau_start=1.0,
                **kwargs,
            )
        )
    return models


def fitRolling(
    obj: Any = None,
    data: Mapping[str, Any] = None,
    window: int = 365,
    as_time: Any = None,
) -> pd.DataFrame:
    """R fitRolling: rolling refits with a one-step-ahead prediction each time.

    Returns a DataFrame indexed by the forecast time (as_time[window:]) whose
    columns are the predicted parameters plus GDincr.
    """

    from . import gamlss

    _require_model(obj, "fitRolling")
    values = _call_data(obj, data, "fitRolling")
    call = obj.call or {}
    formula = call.get("formula")
    family = obj.family
    method = call.get("method", "RS")
    if formula is None:
        raise ValueError("the model call does not record a formula")
    from .formula_parser import parse_formula

    response_name = parse_formula(formula).response
    n = len(np.asarray(values[response_name]))
    window = int(window)
    if window < 1 or window >= n:
        raise ValueError("window must be between 1 and n - 1")
    starts = {
        name: np.asarray(obj.fitted_values[name], dtype=np.float64)
        for name in family.estimable_parameters
        if name in obj.fitted_values
    }
    rows = []
    for start in range(0, n - window):
        stop = start + window
        window_data = {
            key: np.asarray(value)[start:stop] for key, value in values.items()
        }
        kwargs = {}
        for name, full in starts.items():
            kwargs[f"{name}_start"] = full[start:stop]
        model = gamlss(
            formula,
            family=family,
            data=window_data,
            method=method,
            **_parameter_formula_kwargs(call),
            **kwargs,
        )
        next_data = {
            key: np.asarray(value)[stop : stop + 1] for key, value in values.items()
        }
        predicted = predict_params(model, next_data)
        params = {name: float(np.ravel(predicted[name])[0]) for name in predicted}
        response = float(np.asarray(values[response_name])[stop])
        increment = float(np.ravel(family.g_dev_inc(y=response, **params))[0])
        params["GDincr"] = increment
        rows.append(params)
    # R labels the forecasts with the original row names; for a plain
    # data.frame those are 1-based positions, so default to window+1 .. n.
    index = (
        np.arange(window + 1, n + 1)
        if as_time is None
        else np.asarray(as_time)[window:]
    )
    frame = pd.DataFrame(rows, index=index)
    return frame


@dataclass(frozen=True)
class BootResult:
    boot: np.ndarray
    B: int
    trueB: int
    param: Mapping[str, np.ndarray]
    orig_coef: np.ndarray
    orig_param: Mapping[str, np.ndarray]
    orig_call: Any
    seed: int
    failed: int
    coef_names: tuple[str, ...] = ()


def _bootstrap(
    obj: Any,
    data: Any,
    B: int,
    newdata: Any,
    seed: int,
    bayesian: bool,
) -> BootResult:
    from . import gamlss

    _require_model(obj, "BayesianBoot" if bayesian else "NonParametricBoot")
    values = _call_data(obj, data, "BayesianBoot" if bayesian else "NonParametricBoot")
    target = values if newdata is None else dict(newdata)
    call = obj.call or {}
    formula = call.get("formula")
    family = obj.family
    method = call.get("method", "RS")
    if formula is None:
        raise ValueError("the model call does not record a formula")
    names = list(values)
    n = len(np.asarray(values[names[0]]))
    rng = np.random.default_rng(seed)
    original_param = predict_params(obj, target)
    original_coef = _flatten_coefficients(obj)

    coefficients = []
    samples = {name: [] for name in family.estimable_parameters}
    count = int(B)
    if count < 1:
        raise ValueError("B must be a positive integer")
    for _ in range(count):
        if bayesian:
            weights = rng.dirichlet(np.ones(n)) * n
            fit_data = values
            fit_kwargs = {"weights": weights}
        else:
            indices = rng.integers(0, n, size=n)
            fit_data = {
                name: np.asarray(value)[indices] for name, value in values.items()
            }
            fit_kwargs = {}
        try:
            model = gamlss(
                formula,
                family=family,
                data=fit_data,
                method=method,
                **_parameter_formula_kwargs(call),
                **fit_kwargs,
            )
            predicted = predict_params(model, target)
            coefficients.append(_flatten_coefficients(model))
            for name in samples:
                samples[name].append(np.asarray(predicted[name], dtype=np.float64))
        except Exception:  # noqa: BLE001 - R drops failed replicates the same way
            continue
    trueB = len(coefficients)
    matrix = (
        np.vstack(coefficients) if coefficients else np.zeros((0, original_coef.size))
    )
    params = {
        name: (
            np.vstack(values_)
            if values_
            else np.zeros((0, len(np.asarray(original_param[name]))))
        )
        for name, values_ in samples.items()
    }
    return BootResult(
        boot=matrix,
        B=count,
        trueB=trueB,
        param=params,
        orig_coef=original_coef,
        orig_param=original_param,
        orig_call=call,
        seed=int(seed),
        failed=count - trueB,
        coef_names=_coefficient_names(obj),
    )


def NonParametricBoot(
    obj: Any = None,
    data: Mapping[str, Any] = None,
    B: int = 100,
    newdata: Mapping[str, Any] = None,
    seed: int = 0,
) -> BootResult:
    """R NonParametricBoot: resample rows with replacement and refit."""

    return _bootstrap(obj, data, B, newdata, seed, bayesian=False)


def BayesianBoot(
    obj: Any = None,
    data: Mapping[str, Any] = None,
    B: int = 100,
    newdata: Mapping[str, Any] = None,
    seed: int = 0,
) -> BootResult:
    """R BayesianBoot: Rubin's Dirichlet(1) weights, refit each replicate."""

    return _bootstrap(obj, data, B, newdata, seed, bayesian=True)


@dataclass(frozen=True)
class CentilesBoot:
    boot0: np.ndarray
    boot: list
    B: int
    trueB: int
    xvalues: np.ndarray
    cent: np.ndarray
    orig_call: Any
    xname: str
    failed: int


def _centile_matrix(
    model: Any,
    xname: str,
    xvalues: np.ndarray,
    cent: np.ndarray,
) -> np.ndarray:
    covariates = [name for name in model.family.estimable_parameters]
    formula = (model.call or {}).get("formula", "")
    terms = [part.strip() for part in formula.split("~", 1)[-1].split("+")]
    linear = [term for term in terms if term not in ("1", "-1", "")]
    if linear not in ([xname], []):
        raise NotImplementedError(
            "centiles_boot supports a single-covariate model; the formula has "
            f"{linear}"
        )
    params = predict_params(model, {xname: np.asarray(xvalues, dtype=np.float64)})
    args = [
        np.asarray(params[name], dtype=np.float64) for name in model.family.parameters
    ]
    values = np.empty((xvalues.size, cent.size))
    for index, value in enumerate(cent / 100.0):
        values[:, index] = np.asarray(
            model.family.q(float(value), *args), dtype=np.float64
        )
    return values


def centiles_boot(
    obj: Any = None,
    data: Mapping[str, Any] = None,
    xname: str = None,
    xvalues: Any = None,
    power: Any = None,
    cent: Sequence[float] = (2.5, 50, 97.5),
    B: int = 100,
    calibration: bool = False,
    seed: int = 0,
) -> CentilesBoot:
    """R centiles.boot: bootstrap the centile curves of a single-covariate model."""

    from . import gamlss

    _require_model(obj, "centiles_boot")
    if xvalues is None:
        raise ValueError("The 'xvalues' argument is not specified")
    if xname is None:
        raise ValueError("The xname argument is not specified")
    if not isinstance(xname, str):
        raise ValueError("the xname argument is not a character")
    values = _call_data(obj, data, "centiles_boot")
    if xname not in values:
        raise ValueError(
            "the argument 'xname' is not included in the specified dataset."
        )
    call = obj.call or {}
    formula = call.get("formula")
    family = obj.family
    method = call.get("method", "RS")
    if formula is None:
        raise ValueError("the model call does not record a formula")
    xvalues = np.asarray(xvalues, dtype=np.float64)
    centiles = np.asarray(cent, dtype=np.float64)
    n = len(np.asarray(values[xname]))
    rng = np.random.default_rng(seed)

    original = _centile_matrix(obj, xname, xvalues, centiles)
    results = []
    for _ in range(int(B)):
        indices = rng.integers(0, n, size=n)
        resampled = {name: np.asarray(value)[indices] for name, value in values.items()}
        try:
            model = gamlss(
                formula,
                family=family,
                data=resampled,
                method=method,
                **_parameter_formula_kwargs(call),
            )
            effective = centiles
            if calibration:
                residuals = np.asarray(model.residuals, dtype=np.float64)
                z = np.quantile(residuals, centiles / 100.0)
                from scipy.stats import norm

                effective = 100.0 * norm.cdf(z)
            results.append(_centile_matrix(model, xname, xvalues, effective))
        except Exception:  # noqa: BLE001 - failed replicates are reported, not hidden
            continue
    return CentilesBoot(
        boot0=original,
        boot=results,
        B=int(B),
        trueB=len(results),
        xvalues=xvalues,
        cent=centiles,
        orig_call=call,
        xname=xname,
        failed=int(B) - len(results),
    )


__all__ = [
    "BayesianBoot",
    "BootResult",
    "CentilesBoot",
    "NonParametricBoot",
    "centiles_boot",
    "fitQR",
    "fitRolling",
]
