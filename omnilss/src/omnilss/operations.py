"""Migrated low-dependency utility operations from the R `gamlss` package.

R source references:
- file: `gamlss/R/extra.R`
- functions: `fitted.gamlss`, `coef.gamlss`, `coefAll`, `deviance.gamlss`,
  `lp`, `fv`, `is.gamlss`, `IC`, `GAIC.gamlss`, `GAIC_table`, `GAIC_scaled`,
  `.hat.WX`, `numeric.deriv`
- file: `gamlss/R/DevianceIncr.R`
- function: `devianceIncr`
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition
from .model import GAMLSSModel

PARAMETER_CHOICES = ("mu", "sigma", "nu", "tau")
FLOAT_DTYPE = jnp.float64


def _resolve_parameter(what: str | None = None, parameter: str | None = None) -> str:
    """Resolve the target parameter in the same spirit as `match.arg()`."""

    chosen = parameter if parameter is not None else what
    if chosen is None:
        return "mu"
    if chosen not in PARAMETER_CHOICES:
        raise ValueError(
            f"{chosen!r} is not a valid parameter; expected one of {PARAMETER_CHOICES}"
        )
    return chosen


def _require_gamlss(model: GAMLSSModel) -> None:
    if not is_gamlss(model):
        raise TypeError("this is not a gamlss object")


def _require_parameter(model: GAMLSSModel, what: str) -> None:
    if not model.has_parameter(what):
        raise ValueError(f"{what} is not a parameter in the object")


def _get_object_mapping_slot(
    model: GAMLSSModel,
    mapping_name: str,
    mapping: Mapping[str, Any],
    what: str,
    error_message: str = "no terms component",
) -> Any:
    _require_parameter(model, what)
    if what not in mapping:
        raise KeyError(error_message)
    return mapping[what]


def _gaic_single(model: GAMLSSModel, k: float = 2.0, c: bool = False) -> float:
    """Internal GAIC/AIC computation shared across wrappers."""

    _require_gamlss(model)
    value = model.g_dev + model.df_fit * k
    if k == 2 and c:
        value += (2 * model.df_fit * (model.df_fit + 1)) / (model.n - model.df_fit - 1)
    return float(value)


def _row_name(index: int, text_to_show: Sequence[str] | None) -> str:
    if text_to_show is None:
        return f"model_{index + 1}"
    return text_to_show[index]


def is_gamlss(x: Any) -> bool:
    """R reference: `gamlss/R/extra.R::is.gamlss`."""

    return isinstance(x, GAMLSSModel) and x.class_name == "gamlss"


def fitted(
    object: GAMLSSModel, what: str = "mu", parameter: str | None = None
) -> Any:
    """R reference: `gamlss/R/extra.R::fitted.gamlss`."""

    selected = _resolve_parameter(what, parameter)
    _require_parameter(object, selected)
    return object.fitted_values[selected]


def refit(object: GAMLSSModel, **kwargs: Any) -> Any:
    """R reference: `gamlss/R/extra.R::refit`.

    The Python port uses a callable stored in `object.call["callable"]`.
    """

    _require_gamlss(object)
    if object.call is None:
        raise ValueError("need an object with call component")
    if "callable" not in object.call:
        raise ValueError("call component must include a Python callable")

    new_cycle = 2 * int(object.control.get("n.cyc", 0))
    extras = {
        "start_from": object,
        "iter": object.iter,
        "n_cyc": new_cycle,
    }
    extras.update(kwargs)
    callable_fn = object.call["callable"]
    base_kwargs = dict(object.call.get("kwargs", {}))
    base_kwargs.update(extras)
    return callable_fn(**base_kwargs)


def coef(object: GAMLSSModel, what: str = "mu", parameter: str | None = None) -> Any:
    """R reference: `gamlss/R/extra.R::coef.gamlss`."""

    selected = _resolve_parameter(what, parameter)
    _require_parameter(object, selected)
    return object.coefficients[selected]


def coef_all(object: GAMLSSModel, deviance: bool = False) -> dict[str, Any]:
    """R reference: `gamlss/R/extra.R::coefAll`."""

    _require_gamlss(object)
    output: dict[str, Any] = {}
    for parameter in PARAMETER_CHOICES:
        if object.has_parameter(parameter):
            output[parameter] = coef(object, parameter)
    if deviance:
        output["deviance"] = object.g_dev
    return output


def residuals(
    object: GAMLSSModel,
    what: str = "z-scores",
    type: str = "simple",
    terms: Sequence[str] | None = None,
) -> Any:
    """R reference: `gamlss/R/extra.R::residuals.gamlss`."""

    _require_gamlss(object)
    if type not in ("simple", "weighted", "partial"):
        raise ValueError("invalid residual type")
    if what not in ("z-scores",) + PARAMETER_CHOICES:
        raise ValueError("invalid residual target")

    weights = object.weights
    if weights is None:
        weights = jnp.ones(len(jnp.asarray(object.y)))
    weights = jnp.asarray(weights, dtype=FLOAT_DTYPE)

    if what == "z-scores":
        if jnp.all(weights == 1):
            return object.residuals
        if jnp.all(jnp.floor(weights) == weights):
            if object.type == "Continuous":
                return jnp.repeat(jnp.asarray(object.residuals, dtype=FLOAT_DTYPE), weights.astype(int))
            if object.rqres is None:
                raise ValueError("discrete weighted residuals require an rqres callable")
            kwargs: dict[str, Any] = {"y": jnp.repeat(jnp.asarray(object.y), weights.astype(int))}
            active_parameters = object.parameters or object.par
            for parameter_name in active_parameters:
                kwargs[parameter_name] = jnp.repeat(
                    jnp.asarray(fitted(object, parameter_name), dtype=FLOAT_DTYPE),
                    weights.astype(int),
                )
            return object.rqres(**kwargs)
        return object.residuals

    _require_parameter(object, what)
    wv = _get_object_mapping_slot(object, "working_vectors", object.working_vectors, what)
    l_p = lp(object, what=what)
    wt = _get_object_mapping_slot(
        object, "iterative_weights", object.iterative_weights, what
    )
    offset = object.offsets.get(what, jnp.zeros_like(jnp.asarray(l_p, dtype=FLOAT_DTYPE)))

    wv_arr = jnp.asarray(wv, dtype=FLOAT_DTYPE)
    lp_arr = jnp.asarray(l_p, dtype=FLOAT_DTYPE)
    wt_arr = jnp.asarray(wt, dtype=FLOAT_DTYPE)
    offset_arr = jnp.asarray(offset, dtype=FLOAT_DTYPE)

    base = wv_arr - lp_arr + offset_arr
    if type == "simple":
        return base
    if type == "weighted":
        return jnp.sqrt(wt_arr) * base

    terms_pred = lpred(object, what=what, type="terms", terms=terms)
    if isinstance(terms_pred, dict):
        terms_matrix = jnp.asarray(terms_pred["fit"], dtype=FLOAT_DTYPE)
        return base + jnp.sum(terms_matrix, axis=1)
    return base + jnp.sum(jnp.asarray(terms_pred, dtype=FLOAT_DTYPE), axis=1)


def deviance(object: GAMLSSModel, what: str = "G") -> float:
    """R reference: `gamlss/R/extra.R::deviance.gamlss`."""

    _require_gamlss(object)
    if what == "G":
        return float(object.additional_slots.get("G.deviance", object.g_dev))
    if what == "P":
        return float(object.additional_slots["P.deviance"])
    raise ValueError("put G for Global or P for Penalized deviance")


def lp(obj: GAMLSSModel, what: str = "mu", parameter: str | None = None) -> Any:
    """R reference: `gamlss/R/extra.R::lp`."""

    selected = _resolve_parameter(what, parameter)
    _require_parameter(obj, selected)
    return obj.linear_predictors[selected]


def fv(obj: GAMLSSModel, what: str = "mu", parameter: str | None = None) -> Any:
    """R reference: `gamlss/R/extra.R::fv`."""

    return fitted(obj, what=what, parameter=parameter)


def model_frame(
    formula: GAMLSSModel, what: str = "mu", parameter: str | None = None
) -> dict[str, Any]:
    """R reference: `gamlss/R/extra.R::model.frame.gamlss`.

    The Python port returns a subset of stored data columns keyed by term names.
    """

    object = formula
    selected = _resolve_parameter(what, parameter)
    _require_parameter(object, selected)
    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("model frame requires call['data']")
    term_info = terms(object, what=selected)
    labels = term_info.get("term_labels", [])
    if term_info.get("intercept", False):
        labels = ["(Intercept)", *labels]
    frame: dict[str, Any] = {}
    for label in labels:
        if label == "(Intercept)":
            continue
        if label not in call_data:
            raise KeyError(f"term {label!r} not found in data")
        frame[label] = call_data[label]
    response_name = term_info.get("response")
    if response_name and response_name in call_data:
        frame[response_name] = call_data[response_name]
    return frame


def terms(x: GAMLSSModel, what: str = "mu", parameter: str | None = None) -> Any:
    """R reference: `gamlss/R/extra.R::terms.gamlss`."""

    selected = _resolve_parameter(what, parameter)
    return _get_object_mapping_slot(x, "terms", x.terms, selected)


def model_matrix(
    object: GAMLSSModel, what: str = "mu", parameter: str | None = None
) -> Any:
    """R reference: `gamlss/R/extra.R::model.matrix.gamlss`."""

    selected = _resolve_parameter(what, parameter)
    return _get_object_mapping_slot(object, "design_matrices", object.design_matrices, selected)


def formula(x: GAMLSSModel, what: str = "mu", parameter: str | None = None) -> Any:
    """R reference: `gamlss/R/extra.R::formula.gamlss`."""

    selected = _resolve_parameter(what, parameter)
    _require_parameter(x, selected)
    if selected not in x.formulas:
        raise KeyError("no formula component")
    selected_formula = x.formulas[selected]
    text = str(selected_formula)
    if "." in text:
        selected_terms = terms(x, what=selected)
        fallback = selected_terms.get("formula")
        if fallback is not None:
            return fallback
    return selected_formula


def ic(object: GAMLSSModel, k: float = 2.0) -> float:
    """R reference: `gamlss/R/extra.R::IC`."""

    return _gaic_single(object, k=k, c=False)


def lpred(
    obj: GAMLSSModel,
    what: str = "mu",
    parameter: str | None = None,
    type: str = "link",
    terms: Sequence[str] | None = None,
    se_fit: bool = False,
) -> Any:
    """R reference: `gamlss/R/lpred.R::lpred`.

    This staged port supports stored-term predictions and optional callback-based
    computation for `se.fit`.
    """

    _require_gamlss(obj)
    selected = _resolve_parameter(what, parameter)
    if type not in ("link", "response", "terms"):
        raise ValueError("invalid lpred type")
    if type == "link":
        fit_value = lp(obj, what=selected)
        if not se_fit:
            return fit_value
        callback = obj.additional_slots.get("lpred_se_callback")
        if callback is None:
            raise ValueError("se.fit for link predictions requires lpred_se_callback")
        return {"fit": fit_value, "se.fit": callback(obj, selected, type)}
    if type == "response":
        fit_value = fitted(obj, what=selected)
        if not se_fit:
            return fit_value
        callback = obj.additional_slots.get("lpred_se_callback")
        if callback is None:
            raise ValueError("se.fit for response predictions requires lpred_se_callback")
        return {"fit": fit_value, "se.fit": callback(obj, selected, type)}

    callback = obj.additional_slots.get("lpred_terms_callback")
    if callback is not None:
        return callback(obj, selected, se_fit=se_fit, terms=terms)

    terms_info = _get_object_mapping_slot(obj, "terms", obj.terms, selected)
    predictor = jnp.asarray(terms_info.get("predictor_matrix"), dtype=FLOAT_DTYPE)
    column_names = list(terms_info.get("column_names", []))
    if terms is not None:
        indices = [column_names.index(term_name) for term_name in terms]
        predictor = predictor[:, indices]
        column_names = [column_names[index] for index in indices]
    if se_fit:
        se_matrix = jnp.asarray(terms_info.get("se_matrix"), dtype=FLOAT_DTYPE)
        if terms is not None:
            se_matrix = se_matrix[:, indices]
        return {
            "fit": predictor,
            "se.fit": se_matrix,
            "constant": terms_info.get("constant", 0.0),
            "column_names": column_names,
        }
    return predictor


def gaic(
    object: GAMLSSModel, *others: GAMLSSModel, k: float = 2.0, c: bool = False
) -> float | list[dict[str, float | str]]:
    """R reference: `gamlss/R/extra.R::GAIC.gamlss`."""

    if not others:
        return _gaic_single(object, k=k, c=c)

    models = (object, *others)
    rows = [
        {
            "model": f"model_{index + 1}",
            "df": float(model.df_fit),
            "AIC": _gaic_single(model, k=k, c=c),
        }
        for index, model in enumerate(models)
    ]
    return sorted(rows, key=lambda item: float(item["AIC"]))


def gaic_table(
    object: GAMLSSModel,
    *others: GAMLSSModel,
    k: Sequence[float] | None = None,
    text_to_show: Sequence[str] | None = None,
) -> list[dict[str, float | str]]:
    """R reference: `gamlss/R/extra.R::GAIC_table`."""

    models = (object, *others)
    penalties = tuple(k or (2.0, 3.84, round(float(np.log(len(np.asarray(object.y)))), 2)))
    rows: list[dict[str, float | str]] = []

    for index, model in enumerate(models):
        row: dict[str, float | str] = {
            "model": _row_name(index, text_to_show),
            "df": float(model.df_fit),
        }
        for penalty in penalties:
            row[f"k={penalty}"] = _gaic_single(model, k=float(penalty), c=False)
        rows.append(row)
    return rows


def gaic_scaled(
    object: GAMLSSModel | Sequence[Sequence[float]],
    *others: GAMLSSModel,
    k: float = 2.0,
    c: bool = False,
    which: int = 1,
    diff_dev: float = 1000.0,
    text_to_show: Sequence[str] | None = None,
) -> float | list[dict[str, float | str | None]]:
    """R reference: `gamlss/R/extra.R::GAIC_scaled`.

    Plotting arguments from R are intentionally deferred until later stages.
    """

    if not others and not is_gamlss(object):
        matrix = np.asarray(object, dtype=np.float64)
        column_index = which - 1
        selected = matrix[:, column_index]
        winner = int(np.nanargmin(selected))
        delta = selected - selected[winner]
        delta = np.where(delta > diff_dev, np.nan, delta)
        scale_denom = np.nanmax(delta)
        scaled = np.where(np.isnan(delta), np.nan, 1 - delta / scale_denom)
        return [
            {"row": f"row_{idx + 1}", "delta": float(delta[idx]), "scaled": float(scaled[idx])}
            if not np.isnan(delta[idx])
            else {"row": f"row_{idx + 1}", "delta": None, "scaled": None}
            for idx in range(matrix.shape[0])
        ]

    if others:
        models = (object, *others)
        values = np.asarray(
            [_gaic_single(model, k=k, c=c) for model in models], dtype=np.float64
        )
        delta = values - values[np.argmin(values)]
        scaled = 1 - delta / np.max(delta)
        rows: list[dict[str, float | str | None]] = []
        for index, model in enumerate(models):
            rows.append(
                {
                    "model": _row_name(index, text_to_show),
                    "df": float(model.df_fit),
                    "AIC": float(values[index]),
                    "delta": float(np.round(delta[index], 5)),
                    "scaled": float(np.round(scaled[index], 4)),
                }
            )
        return rows

    return _gaic_single(object, k=k, c=c)


def hat_wx(w: Iterable[float], x: Any) -> jnp.ndarray:
    """R reference: `gamlss/R/extra.R::.hat.WX`."""

    w_array = jnp.asarray(list(w), dtype=FLOAT_DTYPE)
    x_array = jnp.asarray(x, dtype=FLOAT_DTYPE)
    if x_array.ndim == 1:
        x_array = jnp.column_stack((jnp.ones_like(x_array), x_array))
    if x_array.shape[0] != w_array.shape[0]:
        raise ValueError("`w' and 'x' are not having the same length")
    sqrt_w = jnp.sqrt(w_array)
    weighted_x = x_array * sqrt_w[:, None]
    q, _ = jnp.linalg.qr(weighted_x, mode="reduced")
    return jnp.sum(q * q, axis=1)


def numeric_deriv(
    func: Callable[..., Any],
    theta: Sequence[str],
    rho: dict[str, Any],
    delta: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """R reference: `gamlss/R/extra.R::numeric.deriv`.

    Returns a tuple `(value, gradient)` instead of an R object with attributes.
    """

    eps = float(np.sqrt(np.finfo(np.float64).eps))
    ans = jnp.atleast_1d(jnp.asarray(func(**rho), dtype=FLOAT_DTYPE))
    grad = np.empty((ans.shape[0], len(theta)), dtype=np.float64)

    for column, name in enumerate(theta):
        old_value = float(rho[name])
        step = eps * min(1.0, abs(old_value)) if delta is None else float(delta)
        # JAX uses float32 by default, so the original R-sized delta can underflow or
        # produce unstable forward differences. Use a representable symmetric step.
        step = max(step, 1e-4)
        upper_rho = dict(rho)
        lower_rho = dict(rho)
        upper_rho[name] = old_value + step
        lower_rho[name] = old_value - step
        ans_upper = jnp.atleast_1d(jnp.asarray(func(**upper_rho), dtype=FLOAT_DTYPE))
        ans_lower = jnp.atleast_1d(jnp.asarray(func(**lower_rho), dtype=FLOAT_DTYPE))
        grad[:, column] = np.asarray(
            (ans_upper - ans_lower) / (2.0 * step), dtype=np.float64
        )

    return ans, jnp.asarray(grad, dtype=FLOAT_DTYPE)


def deviance_increment(obj: GAMLSSModel, newdata: dict[str, Any] | None = None) -> Any:
    """R reference: `gamlss/R/DevianceIncr.R::devianceIncr`."""

    _require_gamlss(obj)
    family = obj.family
    if isinstance(family, dict):
        family = FamilyDefinition.from_mapping(family)
    if not isinstance(family, FamilyDefinition):
        raise TypeError("family must be a FamilyDefinition or compatible mapping")

    response = obj.y if newdata is None else newdata["y"]
    kwargs: dict[str, Any] = {}
    for parameter in family.parameters:
        kwargs[parameter] = fitted(obj, what=parameter)
    return family.g_dev_inc(response, **kwargs)
