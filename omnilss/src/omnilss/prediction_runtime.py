"""Schema-safe prediction runtime wrappers.

These helpers are intentionally thin: they delegate design-matrix rebuilding to
``omnilss.prediction`` so legacy serving/runtime entry points cannot bypass the
saved design schema or structured prediction errors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .family_capabilities import require_family_capability
from .prediction import PredictionSchemaError, predict_params


def _family_name(model: Any) -> str:
    family = getattr(model, "family", None)
    return str(getattr(family, "name", family))


def _require_prediction_route(model: Any) -> None:
    """Require that the model family advertises prediction support.

    The public runtime wrappers allow experimental families by default to remain
    compatible with the development API, but unsupported/unknown families still
    fail before any numerical prediction is attempted.
    """

    require_family_capability(
        _family_name(model),
        "prediction",
        allow_experimental=True,
    )


def predict_mean(model: Any, newdata: dict[str, Any]) -> np.ndarray:
    """Return the predicted ``mu`` parameter using the saved design schema."""

    _require_prediction_route(model)
    params = predict_params(model, newdata, which=["mu"])
    if "mu" not in params:
        raise PredictionSchemaError(
            "Model did not produce a mu prediction",
            parameter="mu",
            reason="missing mu prediction",
            code="missing_mu_prediction",
        )
    return np.asarray(params["mu"], dtype=np.float64)


def predict_quantile(model: Any, newdata: dict[str, Any], q: float = 0.5) -> np.ndarray:
    """Return a conditional quantile using all predicted distribution parameters."""

    if not 0.0 < float(q) < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q!r}")
    _require_prediction_route(model)
    family = getattr(model, "family", None)
    quantile_fn = getattr(family, "q", None)
    if quantile_fn is None:
        raise PredictionSchemaError(
            f"Family {_family_name(model)!r} does not expose a quantile function",
            reason="missing quantile function",
            code="missing_quantile_function",
        )
    params = predict_params(model, newdata)
    return np.asarray(quantile_fn(float(q), **params), dtype=np.float64)


def predict_interval(
    model: Any, newdata: dict[str, Any], alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Return a central ``1 - alpha`` prediction interval."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    lo = predict_quantile(model, newdata, float(alpha) / 2.0)
    hi = predict_quantile(model, newdata, 1.0 - float(alpha) / 2.0)
    return lo, hi


def predict_distribution(model: Any, newdata: dict[str, Any]) -> dict[str, Any]:
    """Return family metadata and all predicted distribution parameters."""

    _require_prediction_route(model)
    return {
        "family": _family_name(model),
        "params": predict_params(model, newdata),
    }
