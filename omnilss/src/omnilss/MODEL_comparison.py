"""R-aligned model comparison summary interfaces.

R source reference:
- files:
  - `gamlss/R/stepGAIC-03-10-13..R`
  - `gamlss/R/DropAddStepGAIC-Parallel.R`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import GAMLSSModel
from .operations import gaic, is_gamlss
from .LR_test_12_06_2013 import _chi2_sf


@dataclass(frozen=True)
class ModelComparisonRow:
    """One row in a staged model-comparison table."""

    index: int
    family: str
    df_fit: float
    deviance: float
    aic: float
    sbc: float
    delta_df: float | None
    delta_deviance: float | None
    lr_p_value: float | None


@dataclass(frozen=True)
class ModelComparisonResult:
    """Structured comparison table for multiple staged models."""

    rows: tuple[ModelComparisonRow, ...]


@dataclass(frozen=True)
class GAICWeightRow:
    """One row in a staged GAIC weight table."""

    index: int
    family: str
    criterion: float
    delta: float
    weight: float


@dataclass(frozen=True)
class GAICWeightsResult:
    """Structured GAIC/AIC weight summary for staged model selection."""

    rows: tuple[GAICWeightRow, ...]


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def compare_models(*models: GAMLSSModel) -> ModelComparisonResult:
    """Staged model-comparison table inspired by `anova`/stepwise summaries.

    Current behavior:
    - Reports deviance, df, AIC, SBC for each model.
    - Adds adjacent-model LR deltas when degrees of freedom increase.
    """

    if not models:
        raise ValueError("at least one model is required")
    for model in models:
        _require_gamlss(model)

    rows: list[ModelComparisonRow] = []
    previous: GAMLSSModel | None = None
    for index, model in enumerate(models, start=1):
        family_name = getattr(model.family, "name", str(model.family))
        df_fit = float(model.df_fit)
        dev = float(model.g_dev)
        aic = float(model.additional_slots.get("aic", gaic(model, k=2.0)))
        sbc = float(model.additional_slots.get("sbc", gaic(model, k=float(np.log(max(model.n, 1))))))

        delta_df: float | None = None
        delta_dev: float | None = None
        lr_p: float | None = None
        if previous is not None:
            delta_df = df_fit - float(previous.df_fit)
            delta_dev = float(previous.g_dev) - dev
            if delta_df > 0 and delta_dev >= 0:
                lr_p = _chi2_sf(delta_dev, delta_df)

        rows.append(
            ModelComparisonRow(
                index=index,
                family=family_name,
                df_fit=df_fit,
                deviance=dev,
                aic=aic,
                sbc=sbc,
                delta_df=delta_df,
                delta_deviance=delta_dev,
                lr_p_value=lr_p,
            )
        )
        previous = model

    return ModelComparisonResult(rows=tuple(rows))


def gaic_weights(
    *models: GAMLSSModel,
    k: float = 2.0,
) -> GAICWeightsResult:
    """Staged GAIC weight summary inspired by `GAIC.weights` patterns."""

    if not models:
        raise ValueError("at least one model is required")
    for model in models:
        _require_gamlss(model)

    criteria = np.array([float(gaic(model, k=k)) for model in models], dtype=np.float64)
    min_value = float(np.min(criteria))
    delta = criteria - min_value
    raw = np.exp(-0.5 * delta)
    denom = float(np.sum(raw))
    weights = raw / denom if denom > 0.0 else np.full_like(raw, np.nan)

    rows: list[GAICWeightRow] = []
    for index, (model, criterion, d_value, weight) in enumerate(zip(models, criteria, delta, weights, strict=False), start=1):
        family_name = getattr(model.family, "name", str(model.family))
        rows.append(
            GAICWeightRow(
                index=index,
                family=family_name,
                criterion=float(criterion),
                delta=float(d_value),
                weight=float(weight),
            )
        )
    return GAICWeightsResult(rows=tuple(rows))


__all__ = [
    "GAICWeightRow",
    "GAICWeightsResult",
    "ModelComparisonResult",
    "ModelComparisonRow",
    "compare_models",
    "gaic_weights",
]
