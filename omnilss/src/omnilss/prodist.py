"""R-aligned predicted distribution parameter interfaces.

R source reference:
- file: `gamlss/R/prodist.R`
- functions: `prodist.gamlss`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import GAMLSSModel
from .predictAll_22_08_22 import PredictAllResult, predict_all


@dataclass(frozen=True)
class ProDistResult:
    """Structured staged `prodist.gamlss` payload."""

    family: str
    parameters: dict[str, np.ndarray | dict[str, np.ndarray]]
    output: str
    type: str


def prodist_data(
    object: GAMLSSModel,
    newdata: dict[str, object] | None = None,
    type: str = "response",
    se_fit: bool = False,
    use_weights: bool = False,
) -> ProDistResult:
    """R reference: `gamlss/R/prodist.R::prodist.gamlss`.

    Staged behavior:
    - Returns predicted distribution parameters as a structured object.
    - Reuses `predict_all()` and drops any observed-response column.
    """

    predicted = predict_all(
        object,
        newdata=newdata,
        type=type,
        se_fit=se_fit,
        output="list",
        use_weights=use_weights,
    )
    values = predicted.values if isinstance(predicted, PredictAllResult) else dict(predicted)
    parameters = {key: value for key, value in values.items() if key != "y"}
    normalized: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
    for key, value in parameters.items():
        if isinstance(value, dict):
            entry: dict[str, np.ndarray] = {}
            for subkey, subvalue in value.items():
                entry[str(subkey)] = np.asarray(subvalue, dtype=np.float64)
            normalized[str(key)] = entry
        else:
            normalized[str(key)] = np.asarray(value, dtype=np.float64)
    return ProDistResult(
        family=str(getattr(object.family, "name", object.family)),
        parameters=normalized,
        output="list",
        type=str(type).strip().lower(),
    )

prodist = prodist_data

__all__ = [
    "ProDistResult",
    "prodist",
    "prodist_data",
]
