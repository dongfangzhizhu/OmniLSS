"""Deep GAMLSS module."""

from __future__ import annotations

try:
    from .deep_gamlss import (
        ParameterNetwork,
        DeepGAMLSS,
        fit_deep_gamlss,
        predict_deep_gamlss,
    )
except ImportError as exc:  # optional dependency gate
    _IMPORT_ERROR = exc

    def _missing(*args, **kwargs):
        raise ImportError(
            "Deep GAMLSS requires optional dependencies. "
            "Install with: pip install omnilss[deep]"
        ) from _IMPORT_ERROR

    ParameterNetwork = _missing  # type: ignore
    DeepGAMLSS = _missing  # type: ignore
    fit_deep_gamlss = _missing  # type: ignore
    predict_deep_gamlss = _missing  # type: ignore

__all__ = [
    "ParameterNetwork",
    "DeepGAMLSS",
    "fit_deep_gamlss",
    "predict_deep_gamlss",
]
