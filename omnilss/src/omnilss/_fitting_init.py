"""Initialization helpers extracted from :mod:`omnilss.fitting`."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .families import FamilyDefinition


def _initial_parameter_value(
    family: FamilyDefinition,
    parameter: str,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
) -> float:
    """Construct staged starting values for non-mu parameters."""

    if parameter == "sigma":
        return _initial_sigma(family, y, mu, w)
    if parameter == "nu":
        if family.name == "PE":
            return 1.8
        if family.name == "TF":
            return 10.0
        if family.name == "JSU":
            return 0.0
        if family.name == "BCCG":
            return 1.0
        if family.name == "BCT":
            return 1.0
        if family.name == "BCPE":
            return 1.0
        if family.name == "GG":
            return 2.0
        if family.name == "GB2":
            return 1.0
        if family.name == "DEL":
            return 0.5
        if family.name == "NET":
            return 1.0
        if family.name == "GT":
            return 5.0
        if family.name == "SHASH":
            return 1.0
        if family.name == "SHASHo":
            return 0.0
        if family.name == "SN1":
            return 0.0
        if family.name == "SN2":
            return 1.0
        if family.name in ("ZAGA", "ZAIG", "ZAP", "ZINBI", "ZIP2", "BEZI", "BEOI"):
            return 0.2
        if family.name in ("BEINF", "BEINF0", "BEINF1"):
            return 0.1
        return 1.0
    if parameter == "tau":
        if family.name == "JSU":
            return 1.0
        if family.name == "BCT":
            return 10.0
        if family.name == "GB2":
            return 1.0
        if family.name == "BCPE":
            return 2.0
        if family.name == "NET":
            return 2.0
        if family.name == "GT":
            return 2.0
        if family.name == "SHASH":
            return 1.0
        if family.name == "SHASHo":
            return 1.0
        if family.name == "BEINF":
            return 0.1
        return 1.0
    return 1.0


def _initial_mu_beta(
    family: FamilyDefinition,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Construct a family-aware starting point for the mu predictor."""

    sqrt_w = np.sqrt(w)
    if family.name in ("PO", "GEOM", "ZIP", "YULE", "WARING", "ZAGA", "ZAIG", "ZAP", "ZINBI", "ZIP2", "NBI"):
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name == "BI":
        clipped = np.clip(y, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
    elif family.name == "BB":
        if fixed_parameter_values is None or "bd" not in fixed_parameter_values:
            raise ValueError("BB initialization requires fixed parameter 'bd'")
        bd = np.maximum(np.asarray(fixed_parameter_values["bd"], dtype=np.float64), 1.0)
        clipped = np.clip(y / bd, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
    elif family.name == "BE":
        clipped = np.clip(y, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
    elif family.name in ("WEI", "BCCG", "BCT", "BCPE", "GA", "GG", "GB2", "EXP", "LOGNO", "LNO", "IG", "IGAMMA"):
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    else:
        target = y
    wx = x * sqrt_w[:, None]
    wy = target * sqrt_w
    beta_mu, _, _, _ = np.linalg.lstsq(wx, wy, rcond=None)
    return beta_mu


def _initial_sigma(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> float:
    """Construct a family-aware starting point for the sigma parameter."""

    eps = np.finfo(np.float64).eps
    if family.name in ("LOGNO", "LNO"):
        log_y = np.log(np.maximum(y, eps))
        sigma = np.sqrt(np.sum(w * np.square(log_y - mu)) / np.sum(w))
    elif family.name == "GA":
        centered = (y - mu) / np.maximum(mu, eps)
        sigma = np.sqrt(np.sum(w * np.square(centered)) / np.sum(w))
    elif family.name == "NBI":
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        sigma = np.maximum((variance_component - mean_level) / np.maximum(mean_level**2, eps), eps)
    elif family.name == "IG":
        centered = (y - mu) / np.maximum(np.power(mu, 1.5), eps)
        sigma = np.sqrt(np.sum(w * np.square(centered)) / np.sum(w))
    elif family.name == "BE":
        clipped_y = np.clip(y, eps, 1.0 - eps)
        clipped_mu = np.clip(mu, eps, 1.0 - eps)
        variance_component = np.sum(w * np.square(clipped_y - clipped_mu)) / np.sum(w)
        mean_level = np.sum(w * clipped_mu) / np.sum(w)
        denom = np.maximum(mean_level * (1.0 - mean_level), eps)
        sigma = np.maximum(variance_component / denom, eps)
    elif family.name == "BB":
        sigma = 1.0
    elif family.name == "WEI":
        log_y = np.log(np.maximum(y, eps))
        log_mu = np.log(np.maximum(mu, eps))
        sigma = np.sqrt(np.sum(w * np.square(log_y - log_mu)) / np.sum(w))
    elif family.name in ("ZIP", "ZIP2", "ZINBI", "ZAP"):
        sigma = 0.1
    elif family.name in ("ZAGA", "ZAIG"):
        y_nonzero = y[y > 0]
        mu_nonzero = mu[y > 0]
        w_nonzero = w[y > 0]
        if len(y_nonzero) > 0:
            if family.name == "ZAGA":
                centered = (y_nonzero - mu_nonzero) / np.maximum(mu_nonzero, eps)
                sigma = np.sqrt(np.sum(w_nonzero * np.square(centered)) / np.sum(w_nonzero))
            else:
                centered = (y_nonzero - mu_nonzero) / np.maximum(np.power(mu_nonzero, 1.5), eps)
                sigma = np.sqrt(np.sum(w_nonzero * np.square(centered)) / np.sum(w_nonzero))
        else:
            sigma = 0.5
    elif family.name == "NBII":
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        sigma = np.maximum((variance_component - mean_level) / np.maximum(mean_level, eps), eps)
    elif family.name == "PARETO2":
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        cv = np.sqrt(variance_component) / np.maximum(mean_level, eps)
        sigma = np.clip(cv**2 / (1.0 + cv**2), eps, 0.9)
    elif family.name in ("IGAMMA", "GG"):
        sigma = 0.5
    elif family.name == "NET":
        sigma = 1.0
    elif family.name == "PE":
        mean_y = np.sum(w * y) / np.sum(w)
        sigma = (np.sum(w * np.abs(y - mean_y)) / np.sum(w) + np.sqrt(np.sum(w * np.square(y - mean_y)) / np.sum(w))) / 2.0
    elif family.name in ("YULE", "WARING"):
        sigma = 1.0 if family.name == "WARING" else 2.0
    else:
        sigma = np.sqrt(np.sum(w * np.square(y - mu)) / np.sum(w))
    return max(float(sigma), eps)


__all__ = ["_initial_mu_beta", "_initial_parameter_value", "_initial_sigma"]
