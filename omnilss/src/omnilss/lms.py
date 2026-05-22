"""R-aligned LMS/z-score utilities.

R reference: `gamlss/R/lms.R::{lms, z.scores}`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LMSResult:
    z: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    nu: np.ndarray


def z_scores(y, mu, sigma, nu) -> np.ndarray:
    """Compute LMS z-scores using Box-Cox transform with log-limit at nu≈0."""
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), np.finfo(np.float64).eps)
    nu_arr = np.asarray(nu, dtype=np.float64)

    ratio = np.maximum(y_arr / np.maximum(mu_arr, np.finfo(np.float64).eps), np.finfo(np.float64).eps)
    use_log = np.abs(nu_arr) < 1e-8
    nu_safe = np.where(use_log, 1.0, nu_arr)
    z_bc = (np.power(ratio, nu_safe) - 1.0) / (nu_safe * sigma_arr)
    z_log = np.log(ratio) / sigma_arr
    return np.where(use_log, z_log, z_bc)


def lms(y, mu, sigma, nu) -> LMSResult:
    """Build LMS payload with computed z-scores."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    z = z_scores(y=y, mu=mu_arr, sigma=sigma_arr, nu=nu_arr)
    return LMSResult(z=z, mu=mu_arr, sigma=sigma_arr, nu=nu_arr)


__all__ = ["lms", "z_scores", "LMSResult"]
