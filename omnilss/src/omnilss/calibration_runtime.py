"""Phase 1 calibration runtime metrics."""

from __future__ import annotations

import numpy as np


def pit_values(cdf_values: np.ndarray) -> np.ndarray:
    p = np.asarray(cdf_values, dtype=np.float64)
    return np.clip(p, 1e-12, 1.0 - 1e-12)


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    yv = np.asarray(y, dtype=np.float64)
    l = np.asarray(lo, dtype=np.float64)
    h = np.asarray(hi, dtype=np.float64)
    inside = (yv >= l) & (yv <= h)
    return float(np.mean(inside))


def calibration_curve(pit: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    p = pit_values(pit)
    edges = np.linspace(0.0, 1.0, bins + 1)
    hist, _ = np.histogram(p, bins=edges)
    freq = hist / np.sum(hist)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, freq


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive distributions."""
    from math import erf, sqrt, pi

    yv = np.asarray(y, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    s = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)

    z = (yv - m) / s
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    crps = s * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))
