"""Statistical validation suite primitives for reproducible credibility checks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RecoveryResult:
    true_value: float
    estimated_value: float
    absolute_error: float


@dataclass(frozen=True)
class CalibrationResult:
    expected_coverage: float
    observed_coverage: float
    absolute_gap: float


@dataclass(frozen=True)
class ConsistencyResult:
    sample_sizes: tuple[int, ...]
    errors: tuple[float, ...]
    monotone_nonincreasing: bool


def synthetic_normal_data(n: int, mu: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


def parameter_recovery_normal(y: np.ndarray, true_mu: float) -> RecoveryResult:
    est_mu = float(np.mean(np.asarray(y, dtype=np.float64)))
    abs_err = abs(est_mu - true_mu)
    return RecoveryResult(true_value=true_mu, estimated_value=est_mu, absolute_error=abs_err)


def calibration_coverage_normal(
    y: np.ndarray,
    mu_hat: float,
    sigma_hat: float,
    z: float = 1.96,
) -> CalibrationResult:
    y = np.asarray(y, dtype=np.float64)
    lower = mu_hat - z * sigma_hat
    upper = mu_hat + z * sigma_hat
    observed = float(np.mean((y >= lower) & (y <= upper)))
    expected = 0.95
    return CalibrationResult(expected_coverage=expected, observed_coverage=observed, absolute_gap=abs(observed - expected))


def asymptotic_consistency(errors_by_n: dict[int, float]) -> ConsistencyResult:
    pairs = sorted((int(k), float(v)) for k, v in errors_by_n.items())
    sizes = tuple(k for k, _ in pairs)
    errs = tuple(v for _, v in pairs)
    monotone = all(errs[i + 1] <= errs[i] + 1e-12 for i in range(len(errs) - 1))
    return ConsistencyResult(sample_sizes=sizes, errors=errs, monotone_nonincreasing=monotone)
