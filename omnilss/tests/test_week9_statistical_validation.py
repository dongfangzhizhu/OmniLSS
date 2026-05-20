from __future__ import annotations

import numpy as np

from omnilss.validation.statistical import (
    asymptotic_consistency,
    calibration_coverage_normal,
    parameter_recovery_normal,
    synthetic_normal_data,
)


def test_parameter_recovery_on_synthetic_normal_data():
    y = synthetic_normal_data(n=2000, mu=2.0, sigma=1.5, seed=42)
    result = parameter_recovery_normal(y, true_mu=2.0)
    assert result.absolute_error < 0.12


def test_calibration_coverage_near_nominal_for_normal_data():
    y = synthetic_normal_data(n=5000, mu=0.0, sigma=1.0, seed=7)
    mu_hat = float(np.mean(y))
    sigma_hat = float(np.std(y))
    calib = calibration_coverage_normal(y, mu_hat=mu_hat, sigma_hat=sigma_hat)
    assert calib.absolute_gap < 0.03


def test_asymptotic_consistency_detects_nonincreasing_error_profile():
    profile = {100: 0.25, 500: 0.10, 2000: 0.04}
    consistency = asymptotic_consistency(profile)
    assert consistency.monotone_nonincreasing is True
