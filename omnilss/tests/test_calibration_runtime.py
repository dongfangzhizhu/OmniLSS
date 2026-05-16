import numpy as np

from omnilss.calibration_runtime import calibration_curve, crps_gaussian, interval_coverage, pit_values


def test_pit_values_clipped_open_interval():
    x = np.array([0.0, 0.5, 1.0])
    p = pit_values(x)
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)


def test_interval_coverage_basic():
    y = np.array([0.0, 1.0, 2.0])
    lo = np.array([-1.0, 0.5, 1.5])
    hi = np.array([0.5, 1.5, 2.5])
    cov = interval_coverage(y, lo, hi)
    assert np.isclose(cov, 1.0)


def test_calibration_curve_shapes():
    pit = np.linspace(0.01, 0.99, 100)
    x, f = calibration_curve(pit, bins=10)
    assert x.shape == (10,)
    assert f.shape == (10,)
    assert np.isclose(np.sum(f), 1.0)


def test_crps_gaussian_non_negative():
    y = np.array([0.0, 1.0, -1.0])
    mu = np.array([0.1, 0.9, -0.8])
    sigma = np.array([1.0, 1.2, 0.7])
    score = crps_gaussian(y, mu, sigma)
    assert score >= 0.0
