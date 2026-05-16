"""Phase 0 stability regression tests on extreme-value inputs."""

import numpy as np

from omnilss.fitting import gamlss


def _assert_model_finite(model):
    assert np.isfinite(float(model.g_dev))
    for param in model.family.parameters:
        if param in model.fitted_values:
            vals = np.asarray(model.fitted_values[param])
            assert np.isfinite(vals).all()


def test_no_family_stability_large_scale_and_outliers():
    rng = np.random.default_rng(123)
    n = 256
    x = rng.normal(size=n)
    y = 1e6 * (0.2 + 0.8 * x) + rng.normal(scale=5e5, size=n)
    y[:5] = np.array([1e9, -1e9, 5e8, -5e8, 2e9], dtype=np.float64)

    model = gamlss("y ~ x", family="NO", data={"x": x, "y": y}, method="RS")
    _assert_model_finite(model)


def test_ga_family_stability_small_positive_and_high_dynamic_range():
    rng = np.random.default_rng(456)
    n = 256
    x = rng.normal(size=n)
    base = np.exp(2.0 + 0.6 * x)
    y = base * rng.lognormal(mean=0.0, sigma=2.0, size=n)
    y[:5] = np.array([1e-12, 1e-10, 1e-8, 1e6, 1e8], dtype=np.float64)
    y = np.clip(y, 1e-12, np.inf)

    model = gamlss("y ~ x", family="GA", data={"x": x, "y": y}, method="RS")
    _assert_model_finite(model)
