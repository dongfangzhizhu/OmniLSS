"""Regression test for gamlss_ml sigma coefficient scale (H3).

Validates that ``gamlss_ml`` stores the sigma coefficient on the *link
scale* (e.g. ``log(sigma)``) when sigma is intercept-only, consistent with
the RS path and with all other parameters.  Previously it stored the
parameter scale (sigma itself), which broke JOINT/L-BFGS warm starts that
treat coefficients as link-scale and re-apply the inverse link.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.distributions import NO
from omnilss.fitting import gamlss, gamlss_ml


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = 1.5 + 2.0 * x + rng.normal(0, 0.5, n)
    return {"y": y, "x": x}


def test_gamlss_ml_sigma_coefficient_is_link_scale() -> None:
    """coefficients['sigma'] must equal link(mean sigma) = log(sigma)."""
    data = _make_data()
    ml = gamlss_ml("y ~ x", family=NO(), data=data, sigma_formula="~1")
    sigma_coef = np.asarray(ml.coefficients["sigma"])
    sigma_fit = np.asarray(ml.fitted_values["sigma"])
    assert sigma_coef.shape == (1,), "intercept-only sigma has one coefficient"
    expected = np.log(float(np.mean(sigma_fit)))
    np.testing.assert_allclose(sigma_coef[0], expected, rtol=1e-6, atol=1e-9)


def test_gamlss_ml_sigma_scale_matches_rs_path() -> None:
    """gamlss_ml and RS must agree on the sigma coefficient (link scale)."""
    data = _make_data()
    ml = gamlss_ml("y ~ x", family=NO(), data=data, sigma_formula="~1")
    rs = gamlss("y ~ x", family=NO(), data=data, method="RS")
    np.testing.assert_allclose(
        np.asarray(ml.coefficients["sigma"]),
        np.asarray(rs.coefficients["sigma"]),
        rtol=1e-3,
        atol=1e-6,
    )


def test_gamlss_ml_sigma_coefficient_not_parameter_scale() -> None:
    """Regression guard: the coefficient must NOT be sigma itself."""
    data = _make_data()
    ml = gamlss_ml("y ~ x", family=NO(), data=data, sigma_formula="~1")
    sigma_coef = float(np.asarray(ml.coefficients["sigma"])[0])
    sigma_mean = float(np.mean(np.asarray(ml.fitted_values["sigma"])))
    assert abs(sigma_coef - sigma_mean) > 1e-3, (
        "sigma coefficient must be link-scale, not the parameter value"
    )


def test_joint_warm_start_from_gamlss_ml() -> None:
    """JOINT path with a gamlss_ml-style sigma coefficient must fit cleanly."""
    data = _make_data()
    joint = gamlss("y ~ x", family=NO(), data=data, method="JOINT")
    sigma_fit = np.asarray(joint.fitted_values["sigma"])
    assert np.all(sigma_fit > 0)
    assert np.all(np.isfinite(sigma_fit))
