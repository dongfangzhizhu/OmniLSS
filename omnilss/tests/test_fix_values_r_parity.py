# SPDX-License-Identifier: GPL-3.0-or-later
"""R's `mu.fix / sigma.fix / nu.fix / tau.fix` (D2.131b).

R half of the R reference lives in
`omnilss/tests/fixtures/r_golden/family_fix_values_2026-11-01.json`, written by
`tmp/d2131b_parity.R` against live R 4.6.0 / gamlss 5.5.0.

R semantics (`gamlss-5.R:310`): a fixed parameter is simply not updated —
`if (family$parameter$mu == TRUE & mu.fix == FALSE)` guards the submodel — so it
holds its starting value forever and `mu.coefficients` comes back `NULL`.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import NO, gamlss

GOLDEN = (
    Path(__file__).parent / "fixtures" / "r_golden" / "family_fix_values_2026-11-01.json"
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(golden) -> dict:
    return {
        "y": np.asarray(golden["data"]["y"], dtype=np.float64),
        "x": np.asarray(golden["data"]["x"], dtype=np.float64),
    }


def _values(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def test_mu_fix_matches_live_r(golden, data):
    """mu held at mu.start: deviance, sigma estimate and fitted mu must be R's."""
    model = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=20,
        mu_fix=True, mu_start=2.0,
    )
    reference = golden["mu_fix"]

    assert float(model.g_dev) == pytest.approx(float(reference["deviance"]), rel=1e-6)
    assert "mu" not in model.coefficients, "R leaves mu.coefficients NULL"
    np.testing.assert_allclose(
        _values(model.coefficients["sigma"]),
        _values(reference["sigma_coef"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _values(model.fitted_values["mu"])[:3],
        _values(reference["mu_fv_head"]),
        rtol=1e-12,
    )


def test_sigma_fix_matches_live_r(golden, data):
    """sigma held at sigma.start, mu still estimated exactly as R does."""
    model = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=20,
        sigma_fix=True, sigma_start=0.5,
    )
    reference = golden["sigma_fix"]

    assert float(model.g_dev) == pytest.approx(float(reference["deviance"]), rel=1e-6)
    assert "sigma" not in model.coefficients
    np.testing.assert_allclose(
        _values(model.coefficients["mu"]),
        _values(reference["mu_coef"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _values(model.fitted_values["sigma"])[:3],
        _values(reference["sigma_fv_head"]),
        rtol=1e-12,
    )


def test_mu_fix_is_not_the_same_as_the_free_fit(golden, data):
    """Guard against a silent no-op: fixing mu must change the deviance."""
    model = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=20,
        mu_fix=True, mu_start=2.0,
    )
    assert float(model.g_dev) > float(golden["reference"]["deviance"]) * 1.5


def test_mu_start_sets_the_fixed_value(golden, data):
    """Without mu.start R falls back to family$mu.initial; with it, it uses it."""
    a = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=20,
        mu_fix=True, mu_start=2.0,
    )
    b = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=20,
        mu_fix=True, mu_start=3.5,
    )
    np.testing.assert_allclose(_values(a.fitted_values["mu"]), 2.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_values(b.fitted_values["mu"]), 3.5, rtol=0, atol=1e-12)
    assert float(a.g_dev) != pytest.approx(float(b.g_dev), rel=1e-3)


@pytest.mark.parametrize("flag", [1, "yes", None])
def test_non_logical_fix_flag_is_rejected(data, flag):
    """R: if (!is.logical(mu.fix)) stop("mu.fix should be logical TRUE or FALSE")."""
    with pytest.raises(ValueError, match="should be logical"):
        gamlss("y ~ x", family=NO(), data=data, method="RS", mu_fix=flag)


@pytest.mark.parametrize("method", ["CG", "MIXED", "RS_JAX"])
def test_fix_flags_are_rs_only_for_now(data, method):
    """Failing closed beats silently estimating a parameter the caller fixed."""
    with pytest.raises(ValueError, match="implemented for method='RS' only"):
        gamlss(
            "y ~ x", family=NO(), data=data, method=method,
            mu_fix=True, mu_start=2.0,
        )


def test_fix_conflicts_with_a_data_column(data):
    """Two channels for the same frozen value is ambiguous, not a preference."""
    with pytest.raises(ValueError, match="already carries a column named 'mu'"):
        gamlss(
            "y ~ x", family=NO(), data={**data, "mu": np.full(len(data["y"]), 2.0)},
            method="RS", mu_fix=True, mu_start=2.0,
        )


def test_fixed_parameter_keeps_its_value_across_iterations(data):
    """The frozen value must not drift: one cycle and twenty cycles agree."""
    short = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=1,
        mu_fix=True, mu_start=2.0,
    )
    long = gamlss(
        "y ~ x", family=NO(), data=data, method="RS", max_iter=20,
        mu_fix=True, mu_start=2.0,
    )
    np.testing.assert_allclose(_values(short.fitted_values["mu"]), 2.0, atol=1e-12)
    np.testing.assert_allclose(_values(long.fitted_values["mu"]), 2.0, atol=1e-12)
