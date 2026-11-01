# SPDX-License-Identifier: GPL-3.0-or-later
"""Categorical predictors and R's `contrasts` argument (D2.134).

Golden reference: `omnilss/tests/fixtures/r_golden/contrasts_2026-11-01.json`,
written by `tmp/d2134_parity.R` against live R 4.6.0 / gamlss 5.5.0.

Before this the RS path could not fit a categorical predictor at all: a
`factor(g)` term raised "references variable 'factor'", and a bare character
column raised "could not convert string to float".  R instead expands the
column into k-1 coded columns, and the `contrasts` argument selects the coding
while leaving the deviance untouched (same column space).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import NO, gamlss

GOLDEN = (
    Path(__file__).parent / "fixtures" / "r_golden" / "contrasts_2026-11-01.json"
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(golden) -> dict:
    return {
        "y": np.asarray(golden["data"]["y"], dtype=np.float64),
        "x": np.asarray(golden["data"]["x"], dtype=np.float64),
        "g": np.asarray(golden["data"]["g"]),
    }


@pytest.mark.parametrize(
    ("scheme", "key"),
    [
        ("contr.treatment", "treatment"),
        ("contr.sum", "sum"),
        ("contr.helmert", "helmert"),
    ],
)
def test_coding_matches_live_r(golden, data, scheme, key):
    """Every supported coding must reproduce R's coefficient vector exactly."""
    model = gamlss(
        "y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
        contrasts=scheme,
    )
    np.testing.assert_allclose(
        np.ravel(model.coefficients["mu"]),
        np.asarray(golden[key]["mu"], dtype=np.float64),
        rtol=1e-8,
        atol=1e-10,
    )


def test_deviance_is_coding_invariant(golden, data):
    """Different codings span the same column space: same fit, different labels."""
    fits = [
        gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
               contrasts=scheme)
        for scheme in ("contr.treatment", "contr.sum", "contr.helmert")
    ]
    for model in fits[1:]:
        assert float(model.g_dev) == pytest.approx(float(fits[0].g_dev), rel=1e-10)
    assert float(fits[0].g_dev) == pytest.approx(
        float(golden["treatment"]["deviance"]), rel=1e-6
    )


def test_default_coding_is_treatment(golden, data):
    """R's default is contr.treatment; omitting contrasts must not change that."""
    explicit = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
                      contrasts="contr.treatment")
    default = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30)
    np.testing.assert_array_equal(
        np.ravel(default.coefficients["mu"]), np.ravel(explicit.coefficients["mu"])
    )


def test_per_variable_mapping(golden, data):
    """contrasts accepts a mapping so one factor can differ from the default."""
    mapping = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
                     contrasts={"g": "contr.sum"})
    plain = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
                   contrasts="contr.sum")
    np.testing.assert_array_equal(
        np.ravel(mapping.coefficients["mu"]), np.ravel(plain.coefficients["mu"])
    )


def test_factor_wrapper_is_accepted(golden, data):
    """factor(g) is the explicit spelling R users write for numeric codes."""
    wrapped = gamlss("y ~ factor(g) + x", family=NO(), data=data, method="RS",
                     max_iter=30)
    bare = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30)
    np.testing.assert_array_equal(
        np.ravel(wrapped.coefficients["mu"]), np.ravel(bare.coefficients["mu"])
    )


@pytest.mark.parametrize("bad", ["contr.poly", {"g": "contr.Fancy"}])
def test_unsupported_coding_is_rejected(data, bad):
    with pytest.raises(ValueError, match="unsupported contrast"):
        gamlss("y ~ g + x", family=NO(), data=data, method="RS", contrasts=bad)


def test_single_level_factor_is_rejected(data):
    """R: contrasts need at least two levels; silently dropping the term is worse."""
    flat = {**data, "g": np.array(["a"] * len(data["y"]))}
    with pytest.raises(ValueError, match="at least two are required"):
        gamlss("y ~ g + x", family=NO(), data=flat, method="RS")


@pytest.mark.parametrize("method", ["CG", "MIXED", "RS_JAX"])
def test_contrasts_is_rs_only_for_now(data, method):
    with pytest.raises(ValueError, match="implemented for method='RS' only"):
        gamlss("y ~ g + x", family=NO(), data=data, method=method,
               contrasts="contr.sum")
