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
from omnilss.prediction import predict_params

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
def test_other_routes_accept_contrasts(golden, data, method):
    """D2.136b: CG, MIXED and the traced RS backend build the same coded matrix."""
    model = gamlss("y ~ g + x", family=NO(), data=data, method=method, max_iter=30,
                   contrasts="contr.treatment")
    np.testing.assert_allclose(
        np.ravel(model.coefficients["mu"]),
        np.asarray(golden["treatment"]["mu"], dtype=np.float64),
        rtol=1e-8,
        atol=1e-10,
    )


def test_mixed_route_no_longer_hits_the_cg_boundary_defect(golden, data):
    """Regression guard for D2.137.

    MIXED used to raise "initial_model linear predictor ... does not match the
    CG design matrix" on every fit: an RS model stopped after one outer cycle
    stores the entry linear predictor next to the coefficients of its single
    glim.fit, and the CG hand-off demanded the two agree to 1e-7.  With the
    hand-off re-deriving eta from the coefficients, MIXED reaches R's deviance.
    """
    model = gamlss("y ~ g + x", family=NO(), data=data, method="MIXED", max_iter=30,
                   contrasts="contr.treatment")
    assert float(model.g_dev) == pytest.approx(
        float(golden["treatment"]["deviance"]), rel=1e-6
    )


# ---------------------------------------------------------------------------
# Prediction-time rebuild (D2.136a)
#
# R reference: the same golden file, now carrying `pred_mu` for the two
# prediction rows (g = a/c, x = 0/1).  R's own numbers are identical under both
# codings — same column space — so a silent treatment-only rebuild would still
# pass a deviance check but fail this one.
# ---------------------------------------------------------------------------

PREDICTION_NEWDATA = {"g": np.array(["a", "c"]), "x": np.array([0.0, 1.0])}


@pytest.mark.parametrize(("scheme", "key"), [("contr.treatment", "treatment"), ("contr.sum", "sum")])
def test_factor_prediction_matches_live_r(golden, data, scheme, key):
    model = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
                   contrasts=scheme)
    params = predict_params(model, PREDICTION_NEWDATA)
    np.testing.assert_allclose(
        np.asarray(params["mu"], dtype=np.float64),
        np.asarray(golden[key]["pred_mu"], dtype=np.float64),
        rtol=1e-8,
        atol=1e-10,
    )


def test_prediction_uses_training_levels_not_batch_levels(golden, data):
    """A single-row batch must not collapse the 3-level contrast to 2 levels."""
    model = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30,
                   contrasts="contr.sum")
    single = predict_params(model, {"g": np.array(["c"]), "x": np.array([1.0])})
    batch = predict_params(model, PREDICTION_NEWDATA)
    assert float(single["mu"][0]) == pytest.approx(float(batch["mu"][1]), rel=1e-12)


def test_unseen_level_is_rejected_at_prediction(golden, data):
    from omnilss.prediction import PredictionSchemaError

    model = gamlss("y ~ g + x", family=NO(), data=data, method="RS", max_iter=30)
    with pytest.raises(PredictionSchemaError, match="unseen levels"):
        predict_params(model, {"g": np.array(["z"]), "x": np.array([0.0])})


def test_numeric_prediction_is_unchanged(golden, data):
    """The factor work must not disturb the plain numeric path."""
    numeric = {k: v for k, v in data.items() if k != "g"}
    model = gamlss("y ~ x", family=NO(), data=numeric, method="RS", max_iter=30)
    params = predict_params(model, {"x": np.array([0.0, 1.0])})
    beta = np.ravel(model.coefficients["mu"])
    np.testing.assert_allclose(
        np.asarray(params["mu"], dtype=np.float64),
        np.array([beta[0], beta[0] + beta[1]]),
        rtol=1e-12,
    )
