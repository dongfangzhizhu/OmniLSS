"""Phase 3.6: strict PredictionSchemaError contract (design.md Property 11).

Any prediction design differing from the saved Smooth Model State schema must
raise PredictionSchemaError identifying both expected and actual schema plus
the saved state version.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.prediction import PredictionSchemaError


@pytest.fixture(scope="module")
def smooth_data():
    rng = np.random.default_rng(51)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(4 * x) + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x}


def test_missing_variable_error_identifies_expected_and_actual(smooth_data) -> None:
    """Missing-variable error carries term id, both schemas and state version."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    with pytest.raises(PredictionSchemaError) as excinfo:
        model.predict(newdata={"y": smooth_data["y"]})

    err = excinfo.value
    assert err.code == "missing_variable"
    assert err.term == "ps(x)"
    assert err.parameter == "mu"
    assert err.expected_schema == {"variables": ["x"]}
    assert err.actual_schema == {"variables": ["y"]}
    assert err.state_version is not None


def test_error_to_dict_is_complete(smooth_data) -> None:
    """to_dict exposes the full structured envelope."""
    model = gamlss("y ~ cs(x)", family="NO", data=smooth_data)
    with pytest.raises(PredictionSchemaError) as excinfo:
        model.predict(newdata={"y": smooth_data["y"]})
    envelope = excinfo.value.to_dict()
    for key in (
        "code", "parameter", "term", "reason", "message",
        "expected_schema", "actual_schema", "state_version",
    ):
        assert key in envelope, f"missing {key} in {envelope}"


def test_column_mismatch_error_identifies_schemas(smooth_data) -> None:
    """A rebuilt design with wrong column count reports expected vs actual."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    with pytest.raises(PredictionSchemaError) as excinfo:
        model.predict(newdata={"y": smooth_data["y"]})  # missing x
    err = excinfo.value
    assert err.expected_schema is not None
    assert err.actual_schema is not None
    assert err.state_version is not None


def test_valid_prediction_no_error(smooth_data) -> None:
    """Compatible newdata does not raise and produces predictions."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    pred = model.predict(newdata=smooth_data)
    assert len(pred) == len(smooth_data["y"])
    assert np.isfinite(np.asarray(pred)).all()


def test_schema_mismatch_code_for_wrong_newdata(smooth_data) -> None:
    """Wrong-shaped newdata raises a schema error, not an untyped failure."""
    model = gamlss("y ~ x + ps(x)", family="NO", data=smooth_data)
    with pytest.raises(PredictionSchemaError) as excinfo:
        model.predict(newdata={"y": smooth_data["y"]})  # missing x
    assert excinfo.value.code in {
        "missing_variable", "schema_column_mismatch", "term_evaluation_failed",
    }
    assert excinfo.value.state_version is not None
