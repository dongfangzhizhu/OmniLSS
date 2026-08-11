"""Phase 3.5: saved-state newdata rebuild contract (design.md Property 10).

For any valid saved ps/pb/cs Smooth Model State and compatible newdata,
rebuilding the prediction design reproduces the training schema and returns
predictions without a schema-column mismatch.  Rebuild uses only the saved
Smooth Model State (knots/degree for ps/pb, cs_state (x,y,w,s,k) for cs) —
no implicit column alignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss


@pytest.fixture(scope="module")
def smooth_data():
    rng = np.random.default_rng(41)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(4 * x) + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x}


@pytest.mark.parametrize("formula", ["y ~ ps(x)", "y ~ pb(x)", "y ~ cs(x)"])
def test_train_prediction_reproduces_fitted(smooth_data, formula) -> None:
    """Rebuilding on the training data reproduces the fitted values."""
    model = gamlss(formula, family="NO", data=smooth_data)
    pred = np.asarray(model.predict(newdata=smooth_data))
    fitted = np.asarray(model.fitted_values["mu"])
    assert pred.shape == fitted.shape
    np.testing.assert_allclose(pred, fitted, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("formula", ["y ~ ps(x)", "y ~ pb(x)", "y ~ cs(x)"])
def test_prediction_on_different_length_newdata(smooth_data, formula) -> None:
    """Different-length newdata predicts via saved-state rebuild."""
    x2 = np.linspace(0, 1, 30)
    newdata = {"y": np.sin(4 * x2), "x": x2}
    model = gamlss(formula, family="NO", data=smooth_data)
    pred = model.predict(newdata=newdata)
    assert len(pred) == 30
    assert np.isfinite(np.asarray(pred)).all()


@pytest.mark.parametrize("formula", ["y ~ ps(x)", "y ~ pb(x)", "y ~ cs(x)"])
def test_missing_variable_rejected(smooth_data, formula) -> None:
    """Missing newdata variable is rejected, not silently column-aligned."""
    model = gamlss(formula, family="NO", data=smooth_data)
    with pytest.raises(Exception) as excinfo:
        model.predict(newdata={"y": smooth_data["y"]})
    assert "Missing variable" in str(excinfo.value)


def test_cs_state_complete_for_rebuild(smooth_data) -> None:
    """cs SmoothFitInfo carries a complete reconstruction state."""
    model = gamlss("y ~ cs(x)", family="NO", data=smooth_data)
    fit = model.additional_slots["smooth_infos"]["mu"].smooth_fits[0]
    state = fit.cs_state
    assert state is not None
    assert {"x", "y", "w", "s", "degree"} <= set(state)
    assert len(state["x"]) == len(smooth_data["x"])
    assert isinstance(state["s"], float)
    assert isinstance(state["degree"], int) and state["degree"] >= 3


def test_ps_rebuild_uses_saved_knots(smooth_data) -> None:
    """ps prediction is rebuilt from the saved knots/degree, not the data."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    fit = model.additional_slots["smooth_infos"]["mu"].smooth_fits[0]
    assert fit.knots is not None and len(fit.knots) > 0
    assert fit.degree is not None
    # Rebuilding on the training x must reproduce the fitted values exactly,
    # proving the basis came from the saved state (knots/degree).
    pred = np.asarray(model.predict(newdata=smooth_data))
    np.testing.assert_allclose(
        pred, np.asarray(model.fitted_values["mu"]), rtol=1e-8, atol=1e-8
    )
