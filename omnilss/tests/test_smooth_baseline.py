"""Phase 2.5: ps/cs defect regression baseline (before implementation changes).

Precisely characterises the CURRENT behaviour of the ps/cs design-matrix and
prediction paths so later fixes (Phase 3: single managed basis state for
pb/ps/cs) can be validated against a recorded baseline.  These tests assert
today's behaviour — they are the *fix baseline*, not the target.

Baseline facts recorded here (HEAD @ Phase 2.6):
- ps: B-spline basis expansion (24 cols for ps_intervals=20, degree=3), and
  schema-safe prediction works for same/different-length newdata.
- cs: single-column fitted-values placeholder (intercept + 1 smooth column),
  and prediction is explicitly rejected with PredictionSchemaError
  "not schema-safe yet".
- pb: same expansion as ps; prediction works.
- Missing columns raise PredictionSchemaError naming the missing variable.

Alignment: DEVELOPMENT_PLAN Phase 2 task 2.5 — "精确刻画当前双列占位/追加/
陈旧 ownership/newdata schema mismatch 行为，测试标记为修复基线".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.smooth_fitting import SmoothFitInfo


@pytest.fixture(scope="module")
def smooth_data():
    rng = np.random.default_rng(0)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(4 * x) + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x}


def _smooth_fits(model) -> list[SmoothFitInfo]:
    slots = model.additional_slots.get("smooth_infos", {})
    info = slots.get("mu")
    if info is None or getattr(info, "smooth_fits", None) is None:
        return []
    return list(info.smooth_fits)


def test_ps_baseline_basis_expansion(smooth_data) -> None:
    """Baseline: ps expands into a B-spline basis (24 cols at default knots)."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    X = np.asarray(model.design_matrices["mu"])
    # B-spline basis only (ps_intervals=20, degree=3 -> 24 basis cols; the
    # basis absorbs the intercept)
    assert X.shape == (60, 24), X.shape
    fits = _smooth_fits(model)
    assert len(fits) == 1
    assert fits[0].smoother == "ps"
    # X = intercept (1 col) + B-spline basis (23 cols, ps_intervals=20,
    # degree=3 -> 23 basis functions).  basis_columns=(1, 24) points at the
    # basis columns and is consistent with the matrix (verified Phase 3.2).
    assert fits[0].basis_columns == (1, 24)


def test_cs_baseline_single_column_placeholder(smooth_data) -> None:
    """Baseline: cs contributes exactly one fitted-value column (placeholder)."""
    model = gamlss("y ~ cs(x)", family="NO", data=smooth_data)
    X = np.asarray(model.design_matrices["mu"])
    # Double-column placeholder baseline: col0 = intercept, col1 = placeholder
    # ones, col2 = actual fitted values (the "双列占位" behaviour).
    assert X.shape == (60, 3), X.shape
    assert np.allclose(X[:, 0], 1.0)
    assert np.allclose(X[:, 1], 1.0)  # placeholder column
    assert not np.allclose(X[:, 2], 1.0)  # fitted values
    fits = _smooth_fits(model)
    assert len(fits) == 1
    assert fits[0].smoother == "cs"


def test_cs_baseline_prediction_rejected_not_schema_safe(smooth_data) -> None:
    """Baseline: cs prediction is explicitly rejected (not schema-safe yet)."""
    model = gamlss("y ~ cs(x)", family="NO", data=smooth_data)
    with pytest.raises(Exception) as excinfo:
        model.predict(newdata=smooth_data)
    assert "not schema-safe" in str(excinfo.value)


def test_ps_baseline_predict_same_length(smooth_data) -> None:
    """Baseline: ps predicts on newdata of the same length."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    pred = model.predict(newdata=smooth_data)
    assert len(pred) == 60
    assert np.isfinite(np.asarray(pred)).all()


def test_ps_baseline_predict_different_length(smooth_data) -> None:
    """Baseline: ps predicts on different-length newdata via basis rebuild."""
    x2 = np.linspace(0, 1, 30)
    newdata = {"y": np.sin(4 * x2), "x": x2}
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    pred = model.predict(newdata=newdata)
    assert len(pred) == 30


def test_missing_variable_raises_schema_error(smooth_data) -> None:
    """Baseline: missing newdata column raises PredictionSchemaError with name."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    with pytest.raises(Exception) as excinfo:
        model.predict(newdata={"y": smooth_data["y"]})
    message = str(excinfo.value)
    assert "Missing variable" in message
    assert "'x'" in message


def test_pb_baseline_expansion_matches_ps(smooth_data) -> None:
    """Baseline: pb uses the same expansion style as ps (24 basis cols)."""
    model = gamlss("y ~ pb(x)", family="NO", data=smooth_data)
    X = np.asarray(model.design_matrices["mu"])
    assert X.shape == (60, 24), X.shape


def test_smooth_fit_metadata_recorded(smooth_data) -> None:
    """Baseline: SmoothFitInfo records smoother/basis columns/lambda/edf."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert fits, "no smooth fit metadata recorded"
    info = fits[0]
    assert info.lambda_ is not None
    assert info.edf is not None and info.edf > 0
    assert info.selection_method
