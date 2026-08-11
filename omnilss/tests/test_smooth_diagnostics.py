"""Phase 3.4: RS smoother lambda-update diagnostics (design.md Property 9).

For any smoother update performed during an RS iteration, model diagnostics
must include old lambda, new lambda, EDF, working-response summary, weight
summary, and solver status for that term.
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
    rng = np.random.default_rng(31)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(4 * x) + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x}


def _diag(model) -> list[dict]:
    return list(model.additional_slots.get("smooth_lambda_diagnostics", ()))


def test_diagnostics_recorded_with_required_fields(smooth_data) -> None:
    """Every update record carries old/new lambda, EDF, summaries, status."""
    model = gamlss("y ~ ps(x) + cs(x, df=5)", family="NO", data=smooth_data)
    records = _diag(model)
    assert records, "no smoother update diagnostics recorded"
    required = {
        "parameter", "iteration", "smoother", "old_lambda", "new_lambda",
        "edf", "lambda_origin", "working_response", "weights", "solver_status",
    }
    for record in records:
        assert required <= set(record), f"missing fields in {record}"
        for summary_key in ("working_response", "weights"):
            summary = record[summary_key]
            assert {"min", "mean", "max", "n_nonfinite"} <= set(summary)
        assert record["solver_status"] in {"preserved", "reestimated", "failed"}


def test_method_selected_smoother_reestimated(smooth_data) -> None:
    """ps() without df/lambda is re-estimated during RS iterations."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    records = [r for r in _diag(model) if r["smoother"] == "ps"]
    assert records
    assert all(r["lambda_origin"] == "method_selected" for r in records)
    # At least one re-estimation changed the lambda.
    assert any(r["solver_status"] == "reestimated" for r in records)
    assert any(
        abs(float(r["old_lambda"]) - float(r["new_lambda"])) > 1e-12
        for r in records
    )


def test_fixed_origin_smoother_preserved(smooth_data) -> None:
    """df-targeted / user-fixed lambdas are preserved (not re-estimated)."""
    model = gamlss("y ~ ps(x, lambda=0.8) + cs(x, df=5)", family="NO", data=smooth_data)
    records = _diag(model)
    assert records
    for record in records:
        assert record["solver_status"] == "preserved"
        assert abs(float(record["old_lambda"]) - float(record["new_lambda"])) < 1e-12
        assert record["lambda_origin"] in {"user_fixed", "df_target"}


def test_diagnostics_cover_each_iteration(smooth_data) -> None:
    """Update records span the RS iterations with increasing iteration ids."""
    model = gamlss("y ~ ps(x)", family="NO", data=smooth_data)
    records = _diag(model)
    iterations = sorted({r["iteration"] for r in records})
    assert len(iterations) >= 2  # multiple outer iterations performed updates
    assert iterations == sorted(iterations)


def test_no_diagnostics_without_smooths(smooth_data) -> None:
    """A linear-only fit records no smoother update diagnostics."""
    model = gamlss("y ~ x", family="NO", data={"y": smooth_data["y"], "x": smooth_data["x"]})
    assert _diag(model) == []
