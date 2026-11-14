# SPDX-License-Identifier: GPL-3.0-or-later
"""P1-2: newdata prediction, JSON serialization round-trip, convergence schema (D2.229).

Every golden fixture under regenerated/ now carries, besides the coefficient
and deviance replay:
  - newdata    : a fixed five-point x grid (-1, -0.5, 0, 0.5, 1)
  - predicted  : R's response-scale parameter values on that grid
  - convergence: R's iter / converged / n_cyc

These tests replay the fit and assert
  (a) our newdata predictions match live R,
  (b) a JSON save/load round-trip preserves those predictions to 1e-10,
  (c) the convergence schema is present and our fit converges too.

Together with test_r_golden_highfreq (coefficients/deviance) this makes each
golden an end-to-end artifact: fit -> predict -> persist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss, load_model_json, save_model_json

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "r_golden" / "regenerated"
MANIFEST = json.loads((GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
CASES = tuple(sorted(MANIFEST))

_MODEL_CACHE: dict[str, tuple[object, dict]] = {}


def _fit(case: str):
    """Fit once per case (module cache) and return (model, payload)."""
    cached = _MODEL_CACHE.get(case)
    if cached is not None:
        return cached
    entry = MANIFEST[case]
    payload = json.loads((GOLDEN_DIR / f"{case}.json").read_text(encoding="utf-8"))
    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    kwargs = {}
    if "n_cyc" in entry:
        kwargs["max_iter"] = int(entry["n_cyc"])
    model = gamlss(payload["formula"], family=entry["family"], data=data, **kwargs)
    _MODEL_CACHE[case] = (model, payload)
    return model, payload


def test_every_case_records_the_p1_2_evidence_axis():
    """A new golden may not omit newdata/predicted/convergence."""
    for case in CASES:
        payload = json.loads((GOLDEN_DIR / f"{case}.json").read_text(encoding="utf-8"))
        assert "newdata" in payload and "x" in payload["newdata"], case
        assert isinstance(payload["predicted"], dict) and payload["predicted"], case
        convergence = payload["convergence"]
        for key in ("iter", "converged", "n_cyc"):
            assert key in convergence, (case, key)


@pytest.mark.parametrize("case", CASES)
def test_newdata_predictions_match_live_r(case):
    model, payload = _fit(case)
    newdata = {"x": np.asarray(payload["newdata"]["x"], dtype=float)}
    predicted = model.predict_params(newdata)
    for param, reference in payload["predicted"].items():
        assert param in predicted, (case, param)
        np.testing.assert_allclose(
            np.asarray(predicted[param], dtype=float),
            np.asarray(reference, dtype=float),
            rtol=2e-4,
            atol=2e-4,
            err_msg=f"{case} newdata prediction for {param}",
        )


@pytest.mark.parametrize("case", CASES)
def test_json_roundtrip_preserves_newdata_predictions(case, tmp_path):
    model, payload = _fit(case)
    newdata = {"x": np.asarray(payload["newdata"]["x"], dtype=float)}
    before = model.predict_params(newdata)

    path = tmp_path / f"{case}.omnilss"
    save_model_json(model, path)
    restored = load_model_json(path)
    after = restored.predict_params(newdata)

    assert set(after) == set(before), case
    for param in before:
        np.testing.assert_allclose(
            np.asarray(after[param], dtype=float),
            np.asarray(before[param], dtype=float),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{case} round-trip drift for {param}",
        )


@pytest.mark.parametrize("case", CASES)
def test_convergence_schema_is_consistent(case):
    model, payload = _fit(case)
    convergence = payload["convergence"]
    # R-side schema sanity
    assert isinstance(convergence["iter"], int) and convergence["iter"] >= 1, case
    assert convergence["converged"] is True, case
    assert isinstance(convergence["n_cyc"], int) and convergence["n_cyc"] >= 1, case
    # Our fit must also converge and report a positive iteration count.  We do
    # not assert iter equality: the two optimisers may take a different number
    # of cycles to the same optimum (that is exactly what n_cyc=100 pins).
    assert int(getattr(model, "iter", 0)) >= 1, case
    assert bool(getattr(model, "converged", False)) is True, case
