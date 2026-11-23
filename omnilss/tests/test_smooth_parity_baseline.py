# SPDX-License-Identifier: GPL-3.0-or-later
"""Smooth parity baseline (D2.308/R3).

Thresholds for the R-vs-Python smooth tests must come from a recorded baseline,
not ad-hoc literals.  The limits must be no looser than the legacy thresholds
and the recorded observations must satisfy them.
"""
from __future__ import annotations

import json
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[2] / "docs" / "development" / "smooth-parity-baseline.json"

LEGACY_CORR = {
    "test_pb_in_mu_normal": 0.99,
    "test_pb_in_mu_gamma": 0.98,
    "test_pb_in_sigma_normal": 0.80,
    "test_mixed_linear_and_smooth": 0.99,
    "test_multiple_smooths": 0.98,
    "test_gamlss_iterative_with_smooth": 0.99,
}
LEGACY_REL = {
    ("test_pb_in_mu_normal", "deviance_rel_max"): 0.30,
    ("test_mixed_linear_and_smooth", "coef_x1_rel_max"): 0.05,
}
OBSERVED = {
    ("test_pb_in_mu_normal", "corr"): "corr_observed",
    ("test_pb_in_mu_normal", "deviance_rel"): "deviance_rel_observed",
    ("test_pb_in_mu_gamma", "corr"): "corr_observed",
    ("test_pb_in_sigma_normal", "sigma_corr"): "sigma_corr_observed",
    ("test_mixed_linear_and_smooth", "coef_x1_rel"): "coef_x1_rel_observed",
    ("test_mixed_linear_and_smooth", "corr"): "corr_observed",
    ("test_multiple_smooths", "corr"): "corr_observed",
    ("test_gamlss_iterative_with_smooth", "corr"): "corr_observed",
}


def _load() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_schema_and_source() -> None:
    data = _load()
    assert data["schema_version"] == 1
    assert "gamlss.dist 6.1-1" in data["source"]
    assert data["formula"]


def test_every_smooth_test_has_a_baseline() -> None:
    tests = _load()["tests"]
    assert set(tests) == set(LEGACY_CORR)


def test_limits_are_not_looser_than_legacy() -> None:
    tests = _load()["tests"]
    for name, legacy in LEGACY_CORR.items():
        assert tests[name]["corr_min"] >= legacy, name
    for (name, key), legacy in LEGACY_REL.items():
        assert tests[name][key] <= legacy, (name, key)


def test_recorded_observations_satisfy_limits() -> None:
    tests = _load()["tests"]
    for (name, metric), observed_key in OBSERVED.items():
        observed = tests[name][observed_key]
        if metric.endswith("corr"):
            assert observed >= tests[name]["corr_min"], (name, metric)
        else:
            key = "deviance_rel_max" if metric == "deviance_rel" else "coef_x1_rel_max"
            assert observed <= tests[name][key], (name, metric)
