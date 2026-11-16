# SPDX-License-Identifier: GPL-3.0-or-later
"""P3-1 frozen contracts (D2.242).

Pins the committed contract artifact to the live package and pins the logic:
every supported link must have a probe (a new link without one is a surface
change nobody froze), the documented inverse offsets must stay declared, the
family invariant must cover both fixed-parameter shapes, and the prediction
behaviours must be true rather than assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "omnilss" / "src"))
sys.path.insert(0, str(REPO / "tools"))

import contract_freeze as freeze  # noqa: E402

ARTIFACT = freeze.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_identifies_p3_1(artifact):
    assert ARTIFACT.is_file()
    assert artifact["milestone"] == "M3"
    assert artifact["task"] == "P3-1"
    assert artifact["decision"] == "D2.242"
    assert artifact["schema_version"] == freeze.SCHEMA_VERSION


def test_every_area_holds_and_the_contract_is_frozen(artifact):
    assert set(artifact["areas"]) == {"family", "link", "formula", "prediction"}
    assert all(item["met"] for item in artifact["criteria"].values())
    assert artifact["verdict"]["contract_frozen"] is True


def test_every_supported_link_has_a_probe(artifact):
    link = artifact["areas"]["link"]
    assert link["supported_links"] == sorted(freeze.LINK_PROBES)
    assert sorted(link["probes"]) == link["supported_links"]
    for name, probe in link["probes"].items():
        assert probe["roundtrip_within_tolerance"] is True, name
        assert probe["mu_eta_finite_nonzero"] is True, name


def test_documented_shifted_log_offsets_stay_declared(artifact):
    probes = artifact["areas"]["link"]["probes"]
    assert probes["logshiftto1"]["offset"] == 1e-5
    assert probes["logshiftto2"]["offset"] == 1e-5
    assert probes["logshiftto0"]["offset"] == 0.0
    assert probes["logit"]["offset"] == 0.0


def test_family_invariant_covers_both_fixed_parameter_shapes(artifact):
    families = artifact["areas"]["family"]["families"]
    assert len(families) == len(freeze._promoted())
    for name, row in families.items():
        assert row["nopar"] == len(row["parameters"]), name
        assert row["estimable_parameters"] == [
            p for p in row["parameters"] if p not in row["fixed_parameters"]
        ], name
        assert row["density_callable"] and row["cdf_callable"], name
        assert row["quantile_callable"] and row["sampler_callable"], name
    # The two shapes that motivated the invariant.
    assert families["BI"]["fixed_parameters"] == ["bd"]
    assert "bd" not in families["BI"]["parameters"]
    assert families["NET"]["fixed_parameters"] == ["nu", "tau"]
    assert "nu" in families["NET"]["parameters"]


def test_formula_cases_cover_linear_smooth_tensor_and_intercept(artifact):
    cases = {case["formula"]: case for case in artifact["areas"]["formula"]["cases"]}
    assert cases["y ~ x"]["linear_variables"] == ["x"]
    assert cases["y ~ x - 1"]["has_intercept"] is False
    assert cases["y ~ 1"]["linear_variables"] == []
    assert cases["y ~ pb(x)"]["smooth_terms"] == [{"smoother": "pb", "variable": "x"}]
    assert cases["y ~ te(x, z)"]["tensor_terms"] == [
        {"smoother": "te", "variables": ["x", "z"]}
    ]


def test_prediction_behaviours_are_recorded_as_true(artifact):
    prediction = artifact["areas"]["prediction"]
    for key in (
        "schema_error_is_value_error",
        "keys_are_estimable_parameters",
        "values_follow_newdata_length",
        "missing_column_raises_schema_error",
        "extra_column_tolerated",
        "predict_response_length_follows_newdata",
    ):
        assert prediction[key] is True, key
    assert prediction["missing_column_error"] == "PredictionSchemaError"


def test_artifact_stays_in_sync_with_the_live_package():
    ok, problems = freeze.check_artifact(ARTIFACT)
    assert ok, problems


def test_ci_and_release_gates_run_the_contract_check():
    for workflow in ("ci.yml", "release.yml"):
        text = (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "tools/contract_freeze.py --check" in text, workflow
