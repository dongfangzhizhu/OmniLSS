# SPDX-License-Identifier: GPL-3.0-or-later
"""M3 assessment snapshot (D2.245).

Pins the committed M3 artifact to its source artifacts and pins the verdict
logic: the gate must equal the conjunction of the six criteria, the two
behavioural criteria must equal their live-probe results, and the admission
criterion must equal the committed inventory counts rather than a literal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import m3_assessment as m3  # noqa: E402

ARTIFACT = m3.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_identifies_m3(artifact):
    assert ARTIFACT.is_file()
    assert artifact["milestone"] == "M3"
    assert artifact["target_version"] == "0.9.0"
    assert artifact["schema_version"] == m3.SCHEMA_VERSION
    assert artifact["decision"] == "D2.245"


def test_every_criterion_holds(artifact):
    assert len(artifact["criteria"]) == artifact["counts"]["criteria"]
    assert artifact["counts"]["met"] == len(artifact["criteria"])
    assert all(item["met"] for item in artifact["criteria"].values())


def test_gate_is_the_conjunction_of_the_criteria(artifact):
    criteria = artifact["criteria"]
    assert artifact["verdict"]["m3_gate_met"] is all(
        item["met"] for item in criteria.values()
    )


def test_behavioural_criteria_equal_their_live_probes(artifact):
    probes = artifact["probes"]
    assert artifact["criteria"]["gen_family_fits_like_r"]["met"] is probes["gen_family"]["ok"]
    assert artifact["criteria"]["fix_on_every_route"]["met"] is probes["fix_routes"]["ok"]
    assert artifact["criteria"]["categorical_prediction_rebuilt"]["met"] is (
        probes["categorical_prediction"]["ok"]
    )
    assert set(probes["fix_routes"]["routes"]) == {"RS", "CG", "MIXED"}
    assert set(probes["categorical_prediction"]["routes"]) == {"CG", "MIXED"}


def test_fix_probe_reports_sigma_as_unestimated_on_every_route(artifact):
    for route, row in artifact["probes"]["fix_routes"]["routes"].items():
        assert row["deviance_matches_r"] is True, route
        assert row["sigma_not_estimated"] is True, route
        assert row["sigma_frozen_at_start"] is True, route


def test_admission_criterion_reads_the_inventory(artifact):
    admission = json.loads(
        (REPO / "docs/development/m3-admission-evidence-2026-11-15.json").read_text(encoding="utf-8")
    )
    counts = artifact["counts"]
    assert counts["promoted_families"] == admission["counts"]["promoted_families"]
    assert counts["admitted_families"] == admission["counts"]["admitted"]
    assert artifact["criteria"]["admission_closed"]["met"] is (
        counts["admitted_families"] == counts["promoted_families"]
    )


def test_prior_milestone_verdicts_are_linked(artifact):
    m1 = json.loads((REPO / "docs/development/m1-assessment-2026-11-15.json").read_text(encoding="utf-8"))
    m2 = json.loads((REPO / "docs/development/m2-assessment-2026-11-15.json").read_text(encoding="utf-8"))
    assert artifact["criteria"]["prior_milestones_intact"]["met"] is (
        bool(m1["verdict"]["m1_gate_met"]) and bool(m2["verdict"]["m2_gate_met"])
    )


def test_artifact_stays_in_sync_with_the_tree():
    ok, problems = m3.check_artifact(ARTIFACT)
    assert ok, problems
