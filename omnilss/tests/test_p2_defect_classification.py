# SPDX-License-Identifier: GPL-3.0-or-later
"""P2 defect classification (D2.236).

Pins the committed classification artifact to its source scan and pins the
classification *structure* rather than today's family lists: every non-pass
entry must carry a category, categories must partition the non-pass set for
each gate, and the verdict must be true only when nothing is unclassified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import p2_defect_classification as p2  # noqa: E402

ARTIFACT = p2.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_identifies_its_source_scan(artifact):
    assert ARTIFACT.is_file()
    assert artifact["schema_version"] == p2.SCHEMA_VERSION
    scan_path = REPO / artifact["source"]["scan"]
    assert scan_path.is_file()
    assert p2._sha256(scan_path) == artifact["source"]["scan_sha256"]
    assert artifact["source"]["families"] > 0


def test_entries_cover_every_family_and_gate(artifact):
    families = artifact["source"]["families"]
    assert len(artifact["entries"]) == families * len(p2.GATES)
    for gate in p2.GATES:
        total = sum(artifact["counts"]["status"][gate].values())
        assert total == families, gate


def test_every_non_pass_entry_has_exactly_one_category(artifact):
    for gate in p2.GATES:
        non_pass = artifact["counts"]["non_pass"][gate]
        categorised = sum(len(fams) for fams in artifact["categories"][gate].values())
        assert categorised == non_pass, gate
    # No entry outside the pass status may keep the pass code.
    for entry in artifact["entries"]:
        if entry["status"] != "pass":
            assert entry["code"] != "pass", entry


def test_every_category_member_is_actually_non_pass(artifact):
    non_pass = {
        (entry["gate"], entry["family"])
        for entry in artifact["entries"]
        if entry["status"] != "pass"
    }
    listed = {
        (gate, family)
        for gate, cats in artifact["categories"].items()
        for families in cats.values()
        for family in families
    }
    assert listed == non_pass


def test_verdict_is_true_only_when_nothing_is_unclassified(artifact):
    assert artifact["verdict"]["all_failures_classified"] == (not artifact["unclassified"])
    assert artifact["unclassified"] == []
    assert artifact["verdict"]["all_failures_classified"] is True


def test_artifact_stays_in_sync_with_the_recorded_scan():
    ok, problems = p2.check_artifact(ARTIFACT)
    assert ok, problems


def test_check_cli_exit_code_is_zero():
    assert p2.main(["--check"]) == 0
