# SPDX-License-Identifier: GPL-3.0-or-later
"""M3 production_safe admission evidence inventory (D2.240).

Pins the committed inventory to the committed artifacts and pins the *logic*:
a family is admitted only if every hard gate is green, the registry audit must
back every claim in the three capability sets, and the counts must equal the
row statuses.  The tool is an inventory, so the tests also assert that it does
not silently change what is admitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "omnilss" / "src"))
sys.path.insert(0, str(REPO / "tools"))

import m3_admission_evidence as m3  # noqa: E402
from omnilss.family_capabilities import _PRODUCTION_SAFE  # noqa: E402

ARTIFACT = m3.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_exists_and_identifies_m3(artifact):
    assert ARTIFACT.is_file()
    assert artifact["milestone"] == "M3"
    assert artifact["target_version"] == "0.9.0"
    assert artifact["schema_version"] == m3.SCHEMA_VERSION


def test_scope_is_exactly_the_frozen_promoted_families(artifact):
    frozen = json.loads(
        (REPO / "docs/development/highfreq-families-frozen.json").read_text(encoding="utf-8")
    )
    promoted = sorted(item["family"] for item in frozen["tiers"]["promoted"])
    assert [row["family"] for row in artifact["families"]] == promoted


def test_every_row_reports_every_hard_gate(artifact):
    for row in artifact["families"]:
        assert set(row["hard_gates"]) == set(m3.HARD_GATES), row["family"]
        assert row["status"] in {"admitted", "eligible", "blocked"}
        assert set(row["soft_flags"]) == set(m3.SOFT_FLAGS), row["family"]


def test_status_follows_the_hard_gates_and_the_registry(artifact):
    for row in artifact["families"]:
        all_green = all(row["hard_gates"].values())
        if row["family"] in _PRODUCTION_SAFE:
            assert row["status"] == "admitted", row["family"]
        elif all_green:
            assert row["status"] == "eligible", row["family"]
        else:
            assert row["status"] == "blocked", row["family"]
            assert row["missing_gates"], row["family"]


def test_admitted_families_are_the_registry_set(artifact):
    admitted = sorted(row["family"] for row in artifact["families"] if row["status"] == "admitted")
    assert admitted == sorted(name for name in _PRODUCTION_SAFE if name in {
        row["family"] for row in artifact["families"]
    })
    assert artifact["counts"]["admitted"] == len(admitted)


def test_counts_and_summary_match_the_rows(artifact):
    rows = artifact["families"]
    counts = artifact["counts"]
    assert counts["promoted_families"] == len(rows)
    for status, key in (("admitted", "admitted"), ("eligible", "eligible_not_admitted"), ("blocked", "blocked")):
        assert counts[key] == sum(1 for row in rows if row["status"] == status), status
    assert sorted(artifact["summary"]["eligible_not_admitted"]) == sorted(
        row["family"] for row in rows if row["status"] == "eligible"
    )


def test_registry_audit_backs_every_claim(artifact):
    audit = artifact["registry_audit"]
    assert audit["violations"] == []
    assert artifact["verdict"]["registry_claims_backed"] is True
    declared = {item["family"] for block in audit["sets"].values() for item in block["declared"]}
    assert {"NO", "ZIP"} <= declared
    for block in audit["sets"].values():
        assert all(item["backed"] for item in block["declared"]), block


def test_artifact_stays_in_sync_with_the_tree(artifact):
    ok, problems = m3.check_artifact(ARTIFACT)
    assert ok, problems
