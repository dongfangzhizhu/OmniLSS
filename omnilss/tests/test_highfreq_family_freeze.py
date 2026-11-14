# SPDX-License-Identifier: GPL-3.0-or-later
"""P1-1: frozen high-frequency family list (D2.230).

The artifact docs/development/highfreq-families-frozen.json classifies every
family in omnilss.family_capabilities._R_CONSISTENCY_EVIDENCE into exactly one
tier: promoted (has a golden case), pending (not probed), excluded (probed and
rejected, with a reason).  These tests pin the partition and keep it in sync
with the golden manifest and the validated set, so adding a validated family
without classifying it fails the build instead of silently drifting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import highfreq_family_freeze as freeze  # noqa: E402

ARTIFACT = freeze.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_exists_and_identifies_the_freeze():
    assert ARTIFACT.is_file()
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert data["schema_version"] == freeze.SCHEMA_VERSION
    assert "P1-1" in data["purpose"]


def test_every_validated_family_is_classified_exactly_once(artifact):
    counts = artifact["counts"]
    assert (
        counts["promoted"] + counts["pending"] + counts["excluded"]
        == counts["total_validated"]
    )
    tiers = artifact["tiers"]
    families = [item["family"] for tier in tiers.values() for item in tier]
    assert len(families) == len(set(families)), "a family appears in two tiers"
    from omnilss.family_capabilities import _R_CONSISTENCY_EVIDENCE

    assert set(families) == set(_R_CONSISTENCY_EVIDENCE)


def test_promoted_tier_equals_families_with_golden_cases(artifact):
    promoted = {item["family"] for item in artifact["tiers"]["promoted"]}
    assert promoted == set(freeze._golden_families())


def test_promoted_cases_point_at_the_golden_manifest(artifact):
    golden = freeze._golden_families()
    for item in artifact["tiers"]["promoted"]:
        assert item["golden_cases"] == golden[item["family"]], item["family"]


def test_every_excluded_family_records_a_reason(artifact):
    for item in artifact["tiers"]["excluded"]:
        assert item.get("code"), item
        assert item.get("note"), item


def test_artifact_stays_in_sync_with_the_current_tree():
    ok, problems = freeze.check_artifact(ARTIFACT)
    assert ok, problems


def test_check_cli_exit_code_is_zero():
    assert freeze.main(["--check"]) == 0


def test_ci_and_release_gates_run_the_freeze_check():
    for workflow in ("ci.yml", "release.yml"):
        text = (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "tools/highfreq_family_freeze.py --check" in text, workflow
