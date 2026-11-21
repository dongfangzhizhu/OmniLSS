# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the M4 (1.0.0) release-readiness assessment gate (D2.283/B4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "omnilss" / "src"))
sys.path.insert(0, str(REPO / "tools"))

import m4_assessment as m4  # noqa: E402

ARTIFACT = m4.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_m4_artifact_exists_and_identifies_1_0_0(artifact):
    assert ARTIFACT.is_file()
    assert ARTIFACT.with_suffix(".md").is_file()
    assert artifact["milestone"] == "M4"
    assert artifact["target_version"] == "1.0.0"
    assert artifact["schema_version"] == m4.SCHEMA_VERSION


def test_m4_gate_is_met_on_all_criteria(artifact):
    assert artifact["verdict"]["m4_gate_met"] is True
    assert artifact["counts"]["met"] == artifact["counts"]["criteria"]
    for name, item in artifact["criteria"].items():
        assert item["met"] is True, (name, item)


def test_m4_enforces_zero_reference_repo_imports(artifact):
    assert artifact["checks"]["reference_import_violations"] == []


def test_m4_artifact_stays_in_sync_with_tree():
    ok, problems = m4.check_artifact(ARTIFACT)
    assert ok, problems
