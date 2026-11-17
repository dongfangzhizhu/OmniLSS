# SPDX-License-Identifier: GPL-3.0-or-later
"""M4 / P4 scope inventory (D2.246).

Pins the inventory to the committed coverage artifact and pins the ordering:
the priority groups must match the plan, the first work item must be the first
surface gap by (priority, plan order), and every counted number must equal the
per-package rows rather than being typed in twice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import m4_scope_inventory as m4  # noqa: E402

ARTIFACT = m4.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_identifies_the_p4_scope(artifact):
    assert ARTIFACT.is_file()
    assert artifact["milestone"] == "M4"
    assert artifact["task"] == "P4 scope"
    assert artifact["decision"] == "D2.246"
    assert artifact["schema_version"] == m4.SCHEMA_VERSION


def test_every_surrounding_package_is_classified(artifact):
    coverage = json.loads(
        (REPO / "docs/development/r-symbol-coverage.json").read_text(encoding="utf-8")
    )
    known = {row["package"] for row in artifact["packages"]} | set(artifact["out_of_scope"])
    in_scope = {
        name
        for name, meta in coverage["stats"]["per_package"].items()
        if meta.get("in_scope") or name in artifact["out_of_scope"]
    }
    assert in_scope <= known, sorted(in_scope - known)
    # The core packages belong to M1/M3, not P4.
    assert {"gamlss", "gamlss.dist"} <= set(artifact["out_of_scope"])


def test_priority_groups_follow_the_plan(artifact):
    groups = artifact["priority_groups"]
    assert groups["1"] == ["gamlss.cens", "gamlss.tr"]
    assert groups["2"] == ["gamlss.mx", "gamlss.add"]
    assert groups["3"] == ["gamlss.spatial", "gamlss.foreach", "gamlss.prepdata"]
    assert groups["4"] == ["gamlss.ggplots", "gamlss.data"]


def test_counts_equal_the_package_rows(artifact):
    rows = artifact["packages"]
    counts = artifact["counts"]
    assert counts["packages"] == len(rows)
    assert counts["missing_symbols"] == sum(len(row["missing_symbols"]) for row in rows)
    for status, key in (
        ("surface_complete", "surface_complete"),
        ("surface_gap", "surface_gap"),
        ("no_function_surface", "no_function_surface"),
    ):
        assert counts[key] == sum(1 for row in rows if row["status"] == status), status
    assert counts["packages_without_evidence_tests"] == sum(
        1
        for row in rows
        if not row["evidence_tests"] and row["status"] != "no_function_surface"
    )


def test_status_and_missing_symbols_agree(artifact):
    for row in artifact["packages"]:
        if row["status"] == "surface_gap":
            assert row["missing_symbols"], row["package"]
        else:
            assert row["missing_symbols"] == [], row["package"]
        if row["status"] == "no_function_surface":
            assert row["exported"] == 0 and row["tracked_symbols"] == 0


def test_evidence_modules_recorded_actually_exist(artifact):
    for row in artifact["packages"]:
        for test in row["evidence_tests"]:
            assert (REPO / "omnilss" / "tests" / test).is_file(), (row["package"], test)


def test_first_work_item_is_the_first_gap_in_plan_order(artifact):
    ordered = sorted(
        (row for row in artifact["packages"] if row["missing_symbols"]),
        key=lambda row: (row["priority"], row["order"]),
    )
    expected = ordered[0]
    first = artifact["verdict"]["first_work_item"]
    assert first["package"] == expected["package"]
    assert first["priority"] == expected["priority"]
    assert first["missing"] == len(expected["missing_symbols"])


def test_cens_and_tr_are_surface_complete_with_evidence(artifact):
    by_name = {row["package"]: row for row in artifact["packages"]}
    for name in ("gamlss.cens", "gamlss.tr"):
        assert by_name[name]["status"] == "surface_complete", name
        assert by_name[name]["evidence_tests"], name


def test_artifact_stays_in_sync_with_the_coverage_artifact():
    ok, problems = m4.check_artifact(ARTIFACT)
    assert ok, problems
