# SPDX-License-Identifier: GPL-3.0-or-later
"""M1 assessment snapshot (D2.234).

Pins the M1 wrap-up artifact to the committed coverage and freeze artifacts,
and pins the *logic* of the verdict rather than today's booleans: the growth
criterion must equal (growth_ratio >= 10), the kernel criterion must equal
(kernel-implemented families == promoted families), and core_m1_parity_met
must equal the conjunction of the three parity criteria.  That way the test
keeps the assessment honest across future changes instead of freezing a
snapshot that would silently become a lie.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import m1_assessment as m1  # noqa: E402

ARTIFACT = m1.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_exists_and_identifies_m1(artifact):
    assert ARTIFACT.is_file()
    assert artifact["milestone"] == "M1"
    assert artifact["target_version"] == "0.5.0"
    assert artifact["schema_version"] == m1.SCHEMA_VERSION


def test_required_parity_criteria_hold(artifact):
    criteria = artifact["criteria"]
    assert criteria["highfreq_surface_parity"]["met"] is True
    assert criteria["highfreq_dpqr_parity_by_evidence"]["met"] is True
    assert criteria["no_placeholder_in_scope"]["met"] is True


def test_every_promoted_dpqr_symbol_has_parity_evidence(artifact):
    counts = artifact["counts"]
    assert counts["dpqr_other"] == 0
    assert counts["dpqr_golden"] + counts["dpqr_r_consistency"] == counts["dpqr_symbols"]
    assert counts["placeholder_in_scope"] == 0


def test_growth_criterion_equals_the_literal_ten_x_test(artifact):
    counts = artifact["counts"]
    criterion = artifact["criteria"]["golden_validated_order_of_magnitude"]
    assert criterion["met"] == (
        counts["growth_ratio"] >= m1.TARGET_ORDER_OF_MAGNITUDE
    )


def test_kernel_criterion_equals_full_implemented_admission(artifact):
    counts = artifact["counts"]
    criterion = artifact["criteria"]["kernel_implemented"]
    assert criterion["met"] == (
        counts["kernel_implemented_families"] == counts["promoted_families"]
    )


def test_core_verdict_is_the_conjunction_of_the_three_parity_criteria(artifact):
    criteria = artifact["criteria"]
    core = (
        criteria["highfreq_surface_parity"]["met"]
        and criteria["highfreq_dpqr_parity_by_evidence"]["met"]
        and criteria["no_placeholder_in_scope"]["met"]
    )
    assert artifact["verdict"]["core_m1_parity_met"] is core
    assert artifact["verdict"]["literal_all_criteria_met"] is all(
        item["met"] for item in criteria.values()
    )


def test_artifact_stays_in_sync_with_current_artifacts():
    ok, problems = m1.check_artifact(ARTIFACT)
    assert ok, problems


def test_check_cli_exit_code_is_zero():
    assert m1.main(["--check"]) == 0
