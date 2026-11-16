# SPDX-License-Identifier: GPL-3.0-or-later
"""M2 assessment snapshot (D2.239).

Pins the committed M2 artifact to the committed source artifacts and pins the
verdict *logic*: m2_gate_met must be the conjunction of the gate criteria, the
live counts must match the per-row payloads, and the failures_explainable
criterion must be a real check over the layered cell vocabulary rather than a
vacuously true expression.

The live rebuild (22 fits + JSON round-trip) is done once per module; it is the
same work the tool's --check performs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import m2_assessment as m2  # noqa: E402

ARTIFACT = m2.DEFAULT_OUT


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current(artifact) -> dict:
    return m2.build(live=bool(artifact["live_replay"]))


def test_artifact_exists_and_identifies_m2(artifact):
    assert ARTIFACT.is_file()
    assert artifact["milestone"] == "M2"
    assert artifact["target_version"] == "0.6.0"
    assert artifact["schema_version"] == m2.SCHEMA_VERSION


def test_promoted_families_match_the_frozen_list(artifact):
    frozen = json.loads((REPO / "docs/development/highfreq-families-frozen.json").read_text(encoding="utf-8"))
    promoted = sorted(item["family"] for item in frozen["tiers"]["promoted"])
    assert artifact["promoted_families"] == promoted
    assert len(artifact["rows"]) == len(promoted)


def test_every_row_records_fit_and_newdata_evidence(artifact):
    for row in artifact["rows"]:
        assert row["case"], row
        static = row["static"]
        assert static["has_deviance"], row["family"]
        assert static["estimated_coefficient_blocks"], row["family"]
        assert static["has_newdata"] and static["has_predicted"], row["family"]
        assert static["has_convergence"], row["family"]
        assert static["predicted_parameters"], row["family"]


def test_live_counts_agree_with_the_row_payloads(artifact):
    live_rows = [row for row in artifact["rows"] if row.get("live")]
    assert len(live_rows) == len(artifact["rows"])
    counts = artifact["counts"]
    assert counts["live_predictions_ok"] == sum(
        1 for row in live_rows if row["live"]["predicted_within_tolerance"]
    )
    assert counts["live_roundtrip_ok"] == sum(
        1 for row in live_rows
        if row["live"]["roundtrip_max_abs_drift"] <= m2.ROUNDTRIP_RTOL
    )
    assert counts["live_converged_ok"] == sum(
        1 for row in live_rows if row["live"]["converged"]
    )
    assert counts["live_errors"] == sorted(
        row["family"] for row in artifact["rows"] if row.get("error")
    )


def test_gate_verdict_is_the_conjunction_of_the_criteria(artifact):
    criteria = artifact["criteria"]
    assert artifact["verdict"]["m2_gate_met"] is all(
        criteria[name]["met"] for name in artifact["verdict"]["gate_criteria"]
    )
    assert set(artifact["verdict"]["gate_criteria"]) == set(criteria)


def test_failures_explainable_rejects_unknown_layered_codes(artifact):
    layered = json.loads((REPO / "docs/development/layered-parity-2026-11-15.json").read_text(encoding="utf-8"))
    codes = {cell["code"] for cell in layered["cells"]}
    assert codes <= set(m2.LAYER_CODE_VOCABULARY)
    # The criterion must equal the real check, not a constant.
    expected = (
        bool(layered["verdict"]["our_side_parity_ok"])
        and bool(json.loads((REPO / "docs/development/p2-defect-classification-2026-11-15.json").read_text(encoding="utf-8"))["verdict"]["all_failures_classified"])
        and not (codes - set(m2.LAYER_CODE_VOCABULARY))
    )
    assert artifact["criteria"]["failures_explainable"]["met"] is expected


def test_artifact_stays_in_sync_with_the_current_tree(artifact, current):
    assert m2._stable(artifact) == m2._stable(current)
