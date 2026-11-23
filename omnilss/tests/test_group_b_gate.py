# SPDX-License-Identifier: GPL-3.0-or-later
"""Group B skip baseline and fail-closed gate (D2.308/R4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import group_b_gate as gate  # noqa: E402

BASELINE = REPO / "docs" / "development" / "group-b-skip-baseline.json"


def _log(passed: int = 512, skipped: int = 29, reasons: tuple[tuple[int, str], ...] = ()) -> str:
    lines = [f"{passed} passed, {skipped} skipped, 6 xfailed in 600.00s"]
    for count, reason in reasons:
        lines.append(f"SKIPPED [{count}] tests/test_demo.py:10: {reason}")
    return "\n".join(lines)


def test_baseline_records_registered_skips() -> None:
    data = gate.load_baseline(BASELINE)
    assert data["baseline_skipped"] == 29
    assert data["allowed_growth"] == 5
    assert data["min_passed"] == 300
    assert "vendored gamlss." in data["allowed_reason_prefixes"]


def test_gate_accepts_baseline() -> None:
    baseline = gate.load_baseline(BASELINE)
    _, _, problems = gate.evaluate(
        _log(reasons=((24, "vendored gamlss.tr source not present"), (5, "R model did not converge for GB2"))),
        baseline,
    )
    assert problems == []


def test_gate_rejects_skip_growth() -> None:
    baseline = gate.load_baseline(BASELINE)
    _, _, problems = gate.evaluate(_log(skipped=35), baseline)
    assert any("exceeds baseline" in p for p in problems)


def test_gate_rejects_new_reason() -> None:
    baseline = gate.load_baseline(BASELINE)
    _, _, problems = gate.evaluate(
        _log(reasons=((1, "something silently skipped"),)), baseline
    )
    assert any("unknown skip reason" in p for p in problems)


def test_gate_rejects_too_few_assertions() -> None:
    baseline = gate.load_baseline(BASELINE)
    _, _, problems = gate.evaluate(_log(passed=299), baseline)
    assert any("required floor" in p for p in problems)


def test_gate_rejects_unusable_r() -> None:
    baseline = gate.load_baseline(BASELINE)
    _, _, problems = gate.evaluate(_log() + "\nRscript: command not found\n", baseline)
    assert any("R is not usable" in p for p in problems)


def test_gate_requires_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"baseline_skipped": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        gate.load_baseline(bad)
