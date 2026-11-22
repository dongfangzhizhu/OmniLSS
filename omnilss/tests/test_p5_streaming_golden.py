# SPDX-License-Identifier: GPL-3.0-or-later
"""P5 streaming golden artifact stability (D2.284/C1).

The artifact is generated on Windows and checked on Linux CI; BLAS/JAX
differences produce ~1e-15 jitter in deterministic metrics.  _stable must
ignore that jitter but still catch real numeric changes.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import p5_streaming_golden as p5  # noqa: E402


def _artifact(value: float) -> dict:
    return {
        "schema_version": p5.SCHEMA_VERSION,
        "task": {"name": "t"},
        "counts": {"criteria": 1, "met": 1},
        "verdict": {"streaming_golden_met": True},
        "criteria": {"batch_invariance": {"met": True, "max_coef_diff": value}},
    }


def test_stable_ignores_float_jitter_below_rounding() -> None:
    a = _artifact(1.0e-15)
    b = _artifact(2.0e-15)
    assert p5._stable(a) == p5._stable(b)


def test_stable_detects_real_numeric_change() -> None:
    a = _artifact(1.0e-15)
    b = copy.deepcopy(a)
    b["criteria"]["batch_invariance"]["max_coef_diff"] = 1.0e-6
    assert p5._stable(a) != p5._stable(b)


def test_stable_keeps_verdict_and_counts() -> None:
    a = _artifact(0.0)
    b = copy.deepcopy(a)
    b["verdict"]["streaming_golden_met"] = False
    assert p5._stable(a) != p5._stable(b)
