"""A family must get the same sample however the scan is invoked (D2.83).

The gate builds every family grid and fit sample from ONE shared RNG stream
in discovery order.  While the loop only walked the families it was asked
for, a subset run handed each family a different sample than a full run did,
and the same code produced different verdicts depending on the invocation:
ZAP measured 206.212 on a subset against 201.76 in the full run whose numbers
had been registered as evidence.

The fix is that the loop always walks every family (drawing exactly what a
full run draws) and the request only selects which jobs are emitted.  These
tests pin the property without needing R.  They build the full job set once
and the subset once -- building it per family would be O(families^2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

_PROBE = ["ZAP", "GAF", "NO"]


@pytest.fixture(scope="module")
def payloads():
    full_jobs, full_index = scan._build_jobs(scan.discover_families(), 8)
    sub_jobs, sub_index = scan._build_jobs(_PROBE, 8)
    return (full_jobs, full_index), (sub_jobs, sub_index)


def test_a_subset_run_gets_the_full_run_sample(payloads):
    (full_jobs, full_index), (sub_jobs, sub_index) = payloads
    assert set(sub_index) == set(_PROBE)
    for family in _PROBE:
        expected = full_index[family]
        actual = sub_index[family]
        assert set(expected) == set(actual), family
        assert (
            full_jobs[expected["fit"]]["data"] == sub_jobs[actual["fit"]]["data"]
        ), family
        for type_char in ("d", "p", "q"):
            if type_char not in expected:
                continue
            full_job = full_jobs[expected[type_char]]
            sub_job = sub_jobs[actual[type_char]]
            assert full_job["args"] == sub_job["args"], (family, type_char)
            assert full_job["argnames"] == sub_job["argnames"], (family, type_char)


def test_the_full_index_covers_every_family(payloads):
    (_, full_index), _ = payloads
    assert set(full_index) == set(scan.discover_families())