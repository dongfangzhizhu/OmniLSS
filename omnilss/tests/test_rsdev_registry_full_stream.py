# SPDX-License-Identifier: GPL-3.0-or-later
"""P0-5: rsdev registry numbers bind to the full family stream.

Discipline (``.workbuddy/memory``):

- D2.18 / D2.83: a registry number is only valid on the **full family
  stream**.  A scanner run on a sublist used to hand every family a different
  sample (shared RNG advanced in discovery order), producing false verdicts
  (ZAP 206.212 on a subset vs 201.76 in the full run that registered it).
  D2.83 fixed it: ``_build_jobs`` always walks every family and the request
  only selects which jobs are emitted.  ``test_scan_sampling_is_order_stable``
  pins the property with three fixed ``_PROBE`` families.
- D2.47 addendum (2026-10-20): the worry that the Bessel JVP fix invalidated
  ZIPIG/ZINBI numbers was **disproved** -- PIG/ZIPIG use the independent
  ``omnilss/pig.py`` kernel, never Bessel; ZIPIG/ZINBI/PE2 numbers matched
  their registrations.  D2.103 then ran the full 146-family stream to 0
  rsdev failures in one pass, and 2026-11-01 confirmed field-for-field parity.

This file extends the "subset == full stream" invariant from the three fixed
``_PROBE`` families to **every family carrying a registry number**, and pins
that each registered family has a real full-stream fit payload for its number
to bind to, plus basic number self-consistency.  That gives the P0-5
acceptance criteria ("登记表数字与全族流一致；子集运行不再产生假警报")
automated regression cover without re-running the multi-hour live-R scan.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

_TABLES = (
    "_RSDEV_R_DIVERGED",
    "_RSDEV_R_STOPPED_EARLY",
    "_RSDEV_PATH_DIVERGENT",
    "_RSDEV_R_DENSITY_RECYCLED",
)


def _registered() -> list[str]:
    return sorted({f for t in _TABLES for f in getattr(scan, t)})


@pytest.fixture(scope="module")
def streams():
    families = scan.discover_families()
    full_jobs, full_index = scan._build_jobs(families, 8)
    registered = _registered()
    sub_jobs, sub_index = scan._build_jobs(registered, 8)
    return (full_jobs, full_index), (sub_jobs, sub_index), registered


def test_every_registered_family_has_a_full_stream_fit_payload(streams):
    """A registered number is meaningless if its family has no stream sample."""
    (_, full_index), _, registered = streams
    for family in registered:
        assert family in full_index, f"{family} missing from full stream"
        assert "fit" in full_index[family], (
            f"{family} has no full-stream fit payload to bind its number to"
        )


def test_registered_families_get_full_stream_sample_in_a_subset_run(streams):
    """The D2.83 invariant, now covering all registered families, not 3 probes."""
    (full_jobs, full_index), (sub_jobs, sub_index), registered = streams
    for family in registered:
        expected = full_index[family]
        actual = sub_index[family]
        assert set(expected) == set(actual), family
        assert full_jobs[expected["fit"]]["data"] == sub_jobs[actual["fit"]]["data"], family
        for type_char in ("d", "p", "q"):
            if type_char not in expected:
                continue
            full_job = full_jobs[expected[type_char]]
            sub_job = sub_jobs[actual[type_char]]
            assert full_job["args"] == sub_job["args"], (family, type_char)
            assert full_job["argnames"] == sub_job["argnames"], (family, type_char)


def test_registry_numbers_are_well_formed():
    """Every registration carries a finite dev_ours and a rationale note."""
    for table in _TABLES:
        for family, record in getattr(scan, table).items():
            dev_ours = record["dev_ours"]
            assert isinstance(dev_ours, (int, float)) and math.isfinite(dev_ours), (
                table, family
            )
            assert isinstance(record["note"], str) and record["note"].strip(), (
                table, family
            )


def test_r_stopped_early_numbers_encode_that_r_halted_higher():
    """'R stopped early' means our optimum is *below* R's halt, at a set budget."""
    for family, record in scan._RSDEV_R_STOPPED_EARLY.items():
        assert record["dev_ours"] < record["dev_r"], family
        assert isinstance(record["n_cyc"], int) and record["n_cyc"] > 0, family
        assert math.isfinite(record["dev_r"]), family
