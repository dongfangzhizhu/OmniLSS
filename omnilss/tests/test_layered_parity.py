# SPDX-License-Identifier: GPL-3.0-or-later
"""RS / CG / MIXED layered parity (D2.238, M2 gate).

Pins the committed matrix to the committed R reference and pins the
classification *logic*: flat_ridge cells must respect the documented bounds, the
verdict must be the conjunction of the problem lists, and every cell must carry
a code.  It also pins that our side has no mismatch and no failing route.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import layered_parity as lp  # noqa: E402

ARTIFACT = lp.DEFAULT_OUT
CODES = {"pass", "flat_ridge", "r_error", "r_not_converged", "mismatch", "py_error"}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_scope_is_families_times_methods(artifact):
    scope = artifact["scope"]
    assert scope["cells"] == len(scope["families"]) * len(scope["methods"])
    assert tuple(scope["methods"]) == lp.METHODS
    assert len(scope["families"]) > 0


def test_every_cell_has_a_known_code(artifact):
    assert len(artifact["cells"]) == artifact["scope"]["cells"]
    for cell in artifact["cells"]:
        assert cell["code"] in CODES, cell
    assert sum(artifact["counts"]["by_code"].values()) == artifact["scope"]["cells"]


def test_our_side_has_no_mismatch_and_no_failing_route(artifact):
    assert artifact["problems"]["mismatch"] == []
    assert artifact["problems"]["py_error"] == []


def test_flat_ridge_cells_respect_the_documented_bounds(artifact):
    tolerances = artifact["tolerances"]
    for cell in artifact["cells"]:
        if cell["code"] != "flat_ridge":
            continue
        assert cell["dev_rel"] <= tolerances["flat_ridge_dev_rtol"], cell
        assert cell["max_abs_coef"] <= tolerances["flat_ridge_max_coef"], cell


def test_verdict_is_the_conjunction_of_the_problem_lists(artifact):
    problems = artifact["problems"]
    assert artifact["verdict"]["our_side_parity_ok"] is not (
        problems["mismatch"] or problems["py_error"]
    )


def test_artifact_stays_in_sync_with_the_r_reference_and_tree():
    ok, problems = lp.check_artifact(ARTIFACT)
    assert ok, problems
