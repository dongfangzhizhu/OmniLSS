# SPDX-License-Identifier: GPL-3.0-or-later
"""M0 milestone manifest: the fixed release evidence bundle (D2.226).

DEVELOPMENT_PLAN section 22.4 requires every milestone to carry a fixed
manifest with R/Python/JAX/NumPy/SciPy/x64 versions, source hash, seeds,
tolerances and commands, plus --require-r, no hidden skips, the coverage diff
and a reproducible commit.

These tests pin the committed M0 artifact (docs/development/milestone-manifest-m0.json)
to the current tree via tools/milestone_manifest.py:
- the artifact has the required schema and every gate passes;
- the environment is pinned to manifest_template.json;
- it stays in sync with the coverage artifacts and the golden seed set (so a
  symbol/test change forces a regeneration instead of silently going stale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import milestone_manifest as mm  # noqa: E402

ARTIFACT = mm.DEFAULT_OUT


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_exists_and_identifies_m0():
    assert ARTIFACT.is_file()
    m = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert m["milestone"] == "M0"
    assert m["version"].startswith("0.4")
    assert m["schema_version"] == mm.SCHEMA_VERSION


def test_required_manifest_fields_present(manifest):
    for key in (
        "environment",
        "r_environment",
        "pinned_expected",
        "environment_pinned_ok",
        "seeds",
        "tolerances",
        "coverage",
        "gates",
        "source",
    ):
        assert key in manifest, key


def test_environment_is_pinned(manifest):
    assert manifest["environment_pinned_ok"] is True, manifest.get("environment_mismatches")
    template = json.loads(mm.TEMPLATE.read_text(encoding="utf-8"))
    assert manifest["pinned_expected"] == template["pinned"]
    assert manifest["tolerances"] == template["tolerances"]
    env = manifest["environment"]
    assert env["jax_enable_x64"] is True
    assert env["python"].startswith(template["pinned"]["python_version_prefix"])


def test_every_m0_gate_passes(manifest):
    for key in mm.GATE_KEYS:
        gate = manifest["gates"].get(key)
        assert gate is not None, key
        assert gate["ok"] is True, (key, gate)


def test_artifact_stays_in_sync_with_current_tree(manifest):
    ok, problems = mm.check_manifest(ARTIFACT)
    assert ok, problems


def test_check_cli_exit_code_is_zero():
    assert mm.main(["--check"]) == 0


def test_ci_and_release_gates_run_the_manifest_check():
    """M0 evidence must be gated in CI and at release, not just present."""
    for workflow in ("ci.yml", "release.yml"):
        text = (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "tools/milestone_manifest.py --check" in text, workflow
