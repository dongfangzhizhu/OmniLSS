"""P0-6: every coverage artifact must explicitly mark ``gamlss2`` out of scope.

The deliverable plan (section 22.6 / P0-6) requires that no coverage report can
be read as claiming 1.0 parity for ``gamlss2``: it has a separate spec, version
and evidence chain and does not block 1.0.  A single source of truth
(``tools/coverage_scope.py``) feeds every generator; these tests pin that the
committed artifacts all carry the declaration and that no report drifted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.coverage_scope import (  # noqa: E402
    OUT_OF_SCOPE_PACKAGES,
    scope_line,
    scope_payload,
)

DEV = ROOT / "docs" / "development"

JSON_REPORTS = (
    DEV / "r-symbol-coverage.json",
    DEV / "r-export-coverage.json",
    DEV / "symbol-coverage-diff.json",
    DEV / "symbol-coverage-baseline.json",
)

MARKDOWN_REPORTS = (
    DEV / "r-symbol-coverage.md",
    DEV / "r-export-coverage.md",
    DEV / "placeholder-migration-queue-2026-11-10.md",
)


def test_single_source_lists_gamlss2():
    assert "gamlss2" in OUT_OF_SCOPE_PACKAGES
    assert scope_payload()["out_of_scope_packages"] == sorted(OUT_OF_SCOPE_PACKAGES)
    assert "gamlss2" in scope_line()
    assert "OUT OF SCOPE" in scope_line()


def test_scope_payload_is_self_consistent():
    payload = scope_payload()
    assert payload["in_scope_definition"]
    assert payload["counting_note"]
    assert "gamlss2" in payload["notice"]
    # The counting note must warn that raw counts still include out-of-scope
    # symbols (surface_missing includes them) -- prevents a misread of the report.
    assert "raw" in payload["counting_note"].lower()


@pytest.mark.parametrize("path", JSON_REPORTS, ids=lambda p: p.name)
def test_json_report_has_explicit_scope_block(path):
    if not path.exists():  # pragma: no cover - committed artifacts
        pytest.skip(f"{path.name} not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "scope" in payload, f"{path.name} missing top-level scope block"
    scope = payload["scope"]
    assert set(scope["out_of_scope_packages"]) == set(OUT_OF_SCOPE_PACKAGES)
    assert "gamlss2" in scope["out_of_scope_packages"]


@pytest.mark.parametrize("path", MARKDOWN_REPORTS, ids=lambda p: p.name)
def test_markdown_report_has_explicit_scope_line(path):
    if not path.exists():  # pragma: no cover - committed artifacts
        pytest.skip(f"{path.name} not present")
    text = path.read_text(encoding="utf-8")
    assert "gamlss2" in text, f"{path.name} does not name gamlss2"
    assert "OUT OF SCOPE" in text or "out of scope" in text, (
        f"{path.name} does not explicitly declare out-of-scope"
    )


def test_every_json_report_agrees_with_single_source():
    """No report may list a different out-of-scope set than the source."""
    for path in JSON_REPORTS:
        if not path.exists():
            continue
        scope = json.loads(path.read_text(encoding="utf-8"))["scope"]
        assert scope["out_of_scope_packages"] == sorted(OUT_OF_SCOPE_PACKAGES), path.name


def test_rendered_diff_report_contains_scope_line():
    """The human-readable diff output (not just the JSON) states the scope."""
    from tools.coverage_diff_report import render_report

    rendered = render_report({"baseline": "x.json", "failure": "none"})
    assert "gamlss2" in rendered
    assert "OUT OF SCOPE" in rendered
