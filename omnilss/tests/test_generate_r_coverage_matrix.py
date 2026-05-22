"""Tests for tools/generate_r_coverage_matrix.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_r_coverage_matrix.py"
_SPEC = importlib.util.spec_from_file_location("generate_r_coverage_matrix", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_parse_namespace_exports_multiline(tmp_path: Path) -> None:
    namespace = tmp_path / "NAMESPACE"
    namespace.write_text(
        'export(foo, bar)\nexport(\n  baz,\n  qux\n)\n', encoding="utf-8"
    )

    got = mod._parse_namespace_exports(namespace)

    assert got == {"foo", "bar", "baz", "qux"}


def test_discover_namespace_prefers_existing_candidate(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    root = repo_root / "omnilss"
    candidate = repo_root / "omnilss" / "R" / "gamlss" / "NAMESPACE"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("export(NO)\n", encoding="utf-8")

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "ROOT", root)

    path = mod._discover_namespace_path("R", "gamlss", "NAMESPACE")

    assert path == candidate


def test_build_matrix_with_local_fixture_layout(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    root = repo_root / "omnilss"
    tests_root = root / "tests"
    docs_root = repo_root / "docs" / "development"
    gamlss_ns = repo_root / "omnilss" / "R" / "gamlss" / "NAMESPACE"
    dist_ns = repo_root / "omnilss" / "R" / "gamlss.dist" / "NAMESPACE"

    tests_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    gamlss_ns.parent.mkdir(parents=True, exist_ok=True)
    dist_ns.parent.mkdir(parents=True, exist_ok=True)

    gamlss_ns.write_text("export(gamlss, NO)\n", encoding="utf-8")
    dist_ns.write_text("export(PO, fooDist)\n", encoding="utf-8")
    (tests_root / "test_r_consistency_custom.py").write_text(
        "family = 'NO'\ndist_name='PO'\n", encoding="utf-8"
    )
    (docs_root / "family-capability-matrix-2026-05-18.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"family": "NO", "evidence_tier": "validated", "r_consistency": "validated"},
                    {"family": "PO", "evidence_tier": "experimental", "r_consistency": "validated"},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "ROOT", root)

    payload = mod.build_matrix()

    assert payload["summary"]["r_export_count"] == 4
    by_symbol = {row["r_symbol"]: row for row in payload["rows"]}
    assert by_symbol["NO"]["implemented"] is True
    assert by_symbol["NO"]["r_consistency_tested"] is True
    assert by_symbol["NO"]["capability_tier"] == "validated"
    assert by_symbol["PO"]["implemented"] is True
    assert by_symbol["PO"]["r_consistency_tested"] is True
    assert by_symbol["fooDist"]["implemented"] is False
