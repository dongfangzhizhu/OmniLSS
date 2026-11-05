"""Regression tests for the D0.1 per-symbol R <-> Python coverage matrix tool."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_symbol_coverage import (  # noqa: E402
    OUT_JSON,
    OUT_MD,
    _file_sha256,
    _norm,
    build_coverage,
    dump_json,
    index_r_definitions,
    parse_namespace,
    r_package_dirs,
    render_markdown,
)

SCRIPT = ROOT / "tools" / "generate_symbol_coverage.py"


@pytest.fixture(scope="module")
def coverage() -> dict:
    """One shared matrix build; a full build is ~10s, so never rebuild per test."""
    return build_coverage(emit=lambda *args, **kwargs: None)


def _row(coverage: dict, package: str, symbol: str) -> dict:
    matches = [
        row for row in coverage["symbols"] if row["r_package"] == package and row["r_symbol"] == symbol
    ]
    assert matches, f"{package}::{symbol} missing from the matrix"
    return matches[0]


def test_every_exported_r_symbol_appears_exactly_once(coverage):
    expected = {
        (package, symbol)
        for package, directory in r_package_dirs().items()
        for symbol in parse_namespace(directory / "NAMESPACE")["exports"]
    }
    actual = [(row["r_package"], row["r_symbol"]) for row in coverage["symbols"]]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected


def test_scope_pins_twelve_packages_and_excludes_gamlss2(coverage):
    packages = sorted(r_package_dirs())
    assert len(packages) == 12
    assert coverage["scope"]["r_packages"] == packages
    assert coverage["scope"]["out_of_scope_packages"] == ["gamlss2"]
    assert coverage["stats"]["per_package"]["gamlss2"]["in_scope"] is False


def test_parse_namespace_handles_patterns_comments_and_s3(tmp_path):
    namespace = tmp_path / "NAMESPACE"
    namespace.write_text(
        "# export(commented.out) must be ignored\n"
        "export(gamlss, centiles.com)\n"
        'export("dNO")\n'
        "exportMethods(show)\n"
        'exportPattern("^draw\\\\.")\n'
        "S3method(print, gamlss)\n"
        "import(stats)\n",
        encoding="utf-8",
    )
    parsed = parse_namespace(namespace)
    assert parsed["exports"] == ["centiles.com", "dNO", "gamlss"]
    assert parsed["auxiliary_exports"] == ["show"]
    assert parsed["s3method_count"] == 1


def test_definition_index_reports_source_and_line(tmp_path):
    source = tmp_path / "R" / "foo.r"
    source.parent.mkdir(parents=True)
    source.write_text(
        "foo <- function(x) x\nbar <<- function() NULL\nnot_a_function = 1\n",
        encoding="utf-8",
    )
    definitions = index_r_definitions(tmp_path)
    assert definitions["foo"] == [f"{source.as_posix()}:1"]
    assert definitions["bar"] == [f"{source.as_posix()}:2"]
    assert "not_a_function" not in definitions


def test_file_hash_ignores_line_endings(tmp_path):
    cache: dict[str, str] = {}
    lf = tmp_path / "lf.r"
    crlf = tmp_path / "crlf.r"
    lf.write_text("a\nb\n", encoding="utf-8", newline="\n")
    crlf.write_text("a\nb\n", encoding="utf-8", newline="\r\n")
    other = tmp_path / "other.r"
    other.write_text("a\nc\n", encoding="utf-8", newline="\n")
    assert _file_sha256(lf, cache) == _file_sha256(crlf, cache)
    assert _file_sha256(lf, cache) != _file_sha256(other, cache)


def test_r_dot_names_map_to_python_underscores(coverage):
    assert _norm("centiles.com") == "centiles_com"
    row = _row(coverage, "gamlss", "centiles.com")
    assert row["python_symbol"] == "centiles_com"
    assert row["layers"]["surface"] != "missing"


def test_dpqr_family_row_carries_registry_and_golden_evidence(coverage):
    row = _row(coverage, "gamlss.dist", "dNO")
    assert row["r_family"] == "NO"
    assert row["r_family_kind"] == "dpqr"
    assert row["r_kind"] == "function"
    assert row["layers"]["kernel"] == "implemented"
    assert row["layers"]["admission"] == "production_safe"
    assert row["layers"]["parity"].startswith(("golden:", "r_consistency_test"))
    assert row["evidence"]["golden_cases"], "golden manifest evidence must reach the row"
    assert row["evidence"]["r_definitions"]
    assert row["r_source_sha256"]
    assert row["r_package_version"] != "unknown"
    assert row["python_source_commit"]


def test_symbols_without_python_surface_are_not_admitted(coverage):
    # R's dotted internals (.binom, .gamlss.bi.list, .hat.WX, ...) were ported
    # in D2.180, so use a genuinely unported export as the example now.
    unported = _row(coverage, "gamlss", "gamlssNews")
    assert unported["python_symbol"] is None
    assert unported["layers"] == {
        "surface": "missing",
        "kernel": "missing",
        "parity": "unverified",
        "admission": "not_available",
    }
    missing = [row for row in coverage["symbols"] if row["layers"]["surface"] == "missing"]
    assert missing, "the matrix must report the unported R symbols"
    for row in missing:
        assert row["python_symbol"] is None
        assert row["python_modules"] == []
        assert row["layers"]["admission"] == "not_available"


def test_python_only_public_exports_have_no_r_counterpart(coverage):
    python_only = coverage["python_only_public_exports"]
    assert coverage["stats"]["python_only_public_exports"] == len(python_only)
    assert python_only == sorted(python_only)
    assert "BetaFamily" in python_only
    r_symbols = {row["r_symbol"] for row in coverage["symbols"]}
    r_symbols |= {_norm(symbol) for symbol in r_symbols}
    assert not (set(python_only) & r_symbols)


def test_layer_counters_are_self_consistent(coverage):
    stats = coverage["stats"]
    total = stats["r_exports_total"]
    assert total == len(coverage["symbols"])
    for layer, counts in stats["per_layer"].items():
        assert layer in {"surface", "kernel", "parity", "admission"}
        assert sum(counts.values()) == total
    assert stats["surface_matched"] + stats["surface_missing"] == total
    assert stats["per_layer"]["surface"]["missing"] == stats["surface_missing"]
    assert sum(entry["exported"] for entry in stats["per_package"].values()) == total
    assert stats["in_scope_exports"] + sum(
        entry["exported"] for entry in stats["per_package"].values() if not entry["in_scope"]
    ) == total
    assert stats["family_linked"] <= total
    assert stats["dpqr_linked"] <= stats["family_linked"]


def test_json_artifact_keeps_one_row_per_symbol(coverage):
    text = dump_json(coverage)
    assert json.loads(text) == coverage
    row_lines = [line for line in text.splitlines() if line.startswith('    {"r_symbol"')]
    assert len(row_lines) == len(coverage["symbols"])
    assert text.endswith("}\n")


def test_checked_in_artifacts_match_a_fresh_render(coverage):
    assert OUT_JSON.read_text(encoding="utf-8") == dump_json(coverage)
    assert OUT_MD.read_text(encoding="utf-8") == render_markdown(coverage)
    assert coverage["provenance"]["r_namespace_sha256"]
    assert coverage["provenance"]["golden_manifest_sha256"] != "absent"


def test_cli_help_writes_nothing(tmp_path):
    output = tmp_path / "matrix.json"
    markdown = tmp_path / "matrix.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--markdown", str(markdown), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert "--check" in result.stdout
    assert not output.exists()
    assert not markdown.exists()


@pytest.mark.slow
def test_cli_check_mode_confirms_artifacts_are_current():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "up to date" in result.stdout
