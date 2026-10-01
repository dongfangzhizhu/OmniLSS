"""Tests for the D0.4 coverage difference report and its ratchet baseline.

Most tests use synthetic matrices so the classification, ratchet and rendering
logic is verified instantly; one ``slow`` test re-runs the real D0.1 matrix and
asserts the checked-in baseline/report are fresh.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.coverage_diff_report import (  # noqa: E402
    BASELINE_PATH,
    CATEGORY_MEANING,
    INFORMATIONAL_CATEGORIES,
    OUT_JSON,
    RATCHET_CATEGORIES,
    REPO_ROOT,
    _display_path,
    build_baseline,
    build_report,
    categorise,
    diff_categories,
    load_baseline,
    main,
    render_report,
    write_baseline,
)

SCRIPT = ROOT / "tools" / "coverage_diff_report.py"


def _row(package: str, symbol: str, *, surface: str, kernel: str, parity: str, admission: str) -> dict:
    return {
        "r_package": package,
        "r_symbol": symbol,
        "layers": {
            "surface": surface,
            "kernel": kernel,
            "parity": parity,
            "admission": admission,
        },
    }


def _matrix(rows: list[dict], python_only: list[str] | None = None) -> dict:
    return {
        "symbols": rows,
        "python_only_public_exports": python_only or [],
        "stats": {"r_exports_total": len(rows)},
    }


#: small matrix covering every category exactly once
ROWS = [
    _row("pkg", "ported", surface="public_api", kernel="implemented", parity="r_consistency_test", admission="production_safe"),
    _row("pkg", "no_tests", surface="module_export", kernel="implemented", parity="unverified", admission="internal"),
    _row("pkg", "ph", surface="registry", kernel="placeholder", parity="capability:unsupported", admission="blocked"),
    _row("pkg", "alias", surface="registry", kernel="compat_alias", parity="capability:experimental", admission="blocked"),
    _row("pkg", "absent", surface="missing", kernel="missing", parity="unverified", admission="not_available"),
]


def test_categorise_maps_matrix_layers_to_the_four_defect_classes() -> None:
    categorised = categorise(ROWS, ["BetaFamily"])
    assert categorised["missing"] == ["pkg::absent"]
    assert categorised["placeholder"] == ["pkg::ph"]
    assert categorised["compat_alias"] == ["pkg::alias"]
    assert categorised["evidence_gap"] == ["pkg::no_tests"]
    assert categorised["extra"] == ["BetaFamily"]
    # every category carries an explanation for the report reader
    assert set(CATEGORY_MEANING) == set(RATCHET_CATEGORIES) | set(INFORMATIONAL_CATEGORIES)


def test_missing_symbols_are_not_double_counted_as_evidence_gaps() -> None:
    categorised = categorise([_row("pkg", "absent", surface="missing", kernel="missing", parity="unverified", admission="not_available")], [])
    assert categorised["missing"] == ["pkg::absent"]
    assert categorised["evidence_gap"] == []


def test_categories_are_sorted_and_deduplicated() -> None:
    rows = [
        _row("b", "x", surface="missing", kernel="missing", parity="unverified", admission="not_available"),
        _row("a", "y", surface="missing", kernel="missing", parity="unverified", admission="not_available"),
        _row("a", "y", surface="missing", kernel="missing", parity="unverified", admission="not_available"),
    ]
    assert categorise(rows, ["zz", "aa"])["missing"] == ["a::y", "b::x"]
    assert categorise(rows, ["zz", "aa"])["extra"] == ["aa", "zz"]


def test_report_paths_are_repo_relative_and_machine_independent(tmp_path: Path) -> None:
    """Committed artifacts must never embed an absolute, machine-specific path."""
    assert _display_path(REPO_ROOT / "docs" / "development" / "x.json") == "docs/development/x.json"
    outside = tmp_path / "x.json"
    assert _display_path(outside) == str(outside.resolve())
    report = build_report(baseline_path=BASELINE_PATH, coverage=_matrix([]), emit=lambda *a, **k: None)
    assert report["baseline"] == "docs/development/symbol-coverage-baseline.json"
    assert "\\" not in report["baseline"]


def test_committed_diff_report_snapshot_is_portable() -> None:
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    assert payload["baseline"] == "docs/development/symbol-coverage-baseline.json"
    rendered = json.dumps(payload, ensure_ascii=False)
    for drive_root in ("D:\\\\AI", "C:\\\\", "/home/runner"):
        assert drive_root not in rendered


def test_baseline_payload_is_deterministic() -> None:
    current = categorise(ROWS, ["BetaFamily"])
    payload = build_baseline(current)
    assert payload["schema_version"] == 1
    assert payload["counts"] == {key: len(values) for key, values in current.items()}
    assert payload["symbols"]["missing"] == ["pkg::absent"]
    assert payload["symbols"] == {key: sorted(values) for key, values in current.items()}


def test_baseline_roundtrip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(build_baseline(categorise(ROWS, [])), path)
    baseline, error = load_baseline(path)
    assert error == "" and baseline is not None

    path.write_text('{"schema_version": 2, "symbols": {}}', encoding="utf-8")
    _, error = load_baseline(path)
    assert "schema_version" in error

    path.write_text('{"schema_version": 1, "symbols": {"missing": []}}', encoding="utf-8")
    _, error = load_baseline(path)
    assert "缺类别" in error

    path.write_text("not json", encoding="utf-8")
    _, error = load_baseline(path)
    assert "不可读" in error

    missing_path = tmp_path / "nope.json"
    _, error = load_baseline(missing_path)
    assert "不存在" in error


def test_diff_reports_new_and_resolved_symbols() -> None:
    baseline = build_baseline(categorise(ROWS, []))
    grown = categorise(
        ROWS
        + [
            _row("pkg", "new_ph", surface="registry", kernel="placeholder", parity="capability:unsupported", admission="blocked"),
            _row("pkg", "new_no_tests", surface="module_def", kernel="implemented", parity="unverified", admission="internal"),
        ],
        [],
    )
    diff = diff_categories(grown, baseline)
    assert diff["placeholder"]["new"] == ["pkg::new_ph"]
    assert diff["evidence_gap"]["new"] == ["pkg::new_no_tests"]
    assert diff["placeholder"]["current_count"] == 2
    assert diff["placeholder"]["baseline_count"] == 1

    shrunken = categorise([ROWS[0]], [])
    diff = diff_categories(shrunken, baseline)
    assert diff["missing"]["resolved"] == ["pkg::absent"]
    assert diff["missing"]["new"] == []
    assert diff["placeholder"]["resolved"] == ["pkg::ph"]


def test_report_fails_closed_on_a_new_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(build_baseline(categorise(ROWS, [])), path)
    grown = _matrix(
        ROWS
        + [
            _row("pkg", "new_ph", surface="registry", kernel="placeholder", parity="capability:unsupported", admission="blocked")
        ]
    )
    report = build_report(baseline_path=path, coverage=grown, emit=lambda *a, **k: None)
    assert report["ok"] is False
    assert report["ratchet_ok"] is False
    assert report["ceiling_ok"] is False
    assert report["regressions"] == [{"category": "placeholder", "symbol": "pkg::new_ph"}]
    rendered = render_report(report)
    assert "REGRESSIONS: 1" in rendered
    assert "[placeholder] pkg::new_ph" in rendered
    assert "FAIL-CLOSED" in rendered


def test_report_passes_and_lists_improvements(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(build_baseline(categorise(ROWS, [])), path)
    improved = _matrix([ROWS[0]])
    report = build_report(baseline_path=path, coverage=improved, emit=lambda *a, **k: None)
    assert report["ok"] is True
    assert report["regressions"] == []
    assert {entry["category"] for entry in report["improvements"]} == {
        "missing",
        "placeholder",
        "compat_alias",
        "evidence_gap",
    }
    assert "improvements: 4" in render_report(report)


def test_report_is_stable_when_nothing_changes(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(build_baseline(categorise(ROWS, ["BetaFamily"])), path)
    report = build_report(baseline_path=path, coverage=_matrix(ROWS, ["BetaFamily"]), emit=lambda *a, **k: None)
    assert report["ok"] is True
    assert report["regressions"] == [] and report["improvements"] == []
    assert "no symbol-level change" in render_report(report)


def test_report_fails_closed_when_baseline_is_missing(tmp_path: Path) -> None:
    report = build_report(baseline_path=tmp_path / "absent.json", coverage=_matrix(ROWS), emit=lambda *a, **k: None)
    assert report["ok"] is False
    assert "不存在" in report["failure"]
    rendered = render_report(report)
    assert "FAIL-CLOSED" in rendered and "--update-baseline" in rendered


def test_extra_category_never_fails_the_gate(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(build_baseline(categorise(ROWS, [])), path)
    report = build_report(
        baseline_path=path,
        coverage=_matrix(ROWS, ["BrandNewClass"]),
        emit=lambda *a, **k: None,
    )
    assert report["ok"] is True
    assert report["categories"]["extra"]["new"] == ["BrandNewClass"]
    assert report["regressions"] == []  # reported, not fatal


def test_cli_update_baseline_and_report_with_synthetic_coverage(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    out = tmp_path / "diff.json"
    assert main(["--update-baseline", "--baseline", str(baseline), "--out", str(out)], coverage=_matrix(ROWS, [])) == 0
    assert "baseline updated" in capsys.readouterr().out
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["counts"]["missing"] == 1

    assert main(["--baseline", str(baseline), "--out", str(out)], coverage=_matrix(ROWS, [])) == 0
    assert "result: OK" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True

    grown = _matrix(ROWS + [ROWS[2]])  # an extra placeholder symbol id is duplicated -> still 1
    extra_placeholder = _matrix(
        ROWS
        + [
            _row("pkg", "another_ph", surface="registry", kernel="placeholder", parity="capability:unsupported", admission="blocked")
        ]
    )
    assert main(["--baseline", str(baseline), "--out", str(out)], coverage=extra_placeholder) == 1
    assert "FAIL-CLOSED" in capsys.readouterr().out
    assert main(["--baseline", str(baseline), "--out", str(out)], coverage=grown) == 0  # duplicate id, no new symbol
    capsys.readouterr()

    assert (
        main(["--baseline", str(baseline), "--out", str(out), "--allow-regressions"], coverage=extra_placeholder) == 0
    )
    assert "FAIL-CLOSED" in capsys.readouterr().out


def test_cli_json_mode_is_machine_readable(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    write_baseline(build_baseline(categorise(ROWS, [])), baseline)
    assert main(["--json", "--baseline", str(baseline), "--out", str(tmp_path / "d.json")], coverage=_matrix(ROWS, [])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert set(payload["categories"]) == set(RATCHET_CATEGORIES) | set(INFORMATIONAL_CATEGORIES)


def test_cli_help_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "diff.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert "--update-baseline" in result.stdout
    assert not out.exists()


@pytest.mark.slow
def test_checked_in_baseline_and_report_are_fresh() -> None:
    """The real matrix must match the committed ratchet and report snapshot."""
    from tools.generate_symbol_coverage import build_coverage

    coverage = build_coverage(emit=lambda *a, **k: None)
    report = build_report(baseline_path=BASELINE_PATH, coverage=coverage, emit=lambda *a, **k: None)
    assert report["ok"], render_report(report)
    assert report["regressions"] == []
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["counts"] == report["current_counts"]
    assert OUT_JSON.read_text(encoding="utf-8") == json.dumps(report, indent=2, ensure_ascii=False) + "\n"