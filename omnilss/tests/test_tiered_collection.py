"""Tests for the D0.3 tiered collection gate (this project's own test surface)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tiered_collection import (  # noqa: E402
    BENCHMARK_KNOWN_EXCLUSIONS,
    REFERENCE_PROJECTS,
    TIERS,
    Tier,
    build_command,
    build_report,
    evaluate_gate,
    main,
    parse_collection,
    parse_summary,
    render_report,
    run_tier,
)

#: Directories that are neither tiers nor owned test surfaces.
NON_TIER_DIRS = {
    ".venv",
    "debianjaxgpu",
    "tmp",
    "scratch",
    "temp_tests",
    "legacy",
    "results",
    "R",
    "__pycache__",
    *REFERENCE_PROJECTS,
}


def test_four_tiers_cover_this_project_and_exclude_reference_projects() -> None:
    assert set(TIERS) == {"omnilss", "omnilss-server", "omnilss-pro", "benchmarks"}
    assert set(REFERENCE_PROJECTS) == {"ondil", "gamlss-python"}
    assert set(TIERS).isdisjoint(REFERENCE_PROJECTS)
    for name in REFERENCE_PROJECTS:
        assert not any(tier.cwd.startswith(name) for tier in TIERS.values())
    assert all(tier.root.is_dir() for tier in TIERS.values())


def test_every_repository_test_file_belongs_to_a_tier() -> None:
    roots = {tier.root.resolve() for tier in TIERS.values()}
    covered = {p.resolve() for root in roots for p in root.rglob("test_*.py")}
    assert len(covered) > 400  # the four tiers own every test file of this repo
    stray: list[str] = []
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir() or child.name in NON_TIER_DIRS or child.resolve() in roots:
            continue
        stray.extend(p.relative_to(ROOT).as_posix() for p in child.rglob("test_*.py"))
    assert stray == [], f"test files outside the four tiers: {stray}"


def test_tier_roots_contain_at_least_one_test_file() -> None:
    for tier in TIERS.values():
        found = [p for p in tier.root.rglob("test_*.py") if "__pycache__" not in p.parts]
        assert found, f"tier {tier.name} collects nothing"


def test_benchmarks_tier_excludes_only_known_broken_scripts() -> None:
    tier = TIERS["benchmarks"]
    assert tier.ignore == BENCHMARK_KNOWN_EXCLUSIONS
    for relative in BENCHMARK_KNOWN_EXCLUSIONS:
        assert (tier.root / relative).is_file(), f"stale exclusion: {relative}"
        assert len(BENCHMARK_KNOWN_EXCLUSIONS) == 5


def test_omnilss_pro_pyproject_declares_src_pythonpath() -> None:
    text = (ROOT / "omnilss-pro" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'pythonpath = ["src"]' in text


def test_build_command_collect_vs_run() -> None:
    tier = TIERS["benchmarks"]
    collect = build_command(tier, mode="collect", python="py")
    assert "--collect-only" in collect
    assert f"--ignore={BENCHMARK_KNOWN_EXCLUSIONS[0]}" in collect
    assert "--collect-only" not in build_command(tier, mode="run", python="py")
    run = build_command(tier, mode="run", python="py")
    assert run[0] == "py" and run[1:3] == ["-m", "pytest"]
    # the main tier collects only its tests/ argument
    assert build_command(TIERS["omnilss"], mode="collect", python="py")[-1] == "tests"


def test_parse_collection_prefers_summary_line() -> None:
    output = "tests/test_a.py::test_x\n\n25 tests collected in 2.19s\n"
    assert parse_collection(output) == {"collected": 25, "errors": 0, "skipped": 0}


def test_parse_collection_falls_back_to_per_file_counts() -> None:
    # Doubled -q (pytest config addopts + explicit -q) hides the summary line.
    output = (
        "omnilss-server/test_server_api.py: 1\n"
        "benchmarks/diagnostic/test_bi_with_bd.py: 3\n"
        "path/mod.py:988: 11 warnings\n"
    )
    assert parse_collection(output) == {"collected": 4, "errors": 0, "skipped": 0}


def test_parse_collection_counts_collection_errors() -> None:
    output = (
        "1 test collected, 1 error in 0.30s\n"
        "ERROR omnilss-pro/tests/test_automl.py\n"
        "ERROR tests/test_other.py\n"
    )
    assert parse_collection(output) == {"collected": 1, "errors": 2, "skipped": 0}


def test_parse_collection_reports_collection_time_skips() -> None:
    summary = "3319 tests collected / 2 skipped in 40.00s\n"
    assert parse_collection(summary) == {"collected": 3319, "errors": 0, "skipped": 2}
    reasons_only = (
        "SKIPPED [1] tests/test_property_based_core.py:4: could not import 'hypothesis'\n"
        "SKIPPED [1] tests/test_core_families_property.py:34: could not import 'hypothesis'\n"
    )
    assert parse_collection(reasons_only) == {"collected": 0, "errors": 0, "skipped": 2}


def test_parse_run_summary_takes_the_final_counts() -> None:
    output = "12 passed, 3 skipped, 1 failed in 4.00s\n"
    assert parse_summary(output) == {"passed": 12, "failed": 1, "skipped": 3, "errors": 0}


def test_gate_semantics_fail_closed_on_every_violation() -> None:
    assert evaluate_gate(mode="collect", exit_code=0, collected=10, errors=0, failed=0) == []
    reasons = evaluate_gate(mode="collect", exit_code=2, collected=1, errors=1, failed=0)
    assert any("errors" in reason for reason in reasons)
    reasons = evaluate_gate(mode="collect", exit_code=5, collected=0, errors=0, failed=0)
    assert "no tests collected" in reasons
    reasons = evaluate_gate(mode="collect", exit_code=1, collected=7, errors=0, failed=0)
    assert any("exit code 1" in reason for reason in reasons)
    reasons = evaluate_gate(mode="run", exit_code=1, collected=9, errors=0, failed=1)
    assert any("failed tests" in reason for reason in reasons)


def test_gate_treats_collection_skips_as_informational_unless_strict() -> None:
    relaxed = evaluate_gate(
        mode="collect", exit_code=0, collected=100, errors=0, failed=0, skipped=3
    )
    assert relaxed == []  # reported, not fatal: D0.3 acceptance is zero *errors*
    strict = evaluate_gate(
        mode="collect",
        exit_code=0,
        collected=100,
        errors=0,
        failed=0,
        skipped=3,
        strict_skips=True,
    )
    assert strict == ["skipped tests are not allowed (--strict-skips): 3"]


def test_run_tier_uses_injected_runner_and_reports_counts() -> None:
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="tests/test_a.py: 4\n4 tests collected in 0.50s\n",
        stderr="",
    )
    seen: dict = {}

    def runner(command, **kwargs):
        seen["command"] = command
        seen["cwd"] = kwargs.get("cwd")
        return completed

    result = run_tier(TIERS["omnilss"], mode="collect", runner=runner, python="py")
    assert result["ok"] is True
    assert result["collected"] == 4
    assert result["errors"] == 0
    assert result["skipped"] == 0
    assert result["exit_code"] == 0
    assert result["strict_skips"] is False
    assert seen["cwd"].endswith("omnilss")
    assert seen["command"][:3] == ["py", "-m", "pytest"]


def test_run_tier_propagates_skips_and_strict_flag() -> None:
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="3319 tests collected / 2 skipped in 40.00s\n",
        stderr="",
    )
    relaxed = run_tier(
        TIERS["omnilss"], mode="collect", runner=lambda *a, **k: completed, python="py"
    )
    assert relaxed["skipped"] == 2
    assert relaxed["ok"] is True
    strict = run_tier(
        TIERS["omnilss"],
        mode="collect",
        runner=lambda *a, **k: completed,
        python="py",
        strict_skips=True,
    )
    assert strict["skipped"] == 2
    assert strict["strict_skips"] is True
    assert strict["ok"] is False
    assert any("--strict-skips" in reason for reason in strict["reasons"])


def test_run_tier_reports_timeout_and_runner_failure() -> None:
    def timing_out(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    def broken(command, **kwargs):
        return subprocess.CompletedProcess(
            args=command, returncode=2, stdout="", stderr="ERROR collecting tests/test_x.py\n"
        )

    timed_out = run_tier(TIERS["omnilss"], mode="collect", runner=timing_out, python="py")
    assert timed_out["exit_code"] == 124
    assert timed_out["ok"] is False
    failed = run_tier(TIERS["omnilss"], mode="collect", runner=broken, python="py")
    assert failed["exit_code"] == 2
    assert failed["errors"] == 1
    assert failed["ok"] is False
    assert failed["log_tail"]


def test_report_totals_render_and_fail_closed_label() -> None:
    results = [
        run_tier(
            TIERS["omnilss-server"],
            mode="collect",
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                args=command, returncode=0, stdout="1 test collected in 0.10s\n", stderr=""
            ),
            python="py",
        ),
        run_tier(
            TIERS["omnilss-pro"],
            mode="collect",
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                args=command, returncode=2, stdout="", stderr="ERROR collecting tests/test_automl.py\n"
            ),
            python="py",
        ),
    ]
    report = build_report(results, mode="collect")
    assert report["totals"] == {"tiers": 2, "collected": 1, "errors": 1, "skipped": 0, "failed": 0}
    assert report["ok"] is False
    assert report["reference_projects_excluded"] == list(REFERENCE_PROJECTS)
    rendered = render_report(report)
    assert "FAIL-CLOSED" in rendered
    assert "[ok] omnilss-server" in rendered and "[FAIL] omnilss-pro" in rendered
    assert json.dumps(report)  # report is JSON serialisable


def test_cli_lists_tiers_and_excludes_reference_projects(capsys) -> None:
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    for name in TIERS:
        assert name in output
    assert "ondil, gamlss-python" in output


def test_cli_runs_selected_tiers_with_injected_runner() -> None:
    completed = subprocess.CompletedProcess(
        args=["pytest"], returncode=0, stdout="3 tests collected in 0.20s\n", stderr=""
    )
    assert main(["--tier", "omnilss-pro", "--json"], runner=lambda *a, **k: completed) == 0
    broken = subprocess.CompletedProcess(args=["pytest"], returncode=2, stdout="", stderr="ERROR x\n")
    assert main(["--tier", "omnilss-pro"], runner=lambda *a, **k: broken) == 1


def test_cli_strict_skips_flag_fails_on_collection_skips(capsys) -> None:
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="10 tests collected / 1 skipped in 0.30s\n",
        stderr="",
    )
    assert main(["--tier", "omnilss-pro"], runner=lambda *a, **k: completed) == 0
    assert main(["--tier", "omnilss-pro", "--strict-skips"], runner=lambda *a, **k: completed) == 1
    output = capsys.readouterr().out
    assert "--strict-skips" in output
    assert "skipped tests are not allowed" in output


def test_small_tiers_collect_cleanly_in_the_real_repo() -> None:
    for name in ("omnilss-server", "omnilss-pro"):
        result = run_tier(TIERS[name], mode="collect", timeout=600.0)
        assert result["ok"], f"{name}: {result['reasons']} {result['log_tail'][-5:]}"
        assert result["collected"] >= 1
        assert result["errors"] == 0


@pytest.mark.slow
def test_all_four_tiers_collect_zero_errors() -> None:
    results = [run_tier(tier, mode="collect", timeout=1800.0) for tier in TIERS.values()]
    report = build_report(results, mode="collect")
    assert report["ok"], render_report(report)
    assert report["totals"]["errors"] == 0
    assert report["totals"]["collected"] > 3000  # the main suite alone collects thousands
    assert report["totals"]["skipped"] == 0
