"""Tests for the D0.2 golden manifest template and ``--require-r`` gate.

Everything here runs without R or JAX: environment probing is injected with a
fake runner/probe so the fail-closed semantics are asserted deterministically.
One ``slow`` test exercises the real CLI end to end.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.golden_manifest_gate import (  # noqa: E402
    FIXTURES_DIR,
    MANIFEST_PATH,
    TEMPLATE_PATH,
    build_report,
    classify,
    djb2_hex,
    effective_tolerance,
    main,
    probe_environment,
    render_report,
    validate_manifest,
    validate_template,
)

SCRIPT = ROOT / "tools" / "golden_manifest_gate.py"


def _template() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _pinned_env() -> dict:
    pinned = _template()["pinned"]
    return {
        "python_version": pinned["python_version_prefix"] + ".19",
        "jax_version": pinned["jax_version_prefix"] + ".2",
        "jax_enable_x64": True,
        "jax_error": None,
        "rscript": "Rscript",
        "rscript_available": True,
        "r_version": pinned["R_version_prefix"] + " (2026-04-24 ucrt)",
        "r_packages": {
            "gamlss": pinned["gamlss_version"],
            "gamlss.dist": pinned["gamlss_dist_version"],
        },
        "r_error": None,
    }


def test_template_pins_versions_tolerances_and_case_set() -> None:
    template = _template()
    assert validate_template(template) == []
    pinned = template["pinned"]
    assert pinned["jax_enable_x64"] is True
    for key in ("R_version_prefix", "gamlss_version", "gamlss_dist_version", "python_version_prefix", "jax_version_prefix"):
        assert isinstance(pinned[key], str) and pinned[key]
    assert template["tolerances"]["default"] > 0
    assert set(template["cases"]) == set(_manifest())


def test_template_rejects_disabled_x64_and_bad_tolerances() -> None:
    broken = copy.deepcopy(_template())
    broken["pinned"]["jax_enable_x64"] = False
    broken["tolerances"]["default"] = 0
    issues = validate_template(broken)
    assert any("jax_enable_x64" in issue for issue in issues)
    assert any("tolerances.default" in issue for issue in issues)


def test_manifest_and_fixtures_validate_against_the_template() -> None:
    assert validate_manifest(_manifest(), _template(), FIXTURES_DIR) == []


def test_every_manifest_content_hash_is_reproducible() -> None:
    manifest = _manifest()
    for case, entry in sorted(manifest.items()):
        text = (FIXTURES_DIR / f"{case}.json").read_text(encoding="utf-8")
        assert djb2_hex(text.rstrip("\n")) == entry["content_hash"], case


def test_tampered_fixture_is_detected(tmp_path: Path) -> None:
    manifest = _manifest()
    for case in manifest:
        (tmp_path / f"{case}.json").write_text(
            (FIXTURES_DIR / f"{case}.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
    victim = next(iter(manifest))
    original = (tmp_path / f"{victim}.json").read_text(encoding="utf-8")
    (tmp_path / f"{victim}.json").write_text(original + " ", encoding="utf-8")
    issues = validate_manifest(manifest, _template(), tmp_path)
    assert issues == [f"case {victim}: fixture 内容哈希与 manifest 不符"
                      "（fixture 被改动或 manifest 未随再生成更新）"]


def test_case_set_drift_is_reported() -> None:
    template = _template()
    manifest = _manifest()
    dropped = dict(manifest)
    dropped.pop(next(iter(dropped)))
    dropped["unknown_case_y_x"] = copy.deepcopy(next(iter(manifest.values())))
    issues = validate_manifest(dropped, template, FIXTURES_DIR)
    assert any("缺 template 声明的 case" in issue for issue in issues)
    assert any("未声明" in issue for issue in issues)


def test_missing_field_and_version_drift_are_reported() -> None:
    template = _template()
    manifest = _manifest()
    name = next(iter(manifest))
    manifest[name].pop("seed")
    manifest[name]["gamlss_version"] = "9.9.9"
    issues = validate_manifest(manifest, template, FIXTURES_DIR)
    assert any(f"case {name}: 缺 provenance 字段 seed" in issue for issue in issues)
    assert any("gamlss_version 漂移" in issue for issue in issues)


def test_effective_tolerance_resolution() -> None:
    template = _template()
    manifest = _manifest()
    name = next(iter(manifest))
    assert effective_tolerance(name, manifest, template) == template["tolerances"]["fitting"]
    with_case_override = copy.deepcopy(manifest)
    with_case_override[name]["tolerance"] = 1e-9
    assert effective_tolerance(name, with_case_override, template) == 1e-9
    with_kind_default = copy.deepcopy(template)
    with_kind_default["tolerances"]["fitting"] = 1e-8
    assert effective_tolerance(name, manifest, with_kind_default) == 1e-8
    assert effective_tolerance("unknown", manifest, template) == template["tolerances"]["default"]


def test_classify_pinned_environment() -> None:
    label, issues = classify(_pinned_env(), _template())
    assert label == "pinned"
    assert issues == []


def test_classify_without_r_is_smoke_only() -> None:
    environment = _pinned_env()
    environment["rscript_available"] = False
    environment["r_error"] = "Rscript 不在 PATH 中"
    label, issues = classify(environment, _template())
    assert label == "smoke-only"
    assert any("R 不可用" in issue and "smoke-only" in issue for issue in issues)


def test_classify_drift_and_missing_package_are_smoke_only() -> None:
    environment = _pinned_env()
    environment["r_version"] = "R version 4.5.0 (2025-04-11)"
    environment["r_packages"]["gamlss.dist"] = "MISSING"
    label, issues = classify(environment, _template())
    assert label == "smoke-only"
    assert any("R 版本漂移" in issue for issue in issues)
    assert any("R 包缺失: gamlss.dist" in issue for issue in issues)


def test_classify_disabled_x64_is_smoke_only() -> None:
    environment = _pinned_env()
    environment["jax_enable_x64"] = False
    label, issues = classify(environment, _template())
    assert label == "smoke-only"
    assert any("jax_enable_x64" in issue for issue in issues)


def test_probe_environment_parses_runner_output_without_r() -> None:
    completed = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=0,
        stdout='R version 4.6.0 (2026-04-24 ucrt)\n5.5.0\n6.1.1\n',
        stderr="",
    )
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        return completed

    environment = probe_environment(runner=runner, probe_jax=False)
    assert environment["rscript_available"] is True
    assert environment["r_version"].startswith("R version 4.6.0")
    assert environment["r_packages"] == {"gamlss": "5.5.0", "gamlss.dist": "6.1.1"}
    assert len(calls) == 1 and calls[0][0] == "Rscript" and "-e" in calls[0]


def test_probe_environment_fails_closed_when_rscript_is_missing() -> None:
    def runner(cmd, **kwargs):
        raise FileNotFoundError("Rscript")

    environment = probe_environment(runner=runner, probe_jax=False)
    assert environment["rscript_available"] is False
    assert "不在 PATH" in environment["r_error"]
    label, issues = classify(environment, _template())
    assert label == "smoke-only"


def test_require_r_is_fail_closed_while_plain_report_only_labels() -> None:
    def probe_without_r():
        environment = _pinned_env()
        environment["rscript_available"] = False
        environment["r_error"] = "Rscript 不在 PATH 中"
        return environment

    labelled = build_report(probe_env=True, require_r=False, probe=probe_without_r)
    assert labelled["validation_issues"] == []
    assert labelled["label"] == "smoke-only"
    assert labelled["ok"] is True  # non-release runs finish with the label only
    enforced = build_report(probe_env=True, require_r=True, probe=probe_without_r)
    assert enforced["label"] == "smoke-only"
    assert enforced["ok"] is False  # release runs fail closed
    rendered = render_report(enforced)
    assert "FAIL-CLOSED" in rendered and "smoke-only" in rendered


def test_build_report_without_probe_stays_validation_only() -> None:
    report = build_report()
    assert report["label"] is None
    assert report["environment"] is None
    assert report["ok"] is True
    assert report["cases"] == len(_manifest())
    assert set(report["tolerances"]) == set(_manifest())


def test_cli_check_is_the_fast_validation_path() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validation: ok" in result.stdout
    assert "not probed" in result.stdout
    assert main(["--check"]) == 0


@pytest.mark.slow
def test_cli_require_r_reports_the_real_runtime() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--require-r", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    payload = json.loads(result.stdout)
    assert payload["cases"] == len(_manifest())
    assert payload["validation_issues"] == []
    assert payload["label"] in {"pinned", "smoke-only"}
    if payload["label"] == "pinned":
        assert result.returncode == 0
    else:
        assert result.returncode == 1
        assert payload["environment_issues"]
