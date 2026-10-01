# SPDX-License-Identifier: GPL-3.0-or-later
"""D0.2: golden manifest template and the fail-closed ``--require-r`` gate.

The fixed-R golden fixtures under ``omnilss/tests/fixtures/r_golden/regenerated``
carry R-side provenance (R version, gamlss / gamlss.dist versions, seed, content
hash). ``manifest_template.json`` pins everything a parity claim needs:

* R / gamlss / gamlss.dist / Python / JAX versions and ``jax_enable_x64``,
* the required per-case provenance fields,
* the comparison tolerances (single source of truth, per case overridable),
* the expected case set.

Modes (exit code 0 = ok, 1 = fail-closed):

``--check``      validate template + manifest + fixture content hashes only
                 (fast, never probes the environment);
``--require-r``  additionally probe the runtime; any missing dependency or
                 version drift fails the gate instead of skipping. Without
                 ``--require-r`` an unpinned runtime is only *labelled*
                 ``smoke-only`` so non-release runs can finish.

The manifest content hash is the djb2 hash of the fixture JSON text (without
the trailing newline) written by ``generate_golden.R``; this tool reproduces it
so fixture tampering or a stale regeneration is detected without running R.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "manifest_template.json"
MANIFEST_PATH = REPO_ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "regenerated" / "manifest.json"
FIXTURES_DIR = MANIFEST_PATH.parent

R_PACKAGES = ("gamlss", "gamlss.dist")
_HEX8 = re.compile(r"^[0-9a-f]{8}$")

#: One ``Rscript -e`` probe: R version + one line per required package.
_R_PROBE = (
    'cat(R.version.string, "\\n", sep="");'
    'cat(as.character(tryCatch(packageVersion("gamlss"), error = function(e) "MISSING")), "\\n", sep="");'
    'cat(as.character(tryCatch(packageVersion("gamlss.dist"), error = function(e) "MISSING")), "\\n", sep="");'
)

LABEL_PINNED = "pinned"
LABEL_SMOKE_ONLY = "smoke-only"


def djb2_hex(text: str) -> str:
    """Reproduce the ``content_hash`` of ``generate_golden.R`` (djb2, 8 hex)."""
    value = 5381
    for code_point in text:
        value = (value * 33 + ord(code_point)) % 4294967296
    return "".join("0123456789abcdef"[(value >> (4 * (7 - index))) & 0xF] for index in range(8))


def load_json(path: Path) -> tuple[object | None, str]:
    """Return ``(parsed, error)``; error is empty when the JSON is readable."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as error:  # noqa: BLE001 - the gate reports any read failure
        return None, f"{type(error).__name__}: {error}"


def effective_tolerance(case_name: str, manifest: dict, template: dict) -> float:
    """Resolve the tolerance: manifest override -> template case -> kind -> default."""
    tolerances = template.get("tolerances", {})
    case_entry = manifest.get(case_name, {}) if isinstance(manifest, dict) else {}
    if isinstance(case_entry, dict) and isinstance(case_entry.get("tolerance"), (int, float)):
        return float(case_entry["tolerance"])
    template_case = template.get("cases", {}).get(case_name, {})
    kind = template_case.get("kind") if isinstance(template_case, dict) else None
    if isinstance(template_case, dict) and isinstance(template_case.get("tolerance"), (int, float)):
        return float(template_case["tolerance"])
    if kind and kind in tolerances:
        return float(tolerances[kind])
    return float(tolerances.get("default", 0.0))


def validate_template(template: object) -> list[str]:
    """Return the issues that make the template unusable as a pin."""
    issues: list[str] = []
    if not isinstance(template, dict):
        return ["template 顶层应为 object"]
    if template.get("schema_version") != 1:
        issues.append(f"template schema_version 应为 1，实际 {template.get('schema_version')!r}")
    pinned = template.get("pinned")
    if not isinstance(pinned, dict):
        issues.append("template 缺 pinned 对象")
    else:
        for key in (
            "R_version_prefix",
            "gamlss_version",
            "gamlss_dist_version",
            "python_version_prefix",
            "jax_version_prefix",
        ):
            if not isinstance(pinned.get(key), str) or not pinned[key]:
                issues.append(f"template pinned.{key} 缺失或为空")
        if pinned.get("jax_enable_x64") is not True:
            issues.append("template pinned.jax_enable_x64 必须为 true（parity 前提）")
    fields = template.get("required_case_fields")
    if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
        issues.append("template required_case_fields 应为字符串数组")
    else:
        for key in ("family", "seed", "content_hash", "R_version"):
            if key not in fields:
                issues.append(f"template required_case_fields 缺 {key}")
    tolerances = template.get("tolerances")
    if not isinstance(tolerances, dict) or not isinstance(tolerances.get("default"), (int, float)):
        issues.append("template tolerances.default 缺失或非数值")
    else:
        for key, value in tolerances.items():
            if not isinstance(value, (int, float)) or value <= 0:
                issues.append(f"template tolerances.{key} 必须为正数")
    cases = template.get("cases")
    if not isinstance(cases, dict) or not cases:
        issues.append("template cases 应为非空 object")
    else:
        for name, entry in cases.items():
            if not isinstance(entry, dict):
                issues.append(f"template cases.{name} 应为 object")
            elif "kind" not in entry and "tolerance" not in entry:
                issues.append(f"template cases.{name} 需要 kind 或 tolerance")
    return issues


def validate_manifest(
    manifest: object,
    template: object,
    fixtures_dir: Path = FIXTURES_DIR,
) -> list[str]:
    """Return every reproducibility issue found in the manifest + fixtures."""
    issues: list[str] = []
    if not isinstance(manifest, dict) or not manifest:
        return ["manifest 顶层应为非空 object"]
    if not isinstance(template, dict):
        return validate_template(template)
    expected_cases = template.get("cases", {})
    missing = sorted(set(expected_cases) - set(manifest))
    extra = sorted(set(manifest) - set(expected_cases))
    if missing:
        issues.append(f"manifest 缺 template 声明的 case: {missing}")
    if extra:
        issues.append(f"manifest 含 template 未声明的 case: {extra}")
    required = template.get("required_case_fields", [])
    pinned = template.get("pinned", {}) if isinstance(template.get("pinned"), dict) else {}
    for name in sorted(manifest):
        entry = manifest[name]
        if not isinstance(entry, dict):
            issues.append(f"case {name}: 条目应为 object")
            continue
        for field in required:
            if field not in entry:
                issues.append(f"case {name}: 缺 provenance 字段 {field}")
        seed, count = entry.get("seed"), entry.get("n")
        if not isinstance(seed, int):
            issues.append(f"case {name}: seed 应为整数，实际 {seed!r}")
        if not isinstance(count, int) or count <= 0:
            issues.append(f"case {name}: n 应为正整数，实际 {count!r}")
        content_hash = entry.get("content_hash")
        if not isinstance(content_hash, str) or not _HEX8.match(content_hash):
            issues.append(f"case {name}: content_hash 应为 8 位小写 hex，实际 {content_hash!r}")
        r_version = entry.get("R_version")
        prefix = pinned.get("R_version_prefix")
        if prefix and isinstance(r_version, str) and not r_version.startswith(prefix):
            issues.append(f"case {name}: R 版本漂移 {r_version}（pinned {prefix}）")
        for field in ("gamlss_version", "gamlss_dist_version"):
            pinned_value = pinned.get(field)
            if pinned_value and entry.get(field) != pinned_value:
                issues.append(
                    f"case {name}: {field} 漂移 {entry.get(field)!r}（pinned {pinned_value}）"
                )
        fixture = fixtures_dir / f"{name}.json"
        if not fixture.exists():
            issues.append(f"case {name}: fixture 缺失 {fixture.name}")
        elif isinstance(content_hash, str) and _HEX8.match(content_hash):
            text = fixture.read_text(encoding="utf-8")
            if djb2_hex(text.rstrip("\n")) != content_hash:
                issues.append(
                    f"case {name}: fixture 内容哈希与 manifest 不符"
                    "（fixture 被改动或 manifest 未随再生成更新）"
                )
    return issues


def probe_environment(
    *,
    rscript: str = "Rscript",
    runner=None,
    probe_jax: bool = True,
    timeout: float = 180.0,
) -> dict:
    """Probe the runtime a parity run would actually use.

    ``runner``/``probe_jax`` exist so unit tests stay fast and R-free. The JAX
    probe imports ``omnilss`` first because the parity tests do, and that import
    is what enables ``jax_enable_x64`` (see task 0.7).
    """
    runner = runner or subprocess.run
    environment: dict = {
        "python_version": platform.python_version(),
        "jax_version": None,
        "jax_enable_x64": None,
        "jax_error": None,
        "rscript": rscript,
        "rscript_available": False,
        "r_version": None,
        "r_packages": {},
        "r_error": None,
    }
    if probe_jax:
        try:
            import jax  # noqa: PLC0415 - heavy dependency, probe only when asked

            environment["jax_version"] = jax.__version__
            src = REPO_ROOT / "omnilss" / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            import omnilss  # noqa: F401,PLC0415  # parity tests import omnilss (enables x64)

            flag = getattr(jax.config, "jax_enable_x64", None)
            if flag is None:
                flag = getattr(jax.config, "x64_enabled", None)
            environment["jax_enable_x64"] = None if flag is None else bool(flag)
        except Exception as error:  # noqa: BLE001 - report, never crash the gate
            environment["jax_error"] = f"{type(error).__name__}: {error}"
    try:
        result = runner(
            [rscript, "-e", _R_PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        environment["r_error"] = f"{rscript} 不在 PATH 中"
        return environment
    except subprocess.TimeoutExpired:
        environment["r_error"] = f"{rscript} 探针超时（>{timeout}s）"
        return environment
    if getattr(result, "returncode", 1) != 0:
        stderr = (getattr(result, "stderr", "") or "").strip()
        environment["r_error"] = (
            f"{rscript} 退出码 {getattr(result, 'returncode', '?')}: {stderr[:300]}"
        )
        return environment
    lines = [line.strip() for line in (getattr(result, "stdout", "") or "").splitlines()]
    if not lines:
        environment["r_error"] = f"{rscript} 无输出"
        return environment
    environment["rscript_available"] = True
    environment["r_version"] = lines[0] or None
    for name, line in zip(R_PACKAGES, lines[1:]):
        environment["r_packages"][name] = line
    return environment


def classify(environment: dict, template: dict) -> tuple[str, list[str]]:
    """Return ``(label, issues)``; anything wrong demotes the run to smoke-only."""
    issues: list[str] = []
    pinned = template.get("pinned", {}) if isinstance(template.get("pinned"), dict) else {}
    python_version = environment.get("python_version")
    prefix = pinned.get("python_version_prefix")
    if prefix and (not python_version or not python_version.startswith(prefix)):
        issues.append(f"Python 版本漂移 {python_version}（pinned {prefix}）")
    if environment.get("rscript_available"):
        r_version = environment.get("r_version")
        r_prefix = pinned.get("R_version_prefix")
        if r_prefix and (not r_version or not r_version.startswith(r_prefix)):
            issues.append(f"R 版本漂移 {r_version}（pinned {r_prefix}）")
        for package, key in (("gamlss", "gamlss_version"), ("gamlss.dist", "gamlss_dist_version")):
            actual = environment.get("r_packages", {}).get(package)
            pinned_value = pinned.get(key)
            if actual in (None, "", "MISSING"):
                issues.append(f"R 包缺失: {package}")
            elif pinned_value and actual != pinned_value:
                issues.append(f"{package} 版本漂移 {actual}（pinned {pinned_value}）")
    else:
        reason = environment.get("r_error") or f"{environment.get('rscript')} 不可用"
        issues.append(f"R 不可用，parity 只能标 smoke-only: {reason}")
    if environment.get("jax_error"):
        issues.append(f"JAX 探针失败: {environment['jax_error']}")
    else:
        jax_version = environment.get("jax_version")
        jax_prefix = pinned.get("jax_version_prefix")
        if jax_prefix and (not jax_version or not jax_version.startswith(jax_prefix)):
            issues.append(f"JAX 版本漂移 {jax_version}（pinned {jax_prefix}）")
        if pinned.get("jax_enable_x64") is True and environment.get("jax_enable_x64") is not True:
            issues.append(f"jax_enable_x64 未开启（实际 {environment.get('jax_enable_x64')}）")
    label = LABEL_PINNED if not issues else LABEL_SMOKE_ONLY
    return label, issues


def build_report(
    *,
    template_path: Path = TEMPLATE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    fixtures_dir: Path = FIXTURES_DIR,
    probe_env: bool = False,
    require_r: bool = False,
    probe=None,
) -> dict:
    """Build the gate report: validation, optional probe, label, overall ok.

    ``probe_env`` runs the environment probe and produces a label;
    ``require_r`` additionally makes environment issues fail the gate (the
    fail-closed mode used by release / parity runs).
    """
    template, template_error = load_json(template_path)
    manifest, manifest_error = load_json(manifest_path)
    issues: list[str] = []
    if template_error:
        issues.append(f"template 不可读: {template_error}")
        template = None
    else:
        issues.extend(validate_template(template))
    if manifest_error:
        issues.append(f"manifest 不可读: {manifest_error}")
        manifest = None
    elif template is not None:
        issues.extend(validate_manifest(manifest, template, fixtures_dir))
    report: dict = {
        "template": str(template_path),
        "manifest": str(manifest_path),
        "cases": len(manifest) if isinstance(manifest, dict) else 0,
        "tolerances": {},
        "validation_issues": issues,
        "environment": None,
        "environment_issues": [],
        "label": None,
        "ok": not issues,
    }
    if isinstance(manifest, dict) and isinstance(template, dict):
        report["tolerances"] = {
            name: effective_tolerance(name, manifest, template) for name in sorted(manifest)
        }
    if probe_env or require_r:
        environment = (probe or probe_environment)()
        label, environment_issues = classify(environment, template or {})
        report["environment"] = environment
        report["environment_issues"] = environment_issues
        report["label"] = label
        if require_r:
            report["ok"] = report["ok"] and not environment_issues
    return report


def render_report(report: dict) -> str:
    """Human readable rendering of a gate report."""
    lines = ["golden manifest gate (D0.2)", f"template: {report['template']}", f"manifest: {report['manifest']}"]
    validation = report["validation_issues"]
    if validation:
        lines.append(f"validation: FAIL ({len(validation)} issue(s))")
        lines.extend(f"  - {issue}" for issue in validation)
    else:
        tolerance_values = sorted({value for value in report["tolerances"].values()})
        rendered = ", ".join(f"{value:g}" for value in tolerance_values) or "n/a"
        lines.append(f"validation: ok ({report['cases']} cases, tolerance {rendered})")
    environment = report["environment"]
    if environment is None:
        lines.append("environment: not probed (use --require-r; --check stays validation-only)")
    else:
        packages = ", ".join(
            f"{name}={environment['r_packages'].get(name, 'n/a')}" for name in R_PACKAGES
        )
        lines.append(
            f"environment: python {environment['python_version']}, "
            f"jax {environment['jax_version']} (x64={environment['jax_enable_x64']}), "
            f"r={'available' if environment['rscript_available'] else 'unavailable'} "
            f"[{environment['r_version'] or '-'}; {packages}]"
        )
        for issue in report["environment_issues"]:
            lines.append(f"  - {issue}")
        lines.append(f"label: {report['label']}")
    lines.append(f"result: {'OK' if report['ok'] else 'FAIL-CLOSED'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="validation only: never probes the environment (fast CI path)",
    )
    parser.add_argument(
        "--require-r",
        action="store_true",
        help="fail-closed: missing R, drift or disabled x64 fails the gate (smoke-only otherwise)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args(argv)
    require_r = bool(args.require_r) and not args.check
    report = build_report(
        template_path=args.template,
        manifest_path=args.manifest,
        fixtures_dir=args.fixtures_dir,
        probe_env=not args.check,
        require_r=require_r,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
