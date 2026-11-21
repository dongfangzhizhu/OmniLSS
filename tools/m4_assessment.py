# SPDX-License-Identifier: GPL-3.0-or-later
"""M4 (1.0.0) release-readiness assessment gate (D2.283/B4).

Aggregates every fail-closed M0..M3 verdict plus M4 deliverables into a single
machine-checked release readiness artifact:

  1. prior_milestones_intact     -- M0 manifest + M1 / M2 / M3 gates all pass.
  2. in_scope_r_surface_complete -- 100% of in-scope R symbols (11 packages)
                                    matched with 0 placeholders and 0 unverified.
  3. surrounding_packages_complete -- M4 scope inventory has 0 surface gaps and
                                      dedicated evidence tests for every package.
  4. capability_admission_backed -- M3 admission audit has 0 violations and the
                                    committed capability matrix matches runtime.
  5. service_boundaries_hardened -- HTTP server enforces production capability
                                    gates and both service tiers have tests.
  6. docs_localization_and_tutorials -- bilingual localization check passes and
                                        all 8 tutorials are wired into mkdocs.
  7. no_reference_repo_imports   -- AST check proving omnilss / omnilss-pro /
                                    omnilss-server never import ondil or
                                    gamlss-python.

Usage::

    python tools/m4_assessment.py --write docs/development/m4-assessment-2026-11-21.json
    python tools/m4_assessment.py --check
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

M0 = ROOT / "docs" / "development" / "milestone-manifest-m0.json"
M1 = ROOT / "docs" / "development" / "m1-assessment-2026-11-15.json"
M2 = ROOT / "docs" / "development" / "m2-assessment-2026-11-15.json"
M3 = ROOT / "docs" / "development" / "m3-assessment-2026-11-15.json"
M3_ADMISSION = ROOT / "docs" / "development" / "m3-admission-evidence-2026-11-15.json"
M4_SCOPE = ROOT / "docs" / "development" / "m4-scope-2026-11-15.json"
COVERAGE = ROOT / "docs" / "development" / "r-symbol-coverage.json"
CAPABILITY_MATRIX = ROOT / "docs" / "development" / "family-capability-matrix-2026-05-18.json"
MKDOCS = ROOT / "mkdocs.yml"
DEFAULT_OUT = ROOT / "docs" / "development" / "m4-assessment-2026-11-21.json"

SCHEMA_VERSION = 1
FORBIDDEN_REFERENCE_IMPORTS = frozenset({"ondil", "gamlss_python", "gamlss_py"})
EXPECTED_TUTORIALS = (
    "tutorials/quickstart.md",
    "tutorials/tutorial-families.md",
    "tutorials/tutorial-fitting-algorithms.md",
    "tutorials/tutorial-smoothers.md",
    "tutorials/tutorial-prediction.md",
    "tutorials/tutorial-model-selection.md",
    "tutorials/tutorial-r-consistency.md",
    "tutorials/tutorial-grpc-serving.md",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_docs_localization() -> list[str]:
    tool_path = ROOT / "omnilss" / "tools" / "check_docs_localization.py"
    spec = importlib.util.spec_from_file_location("check_docs_localization", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.validate_docs_localization(ROOT))


def _scan_reference_imports() -> list[str]:
    violations: list[str] = []
    roots = (
        ROOT / "omnilss" / "src",
        ROOT / "omnilss-pro" / "src",
        ROOT / "omnilss-server",
    )
    for base in roots:
        if not base.is_dir():
            continue
        for py_file in sorted(base.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            rel = py_file.relative_to(ROOT).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".", 1)[0]
                        if top in FORBIDDEN_REFERENCE_IMPORTS:
                            violations.append(f"{rel}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".", 1)[0]
                    if top in FORBIDDEN_REFERENCE_IMPORTS:
                        violations.append(f"{rel}:{node.lineno}: from {node.module}")
    return violations


def _check_service_boundaries() -> dict:
    server_main = ROOT / "omnilss-server" / "main.py"
    server_test = ROOT / "omnilss-server" / "test_server_api.py"
    pro_tls_test = ROOT / "omnilss-pro" / "tests" / "test_client_tls.py"
    server_src = server_main.read_text(encoding="utf-8") if server_main.is_file() else ""
    has_prod_gate = "require_production_family_route" in server_src
    has_rate_prune = "_prune_rate_limit_state" in server_src
    return {
        "ok": bool(
            server_main.is_file()
            and server_test.is_file()
            and pro_tls_test.is_file()
            and has_prod_gate
            and has_rate_prune
        ),
        "server_production_gate_wired": has_prod_gate,
        "server_rate_limit_bounded": has_rate_prune,
        "server_tests_present": server_test.is_file(),
        "pro_tls_tests_present": pro_tls_test.is_file(),
    }


def _check_tutorials() -> dict:
    mkdocs_text = MKDOCS.read_text(encoding="utf-8") if MKDOCS.is_file() else ""
    missing_nav = [ref for ref in EXPECTED_TUTORIALS if ref not in mkdocs_text]
    missing_cn = [
        ref
        for ref in EXPECTED_TUTORIALS
        if not (ROOT / "docs" / ref.replace(".md", "_cn.md")).is_file()
    ]
    localization_errors = _check_docs_localization()
    return {
        "ok": not missing_nav and not missing_cn and not localization_errors,
        "tutorials_expected": len(EXPECTED_TUTORIALS),
        "missing_from_mkdocs_nav": missing_nav,
        "missing_chinese_counterparts": missing_cn,
        "localization_errors": localization_errors,
    }


def build() -> dict:
    from omnilss.family_capabilities import capability_matrix

    m0 = _read(M0)
    m1 = _read(M1)
    m2 = _read(M2)
    m3 = _read(M3)
    m3_adm = _read(M3_ADMISSION)
    m4_scope = _read(M4_SCOPE)
    cov = _read(COVERAGE)
    cap_file = _read(CAPABILITY_MATRIX)

    in_scope = [s for s in cov["symbols"] if s.get("in_scope")]
    in_scope_missing = [s["r_symbol"] for s in in_scope if s["layers"]["surface"] == "missing"]
    in_scope_placeholder = [
        s["r_symbol"]
        for s in in_scope
        if s["layers"]["surface"] == "placeholder" or s["layers"]["kernel"] == "placeholder"
    ]
    in_scope_unverified = [s["r_symbol"] for s in in_scope if s["layers"]["parity"] == "unverified"]

    m0_gates = m0.get("gates", {})
    m0_ok = bool(m0_gates) and all(
        (bool(g.get("ok")) if isinstance(g, dict) else bool(g))
        for g in m0_gates.values()
    ) and bool(m0.get("environment_pinned_ok", False))
    prior_ok = (
        m0_ok
        and bool(m1["verdict"]["m1_gate_met"])
        and bool(m2["verdict"]["m2_gate_met"])
        and bool(m3["verdict"]["m3_gate_met"])
    )

    cap_runtime = capability_matrix()
    cap_in_sync = cap_file == cap_runtime
    adm_ok = (
        bool(m3_adm["verdict"]["registry_claims_backed"])
        and m3_adm["registry_audit"]["violations"] == []
        and int(m3_adm["counts"]["admitted"]) == int(m3_adm["counts"]["promoted_families"])
        and cap_in_sync
    )

    scope_counts = m4_scope["counts"]
    scope_ok = (
        bool(m4_scope["verdict"]["inventory_complete"])
        and int(scope_counts["surface_gap"]) == 0
        and int(scope_counts["missing_symbols"]) == 0
        and int(scope_counts["packages_without_evidence_tests"]) == 0
    )

    service = _check_service_boundaries()
    tutorials = _check_tutorials()
    ref_violations = _scan_reference_imports()

    criteria = {
        "prior_milestones_intact": {
            "met": prior_ok,
            "detail": (
                f"M0={m0_ok}, M1={m1['verdict']['m1_gate_met']}, "
                f"M2={m2['verdict']['m2_gate_met']}, M3={m3['verdict']['m3_gate_met']}"
            ),
        },
        "in_scope_r_surface_complete": {
            "met": not in_scope_missing and not in_scope_placeholder and not in_scope_unverified,
            "detail": (
                f"in-scope matched={len(in_scope) - len(in_scope_missing)}/{len(in_scope)}, "
                f"missing={len(in_scope_missing)}, placeholder={len(in_scope_placeholder)}, "
                f"unverified={len(in_scope_unverified)}"
            ),
        },
        "surrounding_packages_complete": {
            "met": scope_ok,
            "detail": (
                f"packages={scope_counts['packages']}, surface_complete={scope_counts['surface_complete']}, "
                f"gaps={scope_counts['surface_gap']}, missing={scope_counts['missing_symbols']}"
            ),
        },
        "capability_admission_backed": {
            "met": adm_ok,
            "detail": (
                f"admitted={m3_adm['counts']['admitted']}/{m3_adm['counts']['promoted_families']}, "
                f"audit_violations={len(m3_adm['registry_audit']['violations'])}, "
                f"matrix_artifact_synced={cap_in_sync}"
            ),
        },
        "service_boundaries_hardened": {
            "met": bool(service["ok"]),
            "detail": (
                f"prod_gate={service['server_production_gate_wired']}, "
                f"rate_limit_bounded={service['server_rate_limit_bounded']}, "
                f"server_tests={service['server_tests_present']}, "
                f"pro_tls_tests={service['pro_tls_tests_present']}"
            ),
        },
        "docs_localization_and_tutorials": {
            "met": bool(tutorials["ok"]),
            "detail": (
                f"tutorials={tutorials['tutorials_expected']}, "
                f"missing_nav={len(tutorials['missing_from_mkdocs_nav'])}, "
                f"missing_cn={len(tutorials['missing_chinese_counterparts'])}, "
                f"localization_errors={len(tutorials['localization_errors'])}"
            ),
        },
        "no_reference_repo_imports": {
            "met": not ref_violations,
            "detail": f"forbidden reference imports={len(ref_violations)}",
        },
    }

    met_count = sum(1 for item in criteria.values() if item["met"])
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M4",
        "target_version": "1.0.0",
        "decision": "D2.283",
        "sources": {
            "m0_manifest": "docs/development/milestone-manifest-m0.json",
            "m1_assessment": "docs/development/m1-assessment-2026-11-15.json",
            "m2_assessment": "docs/development/m2-assessment-2026-11-15.json",
            "m3_assessment": "docs/development/m3-assessment-2026-11-15.json",
            "m3_admission": "docs/development/m3-admission-evidence-2026-11-15.json",
            "m4_scope": "docs/development/m4-scope-2026-11-15.json",
            "coverage": "docs/development/r-symbol-coverage.json",
            "capability_matrix": "docs/development/family-capability-matrix-2026-05-18.json",
        },
        "checks": {
            "service_boundaries": service,
            "tutorials": tutorials,
            "reference_import_violations": ref_violations,
        },
        "criteria": criteria,
        "counts": {
            "criteria": len(criteria),
            "met": met_count,
            "in_scope_symbols": len(in_scope),
            "in_scope_missing": len(in_scope_missing),
            "in_scope_unverified": len(in_scope_unverified),
            "admitted_families": int(m3_adm["counts"]["admitted"]),
        },
        "verdict": {
            "m4_gate_met": all(item["met"] for item in criteria.values()),
            "notes": (
                "M4 (1.0.0) gate aggregates M0..M3 verdicts, 100% in-scope R surface "
                "coverage with 0 unverified symbols, surrounding package completeness, "
                "fail-closed capability admission, hardened service boundaries, "
                "bilingual tutorial/doc compliance, and strict isolation from reference "
                "repos (ondil / gamlss-python). Out-of-scope gamlss2, P5 online/multivariate "
                "extensions, and P6 GPU kernels are independent tracks."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "milestone": artifact["milestone"],
        "target_version": artifact["target_version"],
        "checks": artifact["checks"],
        "criteria": artifact["criteria"],
        "counts": artifact["counts"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    c = a["counts"]
    lines = [
        "# M4 assessment (target 1.0.0)",
        "",
        f"- m4_gate_met: **{a['verdict']['m4_gate_met']}** ({c['met']}/{c['criteria']} criteria)",
        f"- in-scope R symbols: {c['in_scope_symbols'] - c['in_scope_missing']}/{c['in_scope_symbols']} matched, {c['in_scope_unverified']} unverified",
        f"- production_safe admitted families: {c['admitted_families']}",
        "",
        "| criterion | met | detail |",
        "|---|---|---|",
    ]
    for name, item in a["criteria"].items():
        lines.append(f"| {name} | {item['met']} | {item['detail']} |")
    lines += ["", a["verdict"]["notes"], ""]
    return "\n".join(lines)


def write_artifact(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text(_render_markdown(artifact), encoding="utf-8")


def check_artifact(path: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not path.is_file():
        return False, [f"artifact not found: {path}"]
    artifact = _read(path)
    for key in ("schema_version", "milestone", "criteria", "counts", "verdict", "checks"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build()
    if _stable(artifact) != _stable(current):
        for key in ("checks", "criteria", "counts", "verdict"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drifted vs current tree")
    if not current["verdict"]["m4_gate_met"]:
        failed = [k for k, v in current["criteria"].items() if not v["met"]]
        problems.append(f"M4 gate not met: {', '.join(failed)}")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_artifact(args.path)
        print(f"M4 assessment check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for problem in problems:
            print(f"  - {problem}")
        print("result: FAIL")
        return 1

    artifact = build()
    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
    if args.write is not None:
        write_artifact(artifact, args.write)
        log(f"wrote {args.write}")
        log(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    c = artifact["counts"]
    log(f"m4_gate_met={artifact['verdict']['m4_gate_met']} ({c['met']}/{c['criteria']})")
    return 0 if artifact["verdict"]["m4_gate_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
