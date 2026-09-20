"""Validate coverage/runtime consistency and production route admission.

The report is fail-closed for registry drift and for placeholder or compatibility
alias families that accidentally become admissible through the production route.
It intentionally validates the shared runtime admission helper instead of
reimplementing service-specific routing rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE = ROOT / "docs" / "development" / "r-export-coverage.json"
SRC = ROOT / "omnilss" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _issue(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


def validate(coverage: dict[str, Any]) -> dict[str, Any]:
    from omnilss.family_capabilities import (
        ImplementationStatus,
        family_capability_names,
        get_family_capability,
        method_route_capability_report,
        require_production_family_route,
        validate_capability_matrix_payload,
    )

    issues: list[dict[str, Any]] = []
    families = coverage.get("families")
    if not isinstance(families, dict):
        return {"ok": False, "issues": [_issue("coverage_families_missing", "coverage families must be an object")], "families": {}}

    runtime_names = set(family_capability_names())
    coverage_names = set(families)
    for name in sorted(runtime_names - coverage_names):
        issues.append(_issue("coverage_missing_family", f"runtime family {name!r} is absent from coverage", family=name))
    for name in sorted(coverage_names - runtime_names):
        issues.append(_issue("coverage_extra_family", f"coverage family {name!r} is absent from runtime registry", family=name))

    status_counts: dict[str, int] = {}
    route_results: dict[str, dict[str, Any]] = {}
    for name in sorted(runtime_names):
        capability = get_family_capability(name)
        expected = capability.implementation_status.value
        actual = families.get(name, {}).get("implementation_status")
        status_counts[expected] = status_counts.get(expected, 0) + 1
        if actual != expected:
            issues.append(_issue("implementation_status_drift", f"coverage status for {name!r} is {actual!r}, runtime is {expected!r}", family=name, coverage_status=actual, runtime_status=expected))

        report = method_route_capability_report(name, "RS", strict=False)
        route_results[name] = {"runtime_report": report}
        production_rejected = False
        rejection_code = None
        try:
            require_production_family_route(name, "RS", route="capability_gate")
        except Exception as exc:
            production_rejected = True
            rejection_code = getattr(exc, "report", {}).get("code") if hasattr(exc, "report") else None
            if rejection_code is None:
                rejection_code = report.get("code")
        route_results[name]["production_rejected"] = production_rejected
        route_results[name]["rejection_code"] = rejection_code
        if expected in {ImplementationStatus.PLACEHOLDER.value, ImplementationStatus.COMPAT_ALIAS.value}:
            if not production_rejected:
                issues.append(_issue("unrejected_nonproduction_family", f"{name!r} is {expected} but production RS route was admitted", family=name, implementation_status=expected))
        elif report.get("code") in {"placeholder_requires_opt_in", "compat_alias_requires_opt_in"}:
            issues.append(_issue("unexpected_route_status", f"runtime route report for {name!r} contradicts implementation status", family=name, code_from_report=report.get("code")))

    matrix = validate_capability_matrix_payload(coverage) if "version" in coverage and "method_capability_features" in coverage else None
    if matrix is not None and not matrix["ok"]:
        issues.append(_issue("capability_matrix_invalid", "coverage payload is not a capability matrix artifact", details=matrix["issues"]))

    return {
        "ok": not issues,
        "runtime_family_count": len(runtime_names),
        "coverage_family_count": len(coverage_names),
        "status_counts": status_counts,
        "issues": issues,
        "routes": route_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", nargs="?", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.coverage.read_text(encoding="utf-8"))
        report = validate(payload)
    except (OSError, json.JSONDecodeError) as exc:
        report = {"ok": False, "issues": [_issue("coverage_read_error", str(exc))]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
