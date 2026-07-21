"""Generate reproducible fixed-R-source evidence for family capability admission.

Rejected families intentionally produce an ``excluded`` comparison outcome.  That
outcome is never an R-consistency success and retains the source/evidence path
needed to audit why no numerical comparison was attempted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # supports both ``python tools/...`` and package imports in tests
    from tools.generate_release1_evidence_matrix import (
        DEFAULT_OUTPUT_DIR,
        REPO_ROOT,
        build_matrix,
        source_evidence,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_release1_evidence_matrix import (
        DEFAULT_OUTPUT_DIR,
        REPO_ROOT,
        build_matrix,
        source_evidence,
    )

CASE_ID = "family-capability-gate-gb2-rs-seed-20260722"
FAMILY = "GB2"
METHOD = "RS"
SEED = 20260722


def build_case(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build the deterministic rejected-family evidence case."""
    from omnilss.family_capabilities import require_production_family_route

    matrix = build_matrix(repo_root)
    entry = next(
        item for item in matrix["entries"]
        if item["r_symbol"] == FAMILY and item["python_entry"]
    )
    try:
        admission = require_production_family_route(FAMILY, METHOD, route="benchmark")
    except Exception as exc:
        admission = getattr(exc, "to_dict", lambda: {"message": str(exc)})()
    if admission.get("ok"):
        raise ValueError("evidence case must remain rejected")
    return {
        "schema_version": 1,
        "case_id": CASE_ID,
        "family": FAMILY,
        "method": METHOD,
        "seed": SEED,
        "fixed_r_sources": source_evidence(repo_root),
        "family_status": entry["status"],
        "implementation_status": admission.get("implementation_status"),
        "admission": admission,
        "evidence_paths": entry["evidence_references"],
        "comparison": {
            "status": "excluded",
            "r_consistency_success": False,
            "reason": "production capability rejection occurred before R comparison",
        },
    }


def write_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    """Write JSON evidence and a readable derived summary."""
    case = build_case(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "family-capability-gate-case.json"
    markdown_path = output_dir / "family-capability-gate-case.md"
    json_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "# Family capability gate evidence\n\n"
        f"- Case: `{case['case_id']}`\n"
        f"- Family/status: `{case['family']}` / `{case['implementation_status']}`\n"
        f"- Admission: rejected (`{case['admission']['code']}`)\n"
        "- R consistency: excluded; this rejection is not a success result.\n"
        "- Fixed R source evidence: `r-source-versions.json` and the JSON case record.\n",
        encoding="utf-8",
    )
    return json_path, markdown_path


if __name__ == "__main__":
    for artifact in write_artifacts():
        print(artifact)
