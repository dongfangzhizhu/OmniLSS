# SPDX-License-Identifier: GPL-3.0-or-later
"""M3 production_safe admission evidence inventory (D2.240).

M3 asks for P3 core API + family/link framework + admission closure.  This tool
covers the admission slice only, and it does so as an *inventory*, not a flip:
it records, for every promoted family, which admission evidence exists and
which registry claims that evidence supports.

Two different questions are kept apart on purpose:

  measurement -- what the committed artifacts prove (three parity gates, golden
    fit fixture, M2 live newdata predictions and JSON round-trip, layered route
    parity).  This is computed from artifacts, never from a hand-maintained set.

  registry -- what omnilss.family_capabilities currently claims
    (_PRODUCTION_SAFE, _RS_FIT_EVIDENCE, _PREDICTION_EVIDENCE).

The verdict is fail-closed in one direction only: a registry claim that the
artifacts do not back is a violation.  The other direction (evidence exists but
the family is not yet admitted) is reported as an eligible-but-unregistered
list -- a decision, not an error.

Usage::

    python tools/m3_admission_evidence.py --write docs/development/m3-admission-evidence-2026-11-15.json
    python tools/m3_admission_evidence.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

from omnilss.family_capabilities import (  # noqa: E402
    _PREDICTION_EVIDENCE,
    _PRODUCTION_SAFE,
    _R_CONSISTENCY_EVIDENCE,
    _RS_FIT_EVIDENCE,
    get_family_capability,
)

SCHEMA_VERSION = 1
FROZEN = ROOT / "docs" / "development" / "highfreq-families-frozen.json"
SCAN = ROOT / "docs" / "development" / "rs-parity-scan-2026-11-15.json"
M2 = ROOT / "docs" / "development" / "m2-assessment-2026-11-15.json"
LAYERED = ROOT / "docs" / "development" / "layered-parity-2026-11-15.json"
GOLDEN_MANIFEST = (
    ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "regenerated" / "manifest.json"
)
DEFAULT_OUT = ROOT / "docs" / "development" / "m3-admission-evidence-2026-11-15.json"

GATES = ("deriv", "dpqr", "rsdev")

#: Hard admission gates.  Each maps to a measurement source; a family missing
#: any of them is BLOCKED and cannot be admitted until the evidence exists.
HARD_GATES = (
    "deriv_gate_pass",
    "dpqr_gate_pass",
    "fit_gate_pass",
    "r_consistency_validated",
    "golden_fit_fixture",
    "live_newdata_predictions",
    "live_json_roundtrip",
    "live_converged",
    "no_route_mismatch",
)

#: Soft flags are recorded but do not block admission: flat_ridge means both
#: routes reached the same likelihood with a weakly identified parameter, and
#: r_side_route_error means R's own route raised.  Both must be listed in an
#: admission record so they are never silently forgotten.
SOFT_FLAGS = ("flat_ridge_cells", "r_side_route_errors")

#: Non-golden evidence modules for families the registry declares but the
#: golden set does not cover (currently ZIP's dedicated prediction test).
EXTRA_EVIDENCE_TESTS = {
    "ZIP": "omnilss/tests/test_zip_prediction_serialization.py",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _promoted() -> list[str]:
    return sorted(item["family"] for item in _read(FROZEN)["tiers"]["promoted"])


def _scan_by_family() -> dict[str, dict]:
    return {row["family"]: row for row in _read(SCAN)}


def _m2_by_family() -> dict[str, dict]:
    return {row["family"]: row for row in _read(M2)["rows"]}


def _layered_by_family() -> dict[str, dict[str, dict]]:
    cells: dict[str, dict[str, dict]] = {}
    for cell in _read(LAYERED)["cells"]:
        cells.setdefault(cell["family"], {})[cell["method"]] = cell
    return cells


def _gate_status(scan_row: dict, gate: str) -> str:
    return str((scan_row.get(gate) or {}).get("status", "missing"))


def _hard_gates(
    family: str,
    scan_row: dict,
    m2_row: dict,
    cells: dict[str, dict],
) -> dict[str, bool]:
    static = m2_row.get("static") or {}
    live = m2_row.get("live") or {}
    codes = [cell["code"] for cell in cells.values()]
    return {
        "deriv_gate_pass": _gate_status(scan_row, "deriv") == "pass",
        "dpqr_gate_pass": _gate_status(scan_row, "dpqr") == "pass",
        "fit_gate_pass": _gate_status(scan_row, "rsdev") == "pass",
        "r_consistency_validated": family in _R_CONSISTENCY_EVIDENCE
        and get_family_capability(family).status("r_consistency").value == "validated",
        "golden_fit_fixture": bool(
            static.get("has_deviance")
            and static.get("estimated_coefficient_blocks")
            and static.get("has_convergence")
        ),
        "live_newdata_predictions": bool(live.get("predicted_within_tolerance")),
        "live_json_roundtrip": bool(live)
        and float(live.get("roundtrip_max_abs_drift", float("inf"))) <= 1e-10,
        "live_converged": bool(live.get("converged")),
        "no_route_mismatch": not any(code in {"mismatch", "py_error"} for code in codes),
    }


def _soft_flags(cells: dict[str, dict]) -> dict[str, list[str]]:
    return {
        "flat_ridge_cells": sorted(
            method for method, cell in cells.items() if cell["code"] == "flat_ridge"
        ),
        "r_side_route_errors": sorted(
            method for method, cell in cells.items() if cell["code"] == "r_error"
        ),
    }


def _registry_audit(promoted_rows: dict[str, dict], scan: dict[str, dict]) -> dict:
    """Check each registry claim against the artifacts, not against itself."""

    def measured(name: str, requirement: str) -> bool:
        row = promoted_rows.get(name)
        scan_row = scan.get(name, {})
        if requirement == "production_safe":
            return bool(
                row
                and all(row["hard_gates"].values())
                and row["family"] in _PRODUCTION_SAFE
            )
        if requirement == "rs_fit":
            return _gate_status(scan_row, "rsdev") == "pass" or bool(
                row and row["hard_gates"]["golden_fit_fixture"]
            )
        if requirement == "prediction":
            if row and row["measured"]["golden_has_predicted"]:
                return True
            extra = EXTRA_EVIDENCE_TESTS.get(name)
            return bool(extra and (ROOT / extra).is_file())
        raise KeyError(requirement)

    claims = {
        "_PRODUCTION_SAFE": (sorted(_PRODUCTION_SAFE), "production_safe"),
        "_RS_FIT_EVIDENCE": (sorted(_RS_FIT_EVIDENCE), "rs_fit"),
        "_PREDICTION_EVIDENCE": (sorted(_PREDICTION_EVIDENCE), "prediction"),
    }
    audit = {}
    violations: list[dict] = []
    for set_name, (declared, requirement) in claims.items():
        rows = []
        for name in declared:
            backed = measured(name, requirement)
            rows.append({"family": name, "backed": backed, "requirement": requirement})
            if not backed:
                violations.append(
                    {
                        "set": set_name,
                        "family": name,
                        "requirement": requirement,
                        "problem": "registry claim has no supporting artifact",
                    }
                )
        audit[set_name] = {"declared": rows, "requirement": requirement}
    return {"sets": audit, "violations": violations}


def build() -> dict:
    manifest = _read(GOLDEN_MANIFEST)
    cases = {entry["family"]: case for case, entry in manifest.items()}
    promoted = _promoted()
    scan = _scan_by_family()
    m2 = _m2_by_family()
    layered = _layered_by_family()

    families = []
    for family in promoted:
        m2_row = m2.get(family, {})
        cells = layered.get(family, {})
        hard = _hard_gates(family, scan.get(family, {}), m2_row, cells)
        soft = _soft_flags(cells)
        admitted = family in _PRODUCTION_SAFE
        if admitted:
            status = "admitted"
        elif all(hard.values()):
            status = "eligible"
        else:
            status = "blocked"
        missing = sorted(name for name, ok in hard.items() if not ok)
        actions = []
        if status == "eligible" and not admitted:
            actions.append("add_to_PRODUCTION_SAFE")
            if family not in _RS_FIT_EVIDENCE:
                actions.append("add_to_RS_FIT_EVIDENCE")
            if family not in _PREDICTION_EVIDENCE:
                actions.append("add_to_PREDICTION_EVIDENCE")
        families.append(
            {
                "family": family,
                "golden_case": cases.get(family),
                "status": status,
                "hard_gates": hard,
                "soft_flags": soft,
                "missing_gates": missing,
                "admission_actions": actions,
                "registered": {
                    "production_safe": admitted,
                    "rs_fit": family in _RS_FIT_EVIDENCE,
                    "prediction": family in _PREDICTION_EVIDENCE,
                    "r_consistency": family in _R_CONSISTENCY_EVIDENCE,
                },
                "measured": {
                    "gates": {gate: _gate_status(scan.get(family, {}), gate) for gate in GATES},
                    "golden_has_predicted": bool(
                        (m2_row.get("static") or {}).get("has_predicted")
                    ),
                    "golden_has_newdata": bool(
                        (m2_row.get("static") or {}).get("has_newdata")
                    ),
                    "live_predictions_ok": bool(
                        (m2_row.get("live") or {}).get("predicted_within_tolerance")
                    ),
                    "live_roundtrip_ok": bool(m2_row.get("live"))
                    and float((m2_row.get("live") or {}).get("roundtrip_max_abs_drift", 1.0))
                    <= 1e-10,
                    "layered_codes": {method: cell["code"] for method, cell in cells.items()},
                },
            }
        )

    by_status: dict[str, list[str]] = {}
    for row in families:
        by_status.setdefault(row["status"], []).append(row["family"])
    audit = _registry_audit({row["family"]: row for row in families}, scan)

    admitted = by_status.get("admitted", [])
    eligible = by_status.get("eligible", [])
    blocked = by_status.get("blocked", [])
    counts = {
        "promoted_families": len(families),
        "admitted": len(admitted),
        "eligible_not_admitted": len(eligible),
        "blocked": len(blocked),
        "soft_flagged": sum(
            1
            for row in families
            if row["soft_flags"]["flat_ridge_cells"] or row["soft_flags"]["r_side_route_errors"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M3",
        "target_version": "0.9.0",
        "scope": "production_safe admission evidence for the frozen promoted families",
        "gate_definition": {
            "as_of": "2026-11-15",
            "decision": "D2.240",
            "hard_gates": list(HARD_GATES),
            "soft_flags": list(SOFT_FLAGS),
            "policy": (
                "Hard gates come from committed artifacts (three parity gates, golden fit "
                "fixture, M2 live predictions / round-trip / convergence, layered route "
                "parity). Soft flags are recorded, not blocking: flat_ridge means both "
                "routes reached the same likelihood with a weakly identified parameter, "
                "and an r-side route error is R's own failure. Admission itself is a "
                "separate registry change; this artifact only inventories evidence."
            ),
        },
        "sources": {
            "frozen_families": "docs/development/highfreq-families-frozen.json",
            "parity_scan": "docs/development/rs-parity-scan-2026-11-15.json",
            "m2_assessment": "docs/development/m2-assessment-2026-11-15.json",
            "layered_parity": "docs/development/layered-parity-2026-11-15.json",
            "capability_sets": "omnilss/src/omnilss/family_capabilities.py",
        },
        "families": families,
        "registry_audit": audit,
        "counts": counts,
        "summary": {
            "admitted": admitted,
            "eligible_not_admitted": eligible,
            "blocked": blocked,
        },
        "verdict": {
            "registry_claims_backed": not audit["violations"],
            "admitted_subset_of_eligible": all(
                row["status"] == "admitted" and all(row["hard_gates"].values())
                for row in families
                if row["family"] in _PRODUCTION_SAFE
            ),
            "admission_inventory_complete": all(
                row["status"] in {"admitted", "eligible", "blocked"} for row in families
            ),
            "notes": (
                f"This is an inventory, not an admission decision. Of {counts['promoted_families']} "
                f"promoted families: {counts['admitted']} admitted, "
                f"{counts['eligible_not_admitted']} eligible but unregistered, "
                f"{counts['blocked']} blocked. Soft flags (flat_ridge / r_side_route_error) "
                "are carried in the family rows and must travel with any admission record. "
                "M3 also requires P3-1/P3-2/P3-3, which this artifact does not cover."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "milestone": artifact["milestone"],
        "gate_definition": artifact["gate_definition"],
        "counts": artifact["counts"],
        "summary": artifact["summary"],
        "registry_audit": artifact["registry_audit"],
        "verdict": artifact["verdict"],
        "families": artifact["families"],
    }


def _render_markdown(a: dict) -> str:
    c = a["counts"]
    gate = a["gate_definition"]
    lines = [
        "# M3 production_safe admission evidence inventory",
        "",
        f"- scope: {a['scope']}",
        f"- promoted families: {c['promoted_families']}",
        f"- admitted: {c['admitted']} ({', '.join(a['summary']['admitted'])})",
        f"- eligible but not admitted: {c['eligible_not_admitted']}",
        f"- blocked: {c['blocked']}",
        f"- hard gates: {', '.join(gate['hard_gates'])}",
        f"- soft flags: {', '.join(gate['soft_flags'])}",
        "",
        "| family | status | missing gates | flat_ridge | r-side error |",
        "|---|---|---|---|---|",
    ]
    for row in a["families"]:
        flat = ", ".join(row["soft_flags"]["flat_ridge_cells"]) or "-"
        rerr = ", ".join(row["soft_flags"]["r_side_route_errors"]) or "-"
        missing = ", ".join(row["missing_gates"]) or "-"
        lines.append(
            f"| {row['family']} | {row['status']} | {missing} | {flat} | {rerr} |"
        )
    lines += ["", "## Registry claim audit", "", "| set | family | backed | requirement |", "|---|---|---|---|"]
    for set_name, block in a["registry_audit"]["sets"].items():
        for item in block["declared"]:
            lines.append(
                f"| {set_name} | {item['family']} | {item['backed']} | {item['requirement']} |"
            )
    violations = a["registry_audit"]["violations"]
    lines += ["", f"- registry claim violations: {len(violations)}"]
    for violation in violations:
        lines.append(
            f"  - {violation['set']} / {violation['family']}: {violation['problem']}"
        )
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
    artifact = _read(path)
    for key in ("schema_version", "milestone", "gate_definition", "families",
                "registry_audit", "counts", "summary", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build()
    if _stable(artifact) != _stable(current):
        for key in ("counts", "summary", "registry_audit", "verdict", "families"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drifted vs the current tree")
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
        print(f"M3 admission evidence check: {args.path}")
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
    counts = artifact["counts"]
    log(
        f"promoted={counts['promoted_families']} admitted={counts['admitted']} "
        f"eligible={counts['eligible_not_admitted']} blocked={counts['blocked']} "
        f"soft_flagged={counts['soft_flagged']}"
    )
    log(f"registry_claims_backed={artifact['verdict']['registry_claims_backed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
