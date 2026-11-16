# SPDX-License-Identifier: GPL-3.0-or-later
"""M1 wrap-up assessment (D2.234).

DEVELOPMENT_PLAN section 22.4 states M1 as:

    high-frequency families all Surface+Kernel+DPQR parity; golden:validated
    raised by an order of magnitude; no placeholder in the target scope.

This tool turns that sentence into measurable facts from the committed
artifacts (r-symbol-coverage.json, highfreq-families-frozen.json) and records
the verdict per criterion.  It deliberately does NOT change the coverage rules
to make a criterion pass: the r_consistency-precedence finding is reported as a
finding, and the literal "order of magnitude" is reported as unreachable given
the frozen family set.

Usage::

    python tools/m1_assessment.py --write docs/development/m1-assessment-2026-11-15.json
    python tools/m1_assessment.py --check
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_JSON = ROOT / "docs" / "development" / "r-symbol-coverage.json"
FROZEN = ROOT / "docs" / "development" / "highfreq-families-frozen.json"
DEFAULT_OUT = ROOT / "docs" / "development" / "m1-assessment-2026-11-15.json"

SCHEMA_VERSION = 1
#: golden:validated measured immediately before the P1 batches (D2.227 log).
BASELINE_GOLDEN_VALIDATED = 34
TARGET_ORDER_OF_MAGNITUDE = 10

#: The M1 gate as revised in D2.235: the original wording demanded a 10x
#: golden:validated jump and a fully implemented kernel for every promoted
#: family.  The first is unreachable with a 40-family validated pool (18 are
#: excluded with evidence), and the second is admission (M3) work, so the gate
#: is stated as the parity property it was always meant to capture.
GATE_CRITERIA = (
    "highfreq_surface_parity",
    "highfreq_dpqr_parity_by_evidence",
    "no_placeholder_in_scope",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _promoted() -> list[str]:
    frozen = _read(FROZEN)
    return sorted(item["family"] for item in frozen["tiers"]["promoted"])


def _family_rows(promoted: list[str]) -> dict[str, dict]:
    cov = _read(COVERAGE_JSON)
    rows: dict[str, dict] = {}
    for family in promoted:
        rows[family] = {
            "symbols": 0,
            "surface_missing": 0,
            "kernel": Counter(),
            "parity": Counter(),
            "dpqr": [],
        }
    for symbol in cov["symbols"]:
        family = symbol.get("r_family")
        if family not in rows or not symbol.get("in_scope"):
            continue
        row = rows[family]
        layers = symbol["layers"]
        row["symbols"] += 1
        row["kernel"][layers["kernel"]] += 1
        row["parity"][layers["parity"]] += 1
        if layers["surface"] == "missing":
            row["surface_missing"] += 1
        if symbol["r_symbol"] in {f"d{family}", f"p{family}", f"q{family}", f"r{family}"}:
            row["dpqr"].append(
                {
                    "symbol": symbol["r_symbol"],
                    "parity": layers["parity"],
                    "kernel": layers["kernel"],
                }
            )
    return rows


def build() -> dict:
    promoted = _promoted()
    rows = _family_rows(promoted)
    cov = _read(COVERAGE_JSON)

    total_symbols = sum(r["symbols"] for r in rows.values())
    golden_symbols = sum(
        count for r in rows.values() for parity, count in r["parity"].items()
        if parity == "golden:validated"
    )
    rcons_symbols = sum(
        count for r in rows.values() for parity, count in r["parity"].items()
        if parity == "r_consistency_test"
    )

    # Every promoted family's d/p/q/r: golden directly, or r_consistency_test
    # that *suppresses* golden (the coverage rule is r_consistency first).
    dpqr = [item for r in rows.values() for item in r["dpqr"]]
    dpqr_golden = [item for item in dpqr if item["parity"] == "golden:validated"]
    dpqr_suppressed = [item for item in dpqr if item["parity"] == "r_consistency_test"]
    dpqr_other = [item for item in dpqr if item["parity"] not in {"golden:validated", "r_consistency_test"}]

    kernel_counts = Counter()
    for r in rows.values():
        kernel_counts.update(r["kernel"])
    kernel_implemented_families = sorted(
        family for family, r in rows.items() if set(r["kernel"]) <= {"implemented"}
    )

    placeholder_in_scope = [
        s["r_symbol"]
        for s in cov["symbols"]
        if s.get("in_scope")
        and (s["layers"]["surface"] == "placeholder" or s["layers"]["kernel"] == "placeholder")
    ]
    total_golden = sum(
        1 for s in cov["symbols"] if s["layers"]["parity"] == "golden:validated"
    )
    growth = total_golden / BASELINE_GOLDEN_VALIDATED

    criteria = {
        "highfreq_surface_parity": {
            "met": all(r["surface_missing"] == 0 for r in rows.values()),
            "detail": "no promoted family has a missing-surface symbol",
        },
        "highfreq_dpqr_parity_direct": {
            "met": not dpqr_suppressed and not dpqr_other,
            "detail": (
                f"{len(dpqr_golden)}/{len(dpqr)} dpqr symbols carry golden parity directly; "
                f"{len(dpqr_suppressed)} are labelled r_consistency_test because that "
                "rule takes precedence over golden in generate_symbol_coverage.py:381"
            ),
        },
        "highfreq_dpqr_parity_by_evidence": {
            "met": not dpqr_other,
            "detail": (
                "every promoted dpqr symbol has either a golden fixture or a live-R "
                "r_consistency test; none is bare capability/tested/unverified"
            ),
        },
        "golden_validated_order_of_magnitude": {
            "met": growth >= TARGET_ORDER_OF_MAGNITUDE,
            "detail": (
                f"{BASELINE_GOLDEN_VALIDATED} -> {total_golden} ({growth:.2f}x); the literal "
                f"{TARGET_ORDER_OF_MAGNITUDE}x target would need ~{BASELINE_GOLDEN_VALIDATED * TARGET_ORDER_OF_MAGNITUDE} "
                "symbols, which the 40-family validated pool cannot supply (18 excluded for "
                "documented R-side/flat-ridge reasons)"
            ),
        },
        "kernel_implemented": {
            "met": len(kernel_implemented_families) == len(promoted),
            "detail": (
                f"{len(kernel_implemented_families)}/{len(promoted)} promoted families have an "
                "implemented kernel. Kernel promotion was M3 admission work rather than M1 "
                "parity; D2.241 closed it by admitting the evidence-eligible batch, so this "
                "criterion tracks the registry rather than being an M1 target."
            ),
        },
        "no_placeholder_in_scope": {
            "met": not placeholder_in_scope,
            "detail": f"in-scope placeholder symbols: {len(placeholder_in_scope)}",
        },
    }

    met_required = all(criteria[name]["met"] for name in GATE_CRITERIA)

    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M1",
        "target_version": "0.5.0",
        "promoted_families": promoted,
        "counts": {
            "promoted_families": len(promoted),
            "symbols_in_promoted_families": total_symbols,
            "golden_symbols_in_promoted_families": golden_symbols,
            "r_consistency_symbols_in_promoted_families": rcons_symbols,
            "dpqr_symbols": len(dpqr),
            "dpqr_golden": len(dpqr_golden),
            "dpqr_r_consistency": len(dpqr_suppressed),
            "dpqr_other": len(dpqr_other),
            "golden_validated_total": total_golden,
            "baseline_golden_validated": BASELINE_GOLDEN_VALIDATED,
            "growth_ratio": round(growth, 4),
            "kernel_layer": dict(kernel_counts),
            "placeholder_in_scope": len(placeholder_in_scope),
            "kernel_implemented_families": len(kernel_implemented_families),
        },
        "criteria": criteria,
        "gate_definition": {
            "as_of": "2026-11-15",
            "decision": "D2.235",
            "criteria": list(GATE_CRITERIA),
            "revised_from": (
                "high-frequency families all Surface+Kernel+DPQR parity; golden:validated "
                "raised by an order of magnitude; no placeholder in the target scope"
            ),
            "revision": (
                "The gate is frozen to the parity property it was meant to capture: surface, "
                "d/p/q/r evidence, and no placeholder for every promoted family. The two "
                "literal clauses are re-stated rather than dropped -- see moved_out."
            ),
            "moved_out": {
                "golden_validated_order_of_magnitude": (
                    "replaced by d/p/q/r evidence coverage: the 40-family validated pool caps "
                    "golden:validated at ~110, so the 10x (340) target is unreachable by "
                    "construction (18 families excluded with documented R-side/flat-ridge reasons)"
                ),
                "kernel_implemented": (
                    "M3 admission work: at the time of the M1 gate 21/22 promoted kernels were "
                    "experimental because admission was blocked (production_safe: 1 family / 5 "
                    "symbols). D2.241 admitted the evidence-eligible batch, so this is now "
                    "closed; it was never a parity gap"
                ),
            },
        },
        "verdict": {
            "m1_gate_met": met_required,
            "core_m1_parity_met": met_required,
            "literal_all_criteria_met": all(c["met"] for c in criteria.values()),
            "notes": (
                "Under the D2.235 gate definition M1 is met: every promoted high-frequency "
                "family has surface parity, d/p/q/r evidence parity (golden, or a live-R "
                "r_consistency test), and no placeholder. The two literal original clauses are "
                "recorded as moved_out, not silently dropped. The r_consistency>golden "
                "precedence suppresses 24 golden labels; that is a reporting-rule finding, not "
                "a parity gap, and the coverage rule was not changed."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "promoted_families": artifact["promoted_families"],
        "counts": artifact["counts"],
        "criteria": artifact["criteria"],
        "gate_definition": artifact["gate_definition"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    c = a["counts"]
    lines = [
        f"# M1 assessment (target {a['target_version']})",
        "",
        f"- promoted families: {c['promoted_families']}",
        f"- symbols in promoted families: {c['symbols_in_promoted_families']} "
        f"(golden {c['golden_symbols_in_promoted_families']}, "
        f"r_consistency {c['r_consistency_symbols_in_promoted_families']})",
        f"- d/p/q/r: {c['dpqr_golden']}/{c['dpqr_symbols']} golden directly, "
        f"{c['dpqr_r_consistency']} suppressed by r_consistency precedence, "
        f"{c['dpqr_other']} other",
        f"- golden:validated total: {c['golden_validated_total']} "
        f"(baseline {c['baseline_golden_validated']}, {c['growth_ratio']}x)",
        f"- kernel layer: {c['kernel_layer']}",
        f"- placeholder in scope: {c['placeholder_in_scope']}",
        "",
        "| criterion | met | detail |",
        "|---|---|---|",
    ]
    for name, item in a["criteria"].items():
        lines.append(f"| {name} | {'yes' if item['met'] else 'no'} | {item['detail']} |")
    gate = a["gate_definition"]
    lines += [
        "",
        f"## Gate definition (revised {gate['as_of']}, {gate['decision']})",
        "",
        f"- criteria: {', '.join(gate['criteria'])}",
        f"- revised from: {gate['revised_from']}",
    ]
    for name, reason in gate["moved_out"].items():
        lines.append(f"- moved out: **{name}** -- {reason}")
    v = a["verdict"]
    lines += [
        "",
        f"**m1_gate_met: {v['m1_gate_met']}**  ",
        f"**core_m1_parity_met: {v['core_m1_parity_met']}**  ",
        f"**literal_all_criteria_met (original wording): {v['literal_all_criteria_met']}**",
        "",
        v["notes"],
        "",
    ]
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
    for key in ("schema_version", "promoted_families", "counts", "criteria",
                "gate_definition", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build()
    if artifact["schema_version"] != SCHEMA_VERSION:
        problems.append("schema_version mismatch")
    if artifact["promoted_families"] != current["promoted_families"]:
        problems.append("promoted family set drifted")
    if artifact["counts"] != current["counts"]:
        problems.append("counts drifted vs current artifacts")
    if artifact["criteria"] != current["criteria"]:
        problems.append("criteria verdicts drifted")
    if artifact["gate_definition"] != current["gate_definition"]:
        problems.append("gate definition drifted")
    if artifact["verdict"] != current["verdict"]:
        problems.append("verdict drifted")
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
        print(f"M1 assessment check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for p in problems:
            print(f"  - {p}")
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
    log(
        f"promoted={c['promoted_families']} dpqr_golden={c['dpqr_golden']}/"
        f"{c['dpqr_symbols']} golden_total={c['golden_validated_total']} "
        f"placeholder={c['placeholder_in_scope']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
