# SPDX-License-Identifier: GPL-3.0-or-later
"""M4 / P4 scope inventory (D2.246).

DEVELOPMENT_PLAN P4 orders the surrounding R packages by user value:

    1. gamlss.cens + gamlss.tr          (censoring / truncation)
    2. gamlss.mx + gamlss.add           (finite mixtures / additive terms)
    3. gamlss.spatial + gamlss.foreach + gamlss.prepdata
    4. gamlss.ggplots + gamlss.data     (presentation, lowest priority)
    out of scope: gamlss2

This tool turns that ordering into a checked inventory: per package it records
the surface gap, the evidence level, the dedicated test modules that exist, and
a concrete next action.  It is fail-closed on scope: a package present in the
coverage artifact but missing from this table is an error, so new surrounding
surface cannot appear without being classified.

Usage::

    python tools/m4_scope_inventory.py --write docs/development/m4-scope-2026-11-15.json
    python tools/m4_scope_inventory.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "docs" / "development" / "r-symbol-coverage.json"
DEFAULT_OUT = ROOT / "docs" / "development" / "m4-scope-2026-11-15.json"
TESTS = ROOT / "omnilss" / "tests"

SCHEMA_VERSION = 1

#: P4 order from the plan, with the evidence modules that already exist.
PACKAGES: tuple[dict[str, object], ...] = (
    {
        "package": "gamlss.cens",
        "priority": 1,
        "rationale": "censoring is a modelling need, not a presentation layer",
        "evidence_tests": (
            "test_gamlss_cens.py",
            "test_censored_fit.py",
            "test_censored_predict.py",
            "test_censored_resid.py",
            "test_censored_information_criteria.py",
        ),
    },
    {
        "package": "gamlss.tr",
        "priority": 1,
        "rationale": "truncation pairs with censoring in the same user workflow",
        "evidence_tests": (
            "test_bct_truncation_parity.py",
            "test_truncated_initial_values.py",
        ),
    },
    {
        "package": "gamlss.mx",
        "priority": 2,
        "order": 1,
        "rationale": "finite mixtures are the next modelling capability",
        "evidence_tests": ("test_mixtures.py", "test_mixtures_accessors.py"),
    },
    {
        "package": "gamlss.add",
        "priority": 2,
        "order": 2,
        "rationale": "additive terms (ga/ba/nn/fk/tr) extend the formula surface",
        "evidence_tests": ("test_add_controls.py",),
    },
    {
        "package": "gamlss.spatial",
        "priority": 3,
        "order": 1,
        "rationale": "spatial smoothers build on the smooth-term framework",
        "evidence_tests": ("test_spatial.py",),
    },
    {
        "package": "gamlss.foreach",
        "priority": 3,
        "order": 2,
        "rationale": "parallel/bootstrap helpers are infrastructure",
        "evidence_tests": ("test_gamlss_foreach_evidence.py",),
    },
    {
        "package": "gamlss.prepdata",
        "priority": 3,
        "order": 3,
        "rationale": "data diagnostics support modelling but are not a fit path",
        "evidence_tests": ("test_gamlss_prepdata_evidence.py",),
    },
    {
        "package": "gamlss.ggplots",
        "priority": 4,
        "order": 1,
        "rationale": "presentation layer; pixel parity is explicitly not required",
        "evidence_tests": ("test_gamlss_ggplots_evidence.py",),
    },
    {
        "package": "gamlss.data",
        "priority": 4,
        "order": 2,
        "rationale": "datasets only; no function surface to port",
        "evidence_tests": (),
    },
)

OUT_OF_SCOPE = {
    "gamlss": "core package; its surface is the M1/M3 coverage target, not P4",
    "gamlss.dist": "core distribution package; validated by the golden/parity gates",
    "gamlss2": "separate specification / version / evidence track; must not block 1.0",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_rows() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for symbol in _read(COVERAGE)["symbols"]:
        package = symbol.get("r_package")
        if package:
            rows.setdefault(package, []).append(symbol)
    return rows


def build() -> dict:
    coverage = _read(COVERAGE)
    rows = _package_rows()
    known = {item["package"] for item in PACKAGES} | set(OUT_OF_SCOPE)
    in_coverage = {
        name for name, meta in coverage["stats"]["per_package"].items()
        if meta.get("in_scope") or name in OUT_OF_SCOPE
    }
    unclassified = sorted(name for name in in_coverage if name not in known)
    if unclassified:
        raise SystemExit(
            "surrounding packages in the coverage artifact but not classified here: "
            + ", ".join(unclassified)
        )

    entries = []
    for item in PACKAGES:
        name = str(item["package"])
        symbols = rows.get(name, [])
        surface = Counter(s["layers"]["surface"] for s in symbols)
        parity = Counter(s["layers"]["parity"] for s in symbols)
        missing = sorted(
            s["r_symbol"] for s in symbols if s["layers"]["surface"] == "missing"
        )
        tests = list(item["evidence_tests"])
        missing_tests = [test for test in tests if not (TESTS / test).is_file()]
        if missing_tests:
            raise SystemExit(f"{name}: evidence modules not found: {missing_tests}")
        per_package = coverage["stats"]["per_package"].get(name, {})
        if missing:
            action = f"implement {len(missing)} missing symbols"
        elif parity.get("tested") and not any(
            key in parity for key in ("golden:validated", "r_consistency_test")
        ):
            action = f"upgrade evidence for {parity.get('tested', 0)} tested symbols to golden"
        elif tests:
            action = "hold: surface complete and evidence modules exist"
        else:
            action = "hold: surface complete, no dedicated evidence module"
        if not symbols and not int(per_package.get("exported", 0)):
            status = "no_function_surface"
        elif missing:
            status = "surface_gap"
        else:
            status = "surface_complete"
        entries.append(
            {
                "package": name,
                "priority": int(item["priority"]),
                "order": int(item.get("order", 99)),
                "rationale": item["rationale"],
                "status": status,
                "exported": int(per_package.get("exported", len(symbols))),
                "tracked_symbols": len(symbols),
                "surface": dict(sorted(surface.items())),
                "parity": dict(sorted(parity.items())),
                "missing_symbols": missing,
                "evidence_tests": tests,
                "next_action": action,
            }
        )
    entries.sort(key=lambda row: (row["priority"], row["order"], row["package"]))

    by_priority: dict[str, list[str]] = {}
    for row in entries:
        by_priority.setdefault(str(row["priority"]), []).append(row["package"])

    counts = {
        "packages": len(entries),
        "out_of_scope": len(OUT_OF_SCOPE),
        "surface_complete": sum(1 for row in entries if row["status"] == "surface_complete"),
        "surface_gap": sum(1 for row in entries if row["status"] == "surface_gap"),
        "no_function_surface": sum(
            1 for row in entries if row["status"] == "no_function_surface"
        ),
        "missing_symbols": sum(len(row["missing_symbols"]) for row in entries),
        "packages_without_evidence_tests": sum(
            1
            for row in entries
            if not row["evidence_tests"] and row["status"] != "no_function_surface"
        ),
    }
    first_gap = next((row for row in entries if row["missing_symbols"]), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M4",
        "task": "P4 scope",
        "decision": "D2.246",
        "sources": {
            "coverage": "docs/development/r-symbol-coverage.json",
            "plan_section": "DEVELOPMENT_PLAN section P4",
        },
        "priority_groups": by_priority,
        "out_of_scope": OUT_OF_SCOPE,
        "packages": entries,
        "counts": counts,
        "verdict": {
            "inventory_complete": not unclassified,
            "first_work_item": (
                None
                if first_gap is None
                else {
                    "package": first_gap["package"],
                    "priority": first_gap["priority"],
                    "missing": len(first_gap["missing_symbols"]),
                    "next_action": first_gap["next_action"],
                }
            ),
            "notes": (
                "This is a scope inventory, not an M4 gate. gamlss.cens and gamlss.tr "
                "already have full surface plus dedicated evidence modules; gamlss.mx is "
                "the first package with a surface gap, so it is the recommended next "
                "batch. Packages without a dedicated test module are flagged rather "
                "than assumed covered."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "milestone": artifact["milestone"],
        "task": artifact["task"],
        "priority_groups": artifact["priority_groups"],
        "out_of_scope": artifact["out_of_scope"],
        "packages": artifact["packages"],
        "counts": artifact["counts"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    lines = [
        "# M4 / P4 scope inventory",
        "",
        f"- decision: {a['decision']}",
        f"- packages: {a['counts']['packages']} in scope, "
        f"{a['counts']['out_of_scope']} out of scope",
        f"- surface complete: {a['counts']['surface_complete']}, "
        f"with gap: {a['counts']['surface_gap']} "
        f"({a['counts']['missing_symbols']} missing symbols)",
        f"- packages without dedicated evidence tests: "
        f"{a['counts']['packages_without_evidence_tests']}",
    ]
    first = a["verdict"]["first_work_item"]
    if first:
        lines.append(
            f"- recommended next batch: **{first['package']}** (priority "
            f"{first['priority']}, {first['missing']} missing)"
        )
    lines += ["", "| priority | package | status | exported | missing | evidence tests | next action |", "|---|---|---|---|---|---|---|"]
    for row in a["packages"]:
        lines.append(
            f"| {row['priority']} | {row['package']} | {row['status']} | "
            f"{row['exported']} | {len(row['missing_symbols'])} | "
            f"{len(row['evidence_tests'])} | {row['next_action']} |"
        )
    lines += ["", "## Missing symbols by package", ""]
    for row in a["packages"]:
        if not row["missing_symbols"]:
            continue
        lines.append(f"### {row['package']} ({len(row['missing_symbols'])})")
        lines.append("")
        lines.append(", ".join(row["missing_symbols"]))
        lines.append("")
    lines += ["## Out of scope", ""]
    for name, reason in a["out_of_scope"].items():
        lines.append(f"- {name}: {reason}")
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
    for key in ("schema_version", "milestone", "priority_groups", "packages", "counts", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    current = build()
    if _stable(artifact) != _stable(current):
        for key in ("priority_groups", "packages", "counts", "verdict", "out_of_scope"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drifted vs the current coverage artifact")
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
        print(f"M4 scope check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for problem in problems:
            print(f"  - {problem}")
        print("result: FAIL")
        return 1

    artifact = build()
    log = lambda *a: print(*a, file=sys.stderr) if args.json else print(*a)
    if args.write is not None:
        write_artifact(artifact, args.write)
        log(f"wrote {args.write}")
        log(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    counts = artifact["counts"]
    log(
        f"packages={counts['packages']} complete={counts['surface_complete']} "
        f"gaps={counts['surface_gap']} missing={counts['missing_symbols']}"
    )
    first = artifact["verdict"]["first_work_item"]
    log(f"first_work_item={first['package'] if first else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
