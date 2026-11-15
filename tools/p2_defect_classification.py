# SPDX-License-Identifier: GPL-3.0-or-later
"""P2 defect classification (D2.236).

DEVELOPMENT_PLAN P2 requires every non-pass from the three parity gates to be
*classified* -- R density defect / divergence / floor / early stop / we are
better -- rather than hidden behind a skip.

This tool consumes the machine-readable scan report
(tools/rs_family_parity_scan.py --json ...) and assigns each warn/skip/fail a
category, using the scanner's own registered tables for the rsdev adjudications
and the documented detail strings for the dpqr/deriv skips.  Any entry it
cannot place stays in 'unclassified' and makes the verdict false -- a silent
"skip" is exactly what P2 forbids.

Usage::

    python tools/p2_defect_classification.py --scan docs/development/rs-parity-scan-2026-11-15.json \
        --write docs/development/p2-defect-classification-2026-11-15.json
    python tools/p2_defect_classification.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_SCAN = ROOT / "docs" / "development" / "rs-parity-scan-2026-11-15.json"
DEFAULT_OUT = ROOT / "docs" / "development" / "p2-defect-classification-2026-11-15.json"

GATES = ("deriv", "dpqr", "rsdev")

#: P2's "挂账缺陷" list, used to give a human-facing reason where the register
#: tables only carry the adjudication, not the mechanism.
KNOWN_MECHANISMS = {
    "BCTO": "deriv: BCT tau cross-block needs a custom JVP",
    "SHASH": "density integration defect (SHASH non-o)",
    "NET": "deriv/dpqr: fixed nu/tau clamp",
    "PARETO1o": "score has a ghost 1/sigma term at the support edge",
    "JSU": "fit trajectory stops at a poor stationary point",
    "LNO": "intercept-only fit is an open failure",
    "EXGAUS": "q residual is R's uniroot default tolerance",
}

_REGISTER_TABLES = (
    ("registered_r_early_stop", "_RSDEV_R_STOPPED_EARLY"),
    ("registered_r_diverged", "_RSDEV_R_DIVERGED"),
    ("registered_path_divergent", "_RSDEV_PATH_DIVERGENT"),
    ("registered_r_density_recycled", "_RSDEV_R_DENSITY_RECYCLED"),
    ("registered_r_reference_invalid", "_RSDEV_R_REFERENCE_INVALID"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_table(family: str) -> tuple[str, str] | None:
    for code, name in _REGISTER_TABLES:
        table = getattr(scan, name)
        if family in table:
            return code, table[family].get("note", "")
    return None


def _classify(gate: str, family: str, info: dict) -> dict:
    status = info.get("status", "skip")
    detail = (info.get("detail") or "").strip()
    entry = {"family": family, "gate": gate, "status": status, "detail": detail[:300]}
    if status == "pass":
        entry["code"] = "pass"
        return entry

    # Registered tables describe adjudicated *warn* outcomes.  A skip/fail
    # carries its own cause in the detail (e.g. "R fit error"), which must win:
    # SEP is registered as an early stop but this scan's rsdev entry is an R
    # fit error, and labelling it "early stop" would hide the real cause.
    registered = _registered_table(family) if gate == "rsdev" and status == "warn" else None
    if registered is not None:
        entry["code"] = registered[0]
        entry["evidence"] = registered[1][:600]
        return entry

    lowered = detail.lower()
    if gate == "rsdev":
        # The warn detail is the scanner's own disposition, so classify from it
        # and use table membership only to name the registered outcome.
        if lowered.startswith("r fit error"):
            entry["code"] = "r_fit_error"
        elif lowered.startswith("python raised"):
            entry["code"] = "python_fit_error"
        elif lowered.startswith("python deviance non-finite"):
            entry["code"] = "python_non_finite"
        elif lowered.startswith("no fit job"):
            entry["code"] = "skip_no_fit_job"
        elif "stopped early" in lowered or "is below r" in lowered:
            entry["code"] = (
                "registered_r_early_stop"
                if family in scan._RSDEV_R_STOPPED_EARLY
                else "r_below_halt"
            )
        elif "fit diverged" in lowered:
            entry["code"] = (
                "registered_r_diverged"
                if family in scan._RSDEV_R_DIVERGED
                else "r_diverged"
            )
        elif "did not converge" in lowered:
            entry["code"] = "r_not_converged"
        elif "numerically invalid" in lowered:
            entry["code"] = (
                "registered_r_reference_invalid"
                if family in scan._RSDEV_R_REFERENCE_INVALID
                else "r_reference_invalid"
            )
        elif "not elementwise" in lowered:
            entry["code"] = (
                "registered_r_density_recycled"
                if family in scan._RSDEV_R_DENSITY_RECYCLED
                else "r_density_not_elementwise"
            )
        elif "path diverge" in lowered or "trajectory" in lowered:
            entry["code"] = "registered_path_divergent"
        else:
            entry["code"] = "unclassified"
    elif gate == "dpqr":
        if "no d/p/q counterpart in r" in lowered:
            entry["code"] = "no_r_reference"
        elif lowered.startswith("no parameter metadata"):
            entry["code"] = "skip_no_parameter_metadata"
        elif lowered.startswith("resolve failed"):
            entry["code"] = "skip_resolve_failed"
        else:
            entry["code"] = "unclassified"
    else:  # deriv
        if "timeout" in lowered:
            entry["code"] = "probe_timeout"
        elif lowered.startswith("no score_functions or density"):
            entry["code"] = "skip_no_score_functions"
        elif lowered.startswith("resolve failed"):
            entry["code"] = "skip_resolve_failed"
        elif lowered.startswith("calibration failed"):
            entry["code"] = "skip_calibration_failed"
        elif lowered.startswith("no support grid"):
            entry["code"] = "skip_no_support_grid"
        else:
            entry["code"] = "unclassified"

    mechanism = KNOWN_MECHANISMS.get(family)
    if mechanism:
        entry["mechanism"] = mechanism
    return entry


def build(scan_path: Path) -> dict:
    rows = json.loads(scan_path.read_text(encoding="utf-8"))
    entries = []
    for row in rows:
        family = row["family"]
        for gate in GATES:
            info = row.get(gate)
            if info is None:
                entries.append(
                    {"family": family, "gate": gate, "status": "missing", "detail": "",
                     "code": "gate_not_run"}
                )
                continue
            entries.append(_classify(gate, family, info))

    counts = {gate: Counter() for gate in GATES}
    categories = {gate: {} for gate in GATES}
    for entry in entries:
        counts[entry["gate"]][entry["status"]] += 1
        if entry["code"] != "pass":
            categories[entry["gate"]].setdefault(entry["code"], []).append(entry["family"])

    unclassified = [e for e in entries if e["code"] in {"unclassified", "gate_not_run"}]
    status_counts = {
        gate: {status: counts[gate].get(status, 0) for status in sorted(counts[gate])}
        for gate in GATES
    }
    non_pass = {
        gate: sum(counts[gate][s] for s in counts[gate] if s != "pass")
        for gate in GATES
    }

    try:
        rel_scan = scan_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel_scan = scan_path.as_posix()

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "scan": rel_scan,
            "scan_sha256": _sha256(scan_path),
            "families": len(rows),
            "gates": list(GATES),
            "command": "python tools/rs_family_parity_scan.py --json <scan> --quiet",
        },
        "counts": {"status": status_counts, "non_pass": non_pass},
        "categories": {gate: {code: sorted(fams) for code, fams in categories[gate].items()}
                       for gate in GATES},
        "entries": sorted(entries, key=lambda e: (e["gate"], e["family"])),
        "unclassified": [
            {"family": e["family"], "gate": e["gate"], "status": e["status"],
             "detail": e["detail"]}
            for e in unclassified
        ],
        "verdict": {
            "all_failures_classified": not unclassified,
            "non_pass_total": sum(non_pass.values()),
            "notes": (
                "Every non-pass carries a category. 'no_r_reference' means R's own "
                "package cannot produce the reference; 'registered_*' are the "
                "scanner's adjudicated R-side outcomes (early stop / divergence / "
                "path / density recycling); flat-ridge families are those whose "
                "deviance matched but the optimum did not."
            ),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "counts": artifact["counts"],
        "categories": artifact["categories"],
        "unclassified": artifact["unclassified"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    lines = [
        "# P2 defect classification",
        "",
        f"- scan: {a['source']['scan']} (sha256 {a['source']['scan_sha256'][:12]}..., "
        f"{a['source']['families']} families)",
        f"- non-pass total: {a['verdict']['non_pass_total']}",
        f"- all classified: {a['verdict']['all_failures_classified']}",
        "",
        "| gate | status counts |",
        "|---|---|",
    ]
    for gate, counts in a["counts"]["status"].items():
        rendered = ", ".join(f"{k}={v}" for k, v in counts.items())
        lines.append(f"| {gate} | {rendered} |")
    lines += ["", "## Categories", ""]
    for gate, cats in a["categories"].items():
        lines.append(f"### {gate}")
        lines.append("")
        if not cats:
            lines.append("(no non-pass entries)")
            lines.append("")
            continue
        lines += ["| code | count | families |", "|---|---|---|"]
        for code, families in sorted(cats.items()):
            lines.append(f"| {code} | {len(families)} | {', '.join(families)} |")
        lines.append("")
    if a["unclassified"]:
        lines += ["## Unclassified (must be zero for the verdict)", ""]
        for item in a["unclassified"]:
            lines.append(
                f"- {item['family']} / {item['gate']} / {item['status']}: {item['detail']}"
            )
        lines.append("")
    lines.append(a["verdict"]["notes"])
    lines.append("")
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
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_version", "source", "counts", "categories", "entries",
                "unclassified", "verdict"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems
    scan_path = ROOT / artifact["source"]["scan"]
    if not scan_path.is_file():
        return False, [f"source scan missing: {scan_path}"]
    if _sha256(scan_path) != artifact["source"]["scan_sha256"]:
        return False, ["source scan sha256 changed; regenerate the classification"]
    current = build(scan_path)
    if _stable(artifact) != _stable(current):
        problems.append("classification drifted vs the recorded scan")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_artifact(args.path)
        print(f"P2 classification check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for p in problems:
            print(f"  - {p}")
        print("result: FAIL")
        return 1

    artifact = build(args.scan)
    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print
    if args.write is not None:
        write_artifact(artifact, args.write)
        log(f"wrote {args.write}")
        log(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    log(f"non_pass={artifact['verdict']['non_pass_total']} "
        f"all_classified={artifact['verdict']['all_failures_classified']}")
    for gate, counts in artifact["counts"]["status"].items():
        log(f"  {gate}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
