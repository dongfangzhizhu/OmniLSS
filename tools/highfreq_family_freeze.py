# SPDX-License-Identifier: GPL-3.0-or-later
"""P1-1: freeze the high-frequency family list (D2.230).

DEVELOPMENT_PLAN P1-1 asks for a frozen list of high-frequency families
starting from the 39 r_consistency-validated families
(omnilss.family_capabilities._R_CONSISTENCY_EVIDENCE).  A "frozen" list is only
worth anything if it is exhaustive and mutually exclusive, so this tool is
fail-closed:

  * every validated family must fall into exactly one tier
    (promoted / pending / excluded);
  * "promoted" must equal the families that actually have a golden case in
    omnilss/tests/fixtures/r_golden/regenerated/manifest.json;
  * an unknown validated family (added later) is an error, not a silent
    "pending" -- you must classify it.

Exclusion reasons are recorded here so the decision survives with the code,
not only in a log.  --check validates a committed artifact against the current
tree (drift in the golden set or the validated set fails).

Usage::

    python tools/highfreq_family_freeze.py --write docs/development/highfreq-families-frozen.json
    python tools/highfreq_family_freeze.py --check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

GOLDEN_MANIFEST = (
    ROOT / "omnilss" / "tests" / "fixtures" / "r_golden" / "regenerated" / "manifest.json"
)
DEFAULT_OUT = ROOT / "docs" / "development" / "highfreq-families-frozen.json"
SCHEMA_VERSION = 1

#: Families probed and deliberately NOT promoted, with the reason.  Keys must
#: be re-validated when the probe conditions change (e.g. a longer n.cyc).
EXCLUSIONS: dict[str, dict[str, str]] = {
    "TF": {
        "code": "flat_nu_tau_ridge",
        "note": "n.cyc=100 leaves a large coefficient gap from a different local optimum; deviance matched.",
    },
    "SHASH": {
        "code": "flat_nu_tau_ridge",
        "note": "4-parameter flat ridge; coefficient gap ~1e-1 while deviance matched.",
    },
    "SN1": {
        "code": "flat_nu_ridge",
        "note": "nu ridge; coefficient gap ~8e-2 at n.cyc=100.",
    },
    "JSUo": {
        "code": "flat_nu_tau_ridge",
        "note": "coefficient gap ~5e-3 at n.cyc=100; 4-parameter heavy-tail ridge.",
    },
    "GT": {
        "code": "diverged",
        "note": "R and our fit diverged at n.cyc=100 (mu gap ~8e7).",
    },
    "BCCG": {
        "code": "unstable_seed",
        "note": "2e-3..3e-1 coefficient gap across seeds; one seed jumped to another optimum.",
    },
    "ZIP": {
        "code": "coefficient_gap",
        "note": "sigma/nu gap ~2e-3 at n.cyc=100 on the probe sample.",
    },
    "DEL": {
        "code": "coefficient_gap",
        "note": "sigma gap ~5e-2 and nu ~1.3e1 at n.cyc=100.",
    },
    "SIMPLEX": {
        "code": "coefficient_gap",
        "note": "coefficient gap ~1e-3 at n.cyc=100.",
    },
    "ZIPF": {
        "code": "unstable_seed",
        "note": "2/3 seeds pass 2e-4; third mu gap 7.7e-4 (no sigma in rZIPF).",
    },
    "DELAPORT": {
        "code": "no_r_reference",
        "note": "installed gamlss.dist 6.1.1 has no DELAPORT family / r function.",
    },
    "ZABI": {
        "code": "r_density_vectorization_defect",
        "note": (
            "R's vectorised dZABI collapses the hurdle correction to row 1 "
            "(scalar dBI(0, ..) truncates via ifelse), so R's own fit optimises "
            "a defective objective and its coefficients are not a parity target. "
            "Ours is the elementwise density; pinned by "
            "test_binomial_zero_variants_bd.py::TestTheRReferenceIsDefective."
        ),
    },
    "BCPE": {
        "code": "flat_four_param_ridge",
        "note": (
            "both sides converge in <25 cycles, so a longer budget changes "
            "nothing; they stop at different local optima of a very flat "
            "likelihood (deviance 115.219 vs 115.225, coefficients off by "
            "3e-3..4e-2). Not a fit-coefficient parity target."
        ),
    },
    "BCT": {
        "code": "flat_four_param_ridge",
        "note": (
            "same as BCPE: both converge early, budget-independent, different "
            "optima (deviance 135.069 vs 135.081, nu/tau off by ~1e-1)."
        ),
    },
    "IGA": {
        "code": "no_r_reference",
        "note": "installed gamlss.dist 6.1.1 has no IGA family (gamlss.family('IGA') errors).",
    },
    "ZALG": {
        "code": "unstable_mu",
        "note": "1/3 seeds within 2e-4; mu gap 1.2e-3..4.1e-3 while deviance matched (flat mu ridge).",
    },
    "PARETO2": {
        "code": "flat_mu_sigma_ridge",
        "note": (
            "deviance matches to 4 digits but mu/sigma differ ~1e-2 on every seed "
            "(heavy-tail shape ridge); heavy-tail samples (ymax>2000) also fail "
            "initial-value validation."
        ),
    },
    "ZIBI": {
        "code": "r_density_vectorization_defect",
        "note": (
            "R's vectorised dZIBI pastes row 1's base zero mass onto every y==0 "
            "row (same scalar-dBI collapse); pinned by "
            "test_binomial_zero_variants_bd.py::TestTheRReferenceIsDefective."
        ),
    },
}

#: Validated families not yet probed, with a short next-step hint.
#: Every validated family is now classified, so nothing is pending.  Kept as
#: the extension point for a newly admitted family that has not been probed.
PENDING_NOTES: dict[str, str] = {}


def _golden_families() -> dict[str, list[str]]:
    manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    families: dict[str, list[str]] = {}
    for case, entry in manifest.items():
        families.setdefault(entry["family"], []).append(case)
    return {family: sorted(cases) for family, cases in families.items()}


def _validated_families() -> list[str]:
    from omnilss.family_capabilities import _R_CONSISTENCY_EVIDENCE

    return sorted(_R_CONSISTENCY_EVIDENCE)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def build() -> dict:
    validated = _validated_families()
    golden = _golden_families()

    unclassified = sorted(set(golden) - set(validated))
    if unclassified:
        raise SystemExit(
            "golden families missing from _R_CONSISTENCY_EVIDENCE: " + ", ".join(unclassified)
        )
    unknown_exclusions = sorted(set(EXCLUSIONS) - set(validated))
    if unknown_exclusions:
        raise SystemExit(
            "EXCLUSIONS name families absent from _R_CONSISTENCY_EVIDENCE: "
            + ", ".join(unknown_exclusions)
        )
    unknown_pending = sorted(set(PENDING_NOTES) - set(validated))
    if unknown_pending:
        raise SystemExit(
            "PENDING_NOTES name families absent from _R_CONSISTENCY_EVIDENCE: "
            + ", ".join(unknown_pending)
        )

    promoted = []
    pending = []
    excluded = []
    seen: dict[str, str] = {}
    for family in validated:
        if family in golden:
            tier = "promoted"
            promoted.append({"family": family, "golden_cases": golden[family]})
        elif family in EXCLUSIONS:
            tier = "excluded"
            excluded.append(
                {"family": family, **EXCLUSIONS[family]}
            )
        else:
            tier = "pending"
            pending.append(
                {"family": family, "note": PENDING_NOTES.get(family, "not_yet_probed")}
            )
        if family in seen:
            raise SystemExit(f"family in two tiers: {family} ({seen[family]}, {tier})")
        seen[family] = tier

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "P1-1 frozen high-frequency family list: every r_consistency-validated "
            "family is classified promoted (has golden), pending (not probed), or "
            "excluded (probed, rejected with a reason). Fail-closed on unclassified."
        ),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": _git("rev-parse", "HEAD"),
        "sources": {
            "validated_source": "omnilss.family_capabilities._R_CONSISTENCY_EVIDENCE",
            "golden_manifest": "omnilss/tests/fixtures/r_golden/regenerated/manifest.json",
        },
        "tiers": {
            "promoted": sorted(promoted, key=lambda item: item["family"]),
            "pending": sorted(pending, key=lambda item: item["family"]),
            "excluded": sorted(excluded, key=lambda item: item["family"]),
        },
        "counts": {
            "promoted": len(promoted),
            "pending": len(pending),
            "excluded": len(excluded),
            "total_validated": len(validated),
        },
    }


def _stable(artifact: dict) -> dict:
    return {
        "schema_version": artifact["schema_version"],
        "sources": artifact["sources"],
        "tiers": artifact["tiers"],
        "counts": artifact["counts"],
    }


def _render_markdown(artifact: dict) -> str:
    tiers = artifact["tiers"]
    lines = [
        "# P1-1 frozen high-frequency family list",
        "",
        f"- generated: {artifact['generated_at_utc']}",
        f"- source commit: {artifact['source_commit']}",
        f"- validated families: {artifact['counts']['total_validated']}",
        f"- promoted: {artifact['counts']['promoted']} / "
        f"pending: {artifact['counts']['pending']} / "
        f"excluded: {artifact['counts']['excluded']}",
        "",
        "## Promoted (golden evidence present)",
        "",
        "| family | golden cases |",
        "|---|---|",
    ]
    for item in tiers["promoted"]:
        lines.append(f"| {item['family']} | {', '.join(item['golden_cases'])} |")
    lines += ["", "## Pending (not probed)", "", "| family | note |", "|---|---|"]
    for item in tiers["pending"]:
        lines.append(f"| {item['family']} | {item['note']} |")
    lines += ["", "## Excluded (probed, rejected)", "", "| family | code | note |", "|---|---|---|"]
    for item in tiers["excluded"]:
        lines.append(f"| {item['family']} | {item['code']} | {item['note']} |")
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
    for key in ("schema_version", "tiers", "counts", "sources"):
        if key not in artifact:
            problems.append(f"missing key: {key}")
    if problems:
        return False, problems

    current = build()
    if artifact["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {artifact['schema_version']} != {SCHEMA_VERSION}")
    if _stable(artifact) != _stable(current):
        # Narrow the message to what actually differs.
        for key in ("tiers", "counts", "sources"):
            if artifact.get(key) != current.get(key):
                problems.append(f"{key} drift vs current tree")
    # Partition invariants restated explicitly (defence in depth).
    tiers = artifact["tiers"]
    promoted = {item["family"] for item in tiers["promoted"]}
    pending = {item["family"] for item in tiers["pending"]}
    excluded = {item["family"] for item in tiers["excluded"]}
    validated = set(_validated_families())
    union = promoted | pending | excluded
    if promoted & pending or promoted & excluded or pending & excluded:
        problems.append("tiers overlap")
    if union != validated:
        problems.append(f"tiers do not cover validated families: {sorted(validated - union)} missing")
    if promoted != set(_golden_families()):
        problems.append("promoted set != families with golden cases")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None,
                        help="write the frozen artifact (JSON + Markdown)")
    parser.add_argument("--check", action="store_true",
                        help="validate an existing artifact against the current tree")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT,
                        help="artifact path for --check")
    parser.add_argument("--json", action="store_true", help="print the artifact to stdout")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_artifact(args.path)
        print(f"high-frequency freeze check: {args.path}")
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
        f"promoted={counts['promoted']} pending={counts['pending']} "
        f"excluded={counts['excluded']} validated={counts['total_validated']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
