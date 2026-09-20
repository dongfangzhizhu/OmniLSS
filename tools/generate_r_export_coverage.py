# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate r-export-coverage.json from OmniLSS family registry and capabilities.

This is the single source of truth for implementation status, replacing hand-maintained
completion rates. Run with JAX_ENABLE_X64=True to match R parity test environment.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def build_coverage() -> dict:
    """Build the registry inventory without writing any files."""
    # Add omnilss src to path when this module is imported by another tool.
    sys.path.insert(0, str(REPO_ROOT / "omnilss" / "src"))

    from omnilss.family_capabilities import list_family_capabilities, family_capability_names

    all_names = sorted(family_capability_names())
    cap_lookup = {cap.name: cap for cap in list_family_capabilities()}
    coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo_root": str(REPO_ROOT),
        "stats": {"total_registered": len(all_names)},
        "families": {},
    }
    status_counts = {}
    for name in all_names:
        cap = cap_lookup.get(name)
        if cap is None:
            status = "missing"
            info = {"note": "No FamilyCapability entry found"}
        else:
            status = cap.implementation_status.value
            info = {
                "implementation_status": status,
                "is_production_safe": bool(cap.is_production_safe),
                "has_dqr": bool(getattr(cap, "dqr", False)),
                "has_fitting": bool(getattr(cap, "fitting", False)),
                "has_smoothing": bool(getattr(cap, "smoothing", False)),
            }
            for key, evidence in (("r_consistency", "evidence_tier_dpqr"), ("fitting", "evidence_tier_fitting")):
                try:
                    info[evidence] = cap.status(key).name
                except Exception:
                    info[evidence] = "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        coverage["families"][name] = info

    coverage["stats"]["status_distribution"] = status_counts
    for status, key in (("implemented", "validated_count"), ("experimental", "experimental_count"),
                        ("compat_alias", "compat_alias_count"), ("placeholder", "placeholder_count"),
                        ("missing", "missing_count")):
        coverage["stats"][key] = status_counts.get(status, 0)
    return coverage


def write_coverage(coverage: dict, output: Path, markdown: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if markdown is not None:
        lines = [
            "# OmniLSS registry implementation inventory", "",
            "Registry metadata only: this report does not establish R export coverage or numerical parity.",
            "", "| Family | Implementation status |", "|---|---|",
        ]
        lines.extend(f"| {name} | {info.get('implementation_status', 'missing')} |"
                     for name, info in coverage["families"].items())
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "development" / "r-export-coverage.json")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown registry summary")
    args = parser.parse_args()
    coverage = build_coverage()
    write_coverage(coverage, args.output, args.markdown)

    print(f"Generated: {args.output}")
    print(f"Total registered: {coverage['stats']['total_registered']}")
    print(f"Status distribution: {json.dumps(coverage['stats']['status_distribution'], indent=2)}")


if __name__ == "__main__":
    main()
