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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "development" / "r-export-coverage.json")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown registry summary")
    args = parser.parse_args()
    # Add omnilss src to path
    sys.path.insert(0, str(REPO_ROOT / "omnilss" / "src"))

    from omnilss.family_capabilities import list_family_capabilities, family_capability_names

    # Gather all registered family names
    all_names = sorted(family_capability_names())

    # Build capability lookup
    cap_lookup = {}
    for cap in list_family_capabilities():
        cap_lookup[cap.name] = cap

    coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo_root": str(REPO_ROOT),
        "stats": {
            "total_registered": len(all_names),
        },
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
            # Add evidence tier if available
            try:
                tier = cap.status("dpqr")
                info["evidence_tier_dpqr"] = tier.name
            except Exception:
                info["evidence_tier_dpqr"] = "unknown"
            try:
                tier = cap.status("fitting")
                info["evidence_tier_fitting"] = tier.name
            except Exception:
                info["evidence_tier_fitting"] = "unknown"

        status_counts[status] = status_counts.get(status, 0) + 1
        coverage["families"][name] = info

    coverage["stats"]["status_distribution"] = status_counts
    coverage["stats"]["validated_count"] = status_counts.get("implemented", 0)
    coverage["stats"]["experimental_count"] = status_counts.get("experimental", 0)
    coverage["stats"]["compat_alias_count"] = status_counts.get("compat_alias", 0)
    coverage["stats"]["placeholder_count"] = status_counts.get("placeholder", 0)
    coverage["stats"]["missing_count"] = status_counts.get("missing", 0)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    if args.markdown is not None:
        lines = [
            "# OmniLSS registry implementation inventory", "",
            "Registry metadata only: this report does not establish R export coverage or numerical parity.",
            "", "| Family | Implementation status |", "|---|---|",
        ]
        lines.extend(
            f"| {name} | {info.get('implementation_status', 'missing')} |"
            for name, info in coverage["families"].items()
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Generated: {output_path}")
    print(f"Total registered: {coverage['stats']['total_registered']}")
    print(f"Status distribution: {json.dumps(status_counts, indent=2)}")


if __name__ == "__main__":
    main()
