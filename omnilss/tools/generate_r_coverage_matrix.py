"""Generate an R-to-OmniLSS symbol coverage matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "development" / "r-coverage-matrix-2026-05-22.json"


def _discover_namespace_path(*parts: str) -> Path:
    """Best-effort locate a NAMESPACE file under common repository layouts."""
    preferred = REPO_ROOT.joinpath(*parts)
    if preferred.exists():
        return preferred
    candidates = [
        REPO_ROOT / "omnilss" / Path(*parts),
        ROOT / Path(*parts),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred


def _parse_namespace_exports(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    exports: set[str] = set()
    for match in re.finditer(r"export\(([^)]*)\)", text, flags=re.DOTALL):
        for token in match.group(1).split(","):
            name = token.strip()
            if name:
                exports.add(name)
    return exports


def _load_capability(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return {
        str(row.get("family", "")).upper(): row
        for row in rows
        if str(row.get("family", "")).strip()
    }


def _collect_r_test_families(tests_root: Path) -> set[str]:
    fams: set[str] = set()
    skip = {"SMOOTH", "BATCH1_BC", "BATCH3_4_REMAINING", "BATCH5", "BATCH6", "BATCH7_8"}
    for path in sorted(tests_root.glob("test_r_consistency_*.py")):
        name = path.stem.replace("test_r_consistency_", "").upper()
        if name and name not in skip:
            fams.add(name)
        text = path.read_text(encoding="utf-8")
        fams.update(
            m.group(1).upper()
            for m in re.finditer(r"(?:family|dist_name)\s*=\s*['\"]([A-Za-z0-9]+)['\"]", text)
        )
    return fams


def build_matrix() -> dict[str, Any]:
    gamlss_ns = _discover_namespace_path("R", "gamlss", "NAMESPACE")
    dist_ns = _discover_namespace_path("R", "gamlss.dist", "NAMESPACE")
    gamlss_exports = _parse_namespace_exports(gamlss_ns)
    dist_exports = _parse_namespace_exports(dist_ns)

    from omnilss.distribution_registry import _BUILTIN_FAMILY_FACTORIES

    fams = set(_BUILTIN_FAMILY_FACTORIES.keys())
    tests = _collect_r_test_families(ROOT / "tests")
    cap = _load_capability(REPO_ROOT / "docs" / "development" / "family-capability-matrix-2026-05-18.json")

    rows: list[dict[str, Any]] = []
    for symbol in sorted(gamlss_exports | dist_exports):
        upper = symbol.upper()
        module, factory = _BUILTIN_FAMILY_FACTORIES.get(upper, (None, None))
        rows.append(
            {
                "r_symbol": symbol,
                "r_source": "gamlss.dist" if symbol in dist_exports else "gamlss",
                "python_module": module,
                "python_entry": factory,
                "implemented": upper in fams,
                "r_consistency_tested": upper in tests,
                "capability_tier": cap.get(upper, {}).get("evidence_tier"),
                "capability_r_consistency": cap.get(upper, {}).get("r_consistency"),
            }
        )

    return {
        "generated_on": "2026-05-22",
        "source_paths": {
            "gamlss_namespace": str(gamlss_ns),
            "gamlss_dist_namespace": str(dist_ns),
        },
        "rows": rows,
        "summary": {
            "r_export_count": len(gamlss_exports | dist_exports),
            "implemented_symbol_count": sum(1 for r in rows if r["implemented"]),
            "r_tested_symbol_count": sum(1 for r in rows if r["r_consistency_tested"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
