"""Generate the P5-1b contract artifact (OmniLSS <-> ondil, 22 families).

The artifact is the executable half of
docs/development/p5-ondil-distribution-contract-2026-11-15.md: one row per
overlapping family with its theta order, supports, links, method coverage and
a boundary probe.  The ondil side is read from the vendored source (regex, no
import: ondil needs scikit-learn/numba, contract doc D7), so the declared
ONDIL_PARAMETER_ORDER and FAMILY_SUPPORT tables are checked against the files
instead of being trusted.

    python tools/ondil_contract_scan.py --write docs/development/p5-ondil-contract-2026-11-15.json
    python tools/ondil_contract_scan.py --check
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

ONDIL_DISTRIBUTIONS = ROOT / "ondil" / "src" / "ondil" / "distributions"
DEFAULT_OUT = ROOT / "docs" / "development" / "p5-ondil-contract-2026-11-15.json"

#: Declared differences between the vendored ondil tables and this contract,
#: with the side that wins.  R semantics decide the support; ondil declarations
#: are only the cross-check.
SUPPORT_EXEMPTIONS = {
    "EXP": "R: dexp(0) = rate > 0, so 0 is in the support; ondil excludes it",
    "IGAMMA": "R: dIGAMMA(0) = 0 (open bound); ondil declares 0 closed",
    "LOGNO": "R: dlnorm(0) = 0 (open bound); ondil declares 0 closed",
    "LOGNO2": "R: density at 0 is 0 (open bound); ondil declares 0 closed",
}


def _equivalent(left: dict, right: dict) -> bool:
    """Compare supports up to the 5e-324 / 1-1e-16 spelling of an open bound."""
    for key, tolerance in (("lower", 1e-300), ("upper", 1e-12)):
        first, second = left[key], right[key]
        if (first is None) != (second is None):
            return False
        if first is not None and abs(float(first) - float(second)) > tolerance:
            return False
    return (
        left["lower_closed"] == right["lower_closed"]
        and left["upper_closed"] == right["upper_closed"]
    )


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def _parse_support(raw: str) -> dict:
    """Turn a vendored distribution_support literal into normalized flags.

    np.inf / np.nextafter are rewritten to plain floats first so the result is
    a literal Python tuple that ast.literal_eval can read (no eval of repo
    code).  nextafter(0, 1) is how ondil spells an open lower bound, so
    normalizing it back to (0.0, open) is what makes the comparison meaningful.
    """
    text = raw.strip()
    if not text.startswith("(") or not text.endswith(")"):
        raise ValueError(f"not a support tuple: {raw!r}")
    parts = _split_top_level(text[1:-1])
    if len(parts) != 2:
        raise ValueError(f"cannot split support {raw!r}")
    lower, lower_closed = _bound(parts[0])
    upper, upper_closed = _bound(parts[1])
    return {
        "lower": lower,
        "upper": upper,
        "lower_closed": lower_closed,
        "upper_closed": upper_closed,
    }


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _bound(token: str) -> tuple[float | None, bool]:
    """Parse one bound; nextafter(0, 1) is ondil spelling an open bound."""
    token = token.strip()
    match = re.fullmatch(r"np\.nextafter\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", token)
    if match:
        value = math.nextafter(float(match.group(1)), float(match.group(2)))
        return float(value), False
    if token in {"np.inf", "math.inf", "inf"}:
        return None, False
    if token in {"-np.inf", "-math.inf", "-inf"}:
        return None, False
    return float(ast.literal_eval(token)), True


def read_ondil_tables() -> dict[str, dict]:
    """Parse corresponding_gamlss / parameter_names / distribution_support."""
    tables: dict[str, dict] = {}
    for path in sorted(ONDIL_DISTRIBUTIONS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        names = re.findall(r'corresponding_gamlss[^=]*=\s*"([^"]+)"', text)
        orders = re.findall(r"parameter_names\s*=\s*\{([^}]*)\}", text)
        supports = []
        for line in text.splitlines():
            if "distribution_support" not in line or "=" not in line:
                continue
            raw = line.split("=", 1)[1].split("#", 1)[0].strip()
            if raw.startswith("("):
                supports.append(raw)
        for index, gam in enumerate(names):
            entry: dict = {"module": path.stem}
            if index < len(orders):
                pairs = re.findall(r'(\d+)\s*:\s*"([^"]+)"', orders[index])
                entry["parameter_order"] = [
                    name for _, name in sorted(pairs, key=lambda pair: int(pair[0]))
                ]
            if index < len(supports):
                try:
                    entry["support"] = _parse_support(supports[index])
                except ValueError as error:
                    entry["support_error"] = str(error)
            tables[gam] = entry
    return tables


def build() -> dict:
    from omnilss.ondil_contract import (
        FAMILY_SUPPORT,
        KNOWN_BOUNDARY_VIOLATIONS,
        ONDIL_PARAMETER_ORDER,
        contract_report,
        probe_support,
    )

    ondil = read_ondil_tables()
    families = []
    order_mismatches = []
    support_mismatches = []
    for name in sorted(ONDIL_PARAMETER_ORDER):
        report = contract_report(name)
        probe = probe_support(name)
        reference = ondil.get(name, {})
        if reference.get("parameter_order") != report["estimable"]:
            order_mismatches.append(
                {
                    "family": name,
                    "omnilss": report["estimable"],
                    "ondil": reference.get("parameter_order"),
                }
            )
        declared = FAMILY_SUPPORT[name].normalized()
        if "support" in reference and not _equivalent(reference["support"], declared):
            if name not in SUPPORT_EXEMPTIONS:
                support_mismatches.append(
                    {"family": name, "omnilss": declared, "ondil": reference["support"]}
                )
        report["probe"] = probe
        report["ondil_module"] = reference.get("module");
        report["ondil_support"] = reference.get("support");
        families.append(report)

    boundary_violations = [
        family["family"] for family in families if not family["probe"]["boundary_ok"]
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _commit(),
        "contract": "docs/development/p5-ondil-distribution-contract-2026-11-15.md",
        "ondil_available": _ondil_available(),
        "ondil_source": {
            "path": "ondil/src/ondil/distributions",
            "families_parsed": len(ondil),
        },
        "summary": {
            "families": len(families),
            "order_mismatches": len(order_mismatches),
            "support_mismatches": len(support_mismatches),
            "support_exemptions": sorted(SUPPORT_EXEMPTIONS),
            "boundary_violations": boundary_violations,
            "known_boundary_violations": sorted(KNOWN_BOUNDARY_VIOLATIONS),
        },
        "order_mismatches": order_mismatches,
        "support_mismatches": support_mismatches,
        "families": families,
    }


def _ondil_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("ondil") is not None and importlib.util.find_spec(
        "sklearn"
    ) is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None, help="artifact path")
    parser.add_argument("--check", action="store_true", help="compare with the artifact")
    args = parser.parse_args(argv)

    payload = build()
    summary = payload["summary"]
    print(
        "families={families} order_mismatches={order_mismatches} "
        "support_mismatches={support_mismatches} boundary_violations={violations}".format(
            violations=",".join(summary["boundary_violations"]) or "-", **summary
        )
    )

    if args.check:
        if not DEFAULT_OUT.exists():
            print(f"FAIL: {DEFAULT_OUT} missing; run with --write", file=sys.stderr)
            return 1
        committed = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
        problems = []
        if committed.get("summary", {}).get("families") != summary["families"]:
            problems.append("family count drift")
        if committed.get("summary", {}).get("order_mismatches") != summary["order_mismatches"]:
            problems.append("order mismatch count drift")
        if committed.get("summary", {}).get("support_mismatches") != summary["support_mismatches"]:
            problems.append("support mismatch count drift")
        if committed.get("summary", {}).get("boundary_violations") != summary["boundary_violations"]:
            problems.append("boundary violation set drift")
        committed_names = [row["family"] for row in committed.get("families", [])]
        if committed_names != [row["family"] for row in payload["families"]]:
            problems.append("family list drift")
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}", file=sys.stderr)
            return 1
        print("contract artifact is up to date")
        return 0

    out = args.write or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if not summary["order_mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
