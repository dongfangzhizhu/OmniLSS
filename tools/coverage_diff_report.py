# SPDX-License-Identifier: GPL-3.0-or-later
"""D0.4: automatic R <-> Python coverage difference report and regression ratchet.

Consumes the D0.1 per-symbol matrix (``tools/generate_symbol_coverage.py``) and
enumerates the four defect classes the deliverable plan calls out, then compares
them against a checked-in ratchet baseline:

=================  ==========================================================
category           definition (from the matrix layers)
=================  ==========================================================
``missing``        R export with no Python surface at all (``surface`` layer).
``placeholder``    family registered with placeholder kernels (``kernel`` layer).
``compat_alias``   family provided as a compatibility alias (``kernel`` layer).
``evidence_gap``   implemented symbol with no test evidence (``parity`` layer is
                   ``unverified``): the "no evidence" class.
``extra``          Python-only public exports (``omnilss.__all__`` names without
                   an R counterpart); reported, never fatal.
=================  ==========================================================

Baseline semantics (fail-closed ratchet, ``docs/development/symbol-coverage-baseline.json``):

* every symbol that is ``missing``/``placeholder``/``compat_alias``/
  ``evidence_gap`` in the current matrix but absent from the baseline is a
  **regression**; the command exits 1 and prints the offending symbols;
* symbols that leave those categories are **improvements** and are reported
  without failing;
* ``--update-baseline`` rewrites the ratchet (a reviewable commit that must
  lower or explicitly accept the ceilings), which is the only way to bless a
  regression.

Exit code 0 means "no regression relative to the ratchet"; ``--json`` emits a
machine-readable report for CI artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow ``python tools/coverage_diff_report.py``
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_symbol_coverage import build_coverage  # noqa: E402
from tools.coverage_scope import scope_line, scope_payload  # noqa: E402

BASELINE_PATH = REPO_ROOT / "docs" / "development" / "symbol-coverage-baseline.json"
OUT_JSON = REPO_ROOT / "docs" / "development" / "symbol-coverage-diff.json"

SCHEMA_VERSION = 1

#: Categories that must never grow without an explicit baseline update.
RATCHET_CATEGORIES = ("missing", "placeholder", "compat_alias", "evidence_gap")

#: Categories that are reported for review and cannot fail the gate by themselves.
INFORMATIONAL_CATEGORIES = ("extra",)

CATEGORY_MEANING = {
    "missing": "R 导出符号在 Python 侧没有任何 surface（surface 层为 missing）",
    "placeholder": "家族以 placeholder 核注册（kernel 层为 placeholder）",
    "compat_alias": "家族以兼容别名提供（kernel 层为 compat_alias）",
    "evidence_gap": "已实现但完全没有测试证据（parity 层为 unverified）",
    "extra": "Python 独有的公开导出（omnilss.__all__ 中无 R 对应项），仅报告",
}


def _symbol_id(row: dict) -> str:
    """Stable ``package::symbol`` identifier used by the baseline."""
    return f"{row['r_package']}::{row['r_symbol']}"


def _display_path(path: Path) -> str:
    """Repo-relative path when possible so committed artifacts stay machine independent."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def categorise(rows: list[dict], python_only_exports: list[str]) -> dict[str, list[str]]:
    """Split the matrix rows into the D0.4 defect categories (sorted symbol ids)."""
    buckets: dict[str, set[str]] = {key: set() for key in RATCHET_CATEGORIES}
    for row in rows:
        layers = row["layers"]
        symbol = _symbol_id(row)
        if layers["surface"] == "missing":
            buckets["missing"].add(symbol)
            continue
        if layers["kernel"] == "placeholder":
            buckets["placeholder"].add(symbol)
        elif layers["kernel"] == "compat_alias":
            buckets["compat_alias"].add(symbol)
        if layers["kernel"] == "implemented" and layers["parity"] == "unverified":
            buckets["evidence_gap"].add(symbol)
    result = {key: sorted(values) for key, values in buckets.items()}
    result["extra"] = sorted(python_only_exports)
    return result


def build_baseline(current: dict[str, list[str]]) -> dict:
    """Build the ratchet baseline payload from a categorised matrix."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/coverage_diff_report.py --update-baseline",
        "scope": scope_payload(),
        "note": (
            "Ratchet ceilings for D0.4: a symbol appearing in a ratchet category "
            "here may not newly appear in the same category. Lowering these lists "
            "is an improvement; raising them requires an explicit baseline commit."
        ),
        "category_meaning": CATEGORY_MEANING,
        "counts": {key: len(values) for key, values in current.items()},
        "symbols": {key: sorted(values) for key, values in current.items()},
    }


def load_baseline(path: Path) -> tuple[dict | None, str]:
    """Return ``(baseline, error)``; error is empty when readable and well formed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"baseline 不存在: {path}"
    except Exception as error:  # noqa: BLE001 - report any read/parse failure
        return None, f"baseline 不可读: {type(error).__name__}: {error}"
    if not isinstance(payload, dict):
        return None, "baseline 顶层应为 object"
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None, f"baseline schema_version 应为 {SCHEMA_VERSION}，实际 {payload.get('schema_version')!r}"
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return None, "baseline 缺 symbols 对象"
    missing_keys = [key for key in (*RATCHET_CATEGORIES, *INFORMATIONAL_CATEGORIES) if key not in symbols]
    if missing_keys:
        return None, f"baseline symbols 缺类别: {missing_keys}"
    return payload, ""


def diff_categories(
    current: dict[str, list[str]],
    baseline: dict,
) -> dict[str, dict[str, list[str]]]:
    """Return per-category ``new`` (regressions) and ``resolved`` (improvements)."""
    baseline_symbols = baseline.get("symbols", {}) if isinstance(baseline, dict) else {}
    diff: dict[str, dict[str, list[str]]] = {}
    for key in (*RATCHET_CATEGORIES, *INFORMATIONAL_CATEGORIES):
        previous = set(baseline_symbols.get(key, []) or [])
        present = set(current.get(key, []) or [])
        diff[key] = {
            "new": sorted(present - previous),
            "resolved": sorted(previous - present),
            "current_count": len(present),
            "baseline_count": len(previous),
        }
    return diff


def build_report(
    *,
    baseline_path: Path = BASELINE_PATH,
    coverage: dict | None = None,
    emit=print,
) -> dict:
    """Build the D0.4 diff report (never raises; report carries the verdict)."""
    coverage = coverage if coverage is not None else build_coverage(emit)
    current = categorise(coverage["symbols"], coverage["python_only_public_exports"])
    baseline, baseline_error = load_baseline(baseline_path)
    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "baseline": _display_path(baseline_path),
        "baseline_error": baseline_error,
        "r_exports_total": coverage["stats"]["r_exports_total"],
        "scope": scope_payload(),
        "current_counts": {key: len(values) for key, values in current.items()},
        "category_meaning": CATEGORY_MEANING,
        "categories": {},
        "regressions": [],
        "improvements": [],
        "ok": False,
    }
    if baseline is None:
        report["failure"] = baseline_error
        return report
    diff = diff_categories(current, baseline)
    report["categories"] = diff
    for key in RATCHET_CATEGORIES:
        for symbol in diff[key]["new"]:
            report["regressions"].append({"category": key, "symbol": symbol})
        for symbol in diff[key]["resolved"]:
            report["improvements"].append({"category": key, "symbol": symbol})
    report["ratchet_ok"] = not report["regressions"]
    report["ceiling_ok"] = all(
        report["current_counts"][key] <= diff[key]["baseline_count"] for key in RATCHET_CATEGORIES
    )
    report["ok"] = bool(report["ratchet_ok"] and report["ceiling_ok"])
    return report


def render_report(report: dict, *, max_symbols: int = 25) -> str:
    """Human readable rendering of the D0.4 diff report."""
    lines = ["coverage difference report (D0.4)", f"baseline: {report['baseline']}"]
    lines += ["", scope_line()]
    if report.get("failure"):
        lines.append(f"FAIL-CLOSED: {report['failure']}")
        lines.append("run: python tools/coverage_diff_report.py --update-baseline")
        return "\n".join(lines)
    lines.append(f"R exports: {report['r_exports_total']}")
    lines.append("")
    lines.append("| category | current | baseline | new | resolved |")
    lines.append("|---|---|---|---|---|")
    for key, entry in report["categories"].items():
        marker = "" if key in RATCHET_CATEGORIES else " (informational)"
        lines.append(
            f"| {key}{marker} | {entry['current_count']} | {entry['baseline_count']} | "
            f"{len(entry['new'])} | {len(entry['resolved'])} |"
        )
    regressions = report["regressions"]
    if regressions:
        lines += ["", f"REGRESSIONS: {len(regressions)} new symbol(s) in ratchet categories"]
        for entry in regressions[:max_symbols]:
            lines.append(f"  - [{entry['category']}] {entry['symbol']}")
        if len(regressions) > max_symbols:
            lines.append(f"  - ... {len(regressions) - max_symbols} more (see --json)")
    improvements = report["improvements"]
    if improvements:
        lines += ["", f"improvements: {len(improvements)} symbol(s) left ratchet categories"]
        for entry in improvements[:max_symbols]:
            lines.append(f"  - [{entry['category']}] {entry['symbol']}")
        if len(improvements) > max_symbols:
            lines.append(f"  - ... {len(improvements) - max_symbols} more (see --json)")
    if not regressions and not improvements:
        lines += ["", "no symbol-level change relative to the ratchet baseline"]
    lines.append(f"result: {'OK' if report['ok'] else 'FAIL-CLOSED'}")
    return "\n".join(lines)


def write_baseline(payload: dict, path: Path = BASELINE_PATH) -> None:
    """Write the ratchet baseline (LF endings, deterministic ordering)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None, *, coverage: dict | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--out", type=Path, default=OUT_JSON, help="write the JSON report to this path")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the ratchet baseline from the current matrix (reviewable commit)",
    )
    parser.add_argument(
        "--allow-regressions",
        action="store_true",
        help="report regressions but exit 0 (use only for deliberate, reviewed steps)",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # readable reports on Windows consoles
        except (ValueError, OSError):  # pragma: no cover - wrapped streams
            pass
    built = coverage if coverage is not None else build_coverage()
    if args.update_baseline:
        current = categorise(built["symbols"], built["python_only_public_exports"])
        write_baseline(build_baseline(current), args.baseline)
        print(f"baseline updated: {args.baseline}")
        print("counts: " + ", ".join(f"{key}={len(values)}" for key, values in current.items()))
        return 0
    report = build_report(baseline_path=args.baseline, coverage=built)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    except OSError as error:  # pragma: no cover - read-only workspace
        print(f"warning: could not write {args.out}: {error}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_report(report))
    if report["ok"] or args.allow_regressions:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())