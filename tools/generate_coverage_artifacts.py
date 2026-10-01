"""Generate all P0.1 coverage artifacts from one capability snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="date suffix for the queue artifact (YYYY-MM-DD)")
    args = parser.parse_args()
    date = args.date or __import__("datetime").date.today().isoformat()
    coverage_mod = _load("generate_r_export_coverage", ROOT / "tools" / "generate_r_export_coverage.py")
    queue_mod = _load("generate_placeholder_migration_queue", ROOT / "tools" / "generate_placeholder_migration_queue.py")
    symbol_mod = _load("generate_symbol_coverage", ROOT / "tools" / "generate_symbol_coverage.py")
    diff_mod = _load("coverage_diff_report", ROOT / "tools" / "coverage_diff_report.py")
    coverage = coverage_mod.build_coverage()
    coverage_path = ROOT / "docs" / "development" / "r-export-coverage.json"
    markdown_path = ROOT / "docs" / "development" / "r-export-coverage.md"
    coverage_mod.write_coverage(coverage, coverage_path, markdown_path)
    queue_path = ROOT / "docs" / "development" / f"placeholder-migration-queue-{date}.md"
    # Queue generation consumes the exact in-memory snapshot written above.
    queue_mod.generate_queue(coverage, queue_path, date)
    # D0.1: the real R <-> Python per-symbol matrix (parses R sources, registry,
    # tests and the golden manifest), separate from the registry inventory above.
    symbol_coverage = symbol_mod.build_coverage()
    symbol_mod.write_coverage(symbol_coverage)
    # D0.4: the ratchet diff consumes the exact in-memory matrix written above, so
    # the report snapshot can never disagree with the committed matrix.
    diff_report = diff_mod.build_report(coverage=symbol_coverage)
    diff_mod.OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    diff_mod.OUT_JSON.write_text(
        json.dumps(diff_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Generated: {coverage_path}")
    print(f"Generated: {markdown_path}")
    print(f"Generated: {queue_path}")
    print(f"Generated: {symbol_mod.OUT_JSON}")
    print(f"Generated: {symbol_mod.OUT_MD}")
    print(f"Generated: {diff_mod.OUT_JSON}")
    if not diff_report["ok"]:
        print(diff_mod.render_report(diff_report))
        print("coverage ratchet regression: run tools/coverage_diff_report.py --update-baseline after review")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
