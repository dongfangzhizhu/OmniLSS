"""Validate a generated OmniLSS family capability matrix artifact.

Usage
-----
PYTHONPATH=src python tools/validate_capability_matrix.py \
    ../docs/development/family-capability-matrix-2026-05-18.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from omnilss.family_capabilities import validate_capability_matrix_payload

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT.parent / "docs" / "development" / "family-capability-matrix-2026-05-18.json"
)


def validate_capability_matrix_file(
    path: str | Path = DEFAULT_INPUT,
) -> dict[str, object]:
    """Validate a capability matrix JSON file and return a report."""

    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = validate_capability_matrix_payload(payload)
    return {"path": str(input_path), **report}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Capability matrix JSON path (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args(argv)
    report = validate_capability_matrix_file(args.path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
