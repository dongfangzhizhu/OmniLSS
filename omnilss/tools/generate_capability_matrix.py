"""Generate the machine-readable OmniLSS family capability matrix.

Usage
-----
PYTHONPATH=src python tools/generate_capability_matrix.py \
    --output ../docs/development/family-capability-matrix-2026-05-18.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from omnilss.family_capabilities import capability_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT.parent / "docs" / "development" / "family-capability-matrix-2026-05-18.json"
)


def write_capability_matrix(path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write the current runtime capability matrix to ``path`` as JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = capability_matrix()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)
    output = write_capability_matrix(args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
