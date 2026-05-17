"""Validate an OmniLSS JSON model artifact.

Usage
-----
PYTHONPATH=src python tools/validate_model_artifact.py model.omnilss
PYTHONPATH=src python tools/validate_model_artifact.py model.omnilss --fail-on-warning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from omnilss.serialization import validate_model_json


def validate_artifact(path: str | Path) -> dict:
    """Return the schema validation report for a model artifact."""

    return validate_model_json(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Path to .omnilss JSON artifact")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero when the artifact has warnings as well as errors",
    )
    args = parser.parse_args(argv)

    report = validate_artifact(args.artifact)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ok", False):
        return 1
    if args.fail_on_warning and report.get("warnings"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
