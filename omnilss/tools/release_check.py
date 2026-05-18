"""Run local release checks for OmniLSS v0.3.0.

This helper runs packaging checks in order and prints actionable diagnostics
when optional tooling is missing in constrained environments.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

_LOCALIZATION_TOOL = Path(__file__).with_name("check_docs_localization.py")
_CAPABILITY_MATRIX_TOOL = Path(__file__).with_name("validate_capability_matrix.py")
_LOCALIZATION_SPEC = importlib.util.spec_from_file_location(
    "check_docs_localization", _LOCALIZATION_TOOL
)
check_docs_localization = importlib.util.module_from_spec(_LOCALIZATION_SPEC)
assert _LOCALIZATION_SPEC is not None and _LOCALIZATION_SPEC.loader is not None
_LOCALIZATION_SPEC.loader.exec_module(check_docs_localization)
validate_docs_localization = check_docs_localization.validate_docs_localization

_CAPABILITY_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "validate_capability_matrix", _CAPABILITY_MATRIX_TOOL
)
validate_capability_matrix = importlib.util.module_from_spec(_CAPABILITY_MATRIX_SPEC)
assert (
    _CAPABILITY_MATRIX_SPEC is not None
    and _CAPABILITY_MATRIX_SPEC.loader is not None
)
_CAPABILITY_MATRIX_SPEC.loader.exec_module(validate_capability_matrix)
validate_capability_matrix_file = (
    validate_capability_matrix.validate_capability_matrix_file
)


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> int:
    for tool in ("python",):
        if shutil.which(tool) is None:
            print(f"Missing required executable: {tool}", file=sys.stderr)
            return 2

    localization_errors = validate_docs_localization()
    if localization_errors:
        print("Documentation localization check failed:", file=sys.stderr)
        for error in localization_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    matrix_report = validate_capability_matrix_file()
    if not matrix_report["ok"]:
        print("Capability matrix validation failed:", file=sys.stderr)
        for issue in matrix_report["issues"]:
            print(
                f"- {issue['severity']} {issue['code']} at {issue['path']}: "
                f"{issue['message']}",
                file=sys.stderr,
            )
        return 1

    if _run([sys.executable, "-m", "build"]) != 0:
        print(
            "Build step failed. If build module is missing, install with: "
            "pip install build",
            file=sys.stderr,
        )
        return 1

    if _run([sys.executable, "-m", "twine", "check", "dist/*"]) != 0:
        print(
            "Twine check failed. Ensure twine is installed: pip install twine",
            file=sys.stderr,
        )
        return 1

    print("Release checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
