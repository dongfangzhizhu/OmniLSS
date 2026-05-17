"""Run local release checks for OmniLSS v0.3.0.

This helper runs packaging checks in order and prints actionable diagnostics
when optional tooling is missing in constrained environments.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


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
