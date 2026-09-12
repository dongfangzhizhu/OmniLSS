"""Generate a reproducible Python/R/WSL environment baseline."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect() -> dict:
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {},
        "r": {"available": False},
        "wsl": {"available": False},
    }
    try:
        import jax, jaxlib, numpy, scipy
        result["packages"] = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__}
        result["jax_devices"] = [str(d) for d in jax.devices()]
    except ImportError as exc:
        result["packages_error"] = str(exc)
    r = _run(["Rscript", "-e", "cat(R.version.string); cat('\\n'); for (p in c('gamlss','gamlss.dist')) { if (requireNamespace(p, quietly=TRUE)) cat(p, '=', as.character(packageVersion(p)), '\\n') }"])
    if r:
        result["r"] = {"available": True, "version": r}
    wsl = _run(["wsl.exe", "--", "bash", "-lc", "python -c 'import sys; print(sys.version); import jax; print(jax.__version__); print(jax.devices())'"])
    if wsl:
        result["wsl"] = {"available": True, "output": wsl}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path("docs/reports/environment-baseline.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()