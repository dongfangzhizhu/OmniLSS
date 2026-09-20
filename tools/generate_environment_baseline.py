"""Generate a reproducible Python/R/WSL environment baseline."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str], timeout: int = 30) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_value(repo_root: Path, *args: str) -> str | None:
    return _run(["git", "-C", str(repo_root), *args])


def collect(repo_root: Path | None = None, wsl_distribution: str = "debianjaxgpu") -> dict:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    git_status = _git_value(repo_root, "status", "--porcelain")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {},
        "r": {"available": False, "status": "unavailable"},
        "wsl": {"available": False, "status": "unavailable", "distribution": wsl_distribution},
        "source": {
            "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_dirty": None if git_status is None else bool(git_status),
            "git_status": git_status,
        },
        "jax": {"x64_enabled": None, "backend": None, "devices": []},
    }
    try:
        import jax, jaxlib, numpy, scipy
        result["packages"] = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__}
        result["jax"] = {
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
        }
    except ImportError as exc:
        result["packages_error"] = str(exc)
    rscript = shutil.which("Rscript")
    r = _run([rscript or "Rscript", "-e", "cat(R.version.string); cat('\\n'); for (p in c('gamlss','gamlss.dist','jsonlite')) { cat(p, '=', if (requireNamespace(p, quietly=TRUE)) as.character(packageVersion(p)) else 'MISSING', '\\n') }"]) if rscript else None
    if r:
        result["r"] = {"available": True, "status": "available", "executable": rscript, "version": r}
    elif rscript:
        result["r"] = {"available": False, "status": "error", "executable": rscript}
    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
    wsl = _run([wsl_exe, "-d", wsl_distribution, "--", "python", "-c", "import sys, jax; print(sys.version); print(jax.__version__); print(jax.config.jax_enable_x64); print(jax.devices())"], timeout=60) if wsl_exe else None
    if wsl:
        result["wsl"] = {"available": True, "status": "available", "distribution": wsl_distribution, "executable": wsl_exe, "output": wsl}
    elif wsl_exe:
        result["wsl"] = {"available": False, "status": "error", "distribution": wsl_distribution, "executable": wsl_exe}
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
