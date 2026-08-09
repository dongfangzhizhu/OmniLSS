"""Generate reproducible Release 1 R golden artifacts from vendored sources only.

The runner never loads ``gamlss`` or ``gamlss.dist`` from a global R library:
those packages are installed into a case-local isolated library before execution.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CORE_R_PACKAGES = ("gamlss", "gamlss.dist")
REQUIRED_MANIFEST_FIELDS = frozenset({
    "gamlss_source", "gamlss_dist_source", "r_version", "python_version",
    "jax_version", "jaxlib_version", "numpy_version", "scipy_version",
    "commit_id", "dirty_worktree", "precision_policy", "seed", "formula",
    "parameters", "tolerances", "test_id", "result_path",
})


@dataclass(frozen=True)
class GoldenCase:
    """A deterministic fixed-R comparison case."""

    test_id: str
    family: str
    operation: str
    seed: int
    formula: str | None
    parameters: Mapping[str, object]
    tolerances: Mapping[str, float]


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _description_version(source: Path) -> str:
    match = re.search(r"^Version:\s*(.+)$", (source / "DESCRIPTION").read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"vendored package is missing Version: {source}")
    return match.group(1).strip()


def _git_value(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def git_provenance() -> tuple[str, bool]:
    """Return commit ID and whether any tracked or untracked files are present."""
    return _git_value("rev-parse", "HEAD"), bool(_git_value("status", "--porcelain"))


def case_directory(case: GoldenCase, output_root: Path, commit_id: str, dirty: bool) -> Path:
    """Return the deterministic path reserved for a case and worktree state."""
    revision = "worktree" if dirty else commit_id
    return output_root / revision / case.test_id


def enable_jax_x64() -> None:
    """Enable and verify x64 immediately before any JAX-dependent work begins."""
    # Keep this import function-local: importing this module must not initialize JAX.
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64 or jnp.asarray(1.0).dtype != jnp.float64:
        raise RuntimeError("Release 1 requires verified jax-x64 precision")


def _resolve_rscript() -> str:
    candidates = (os.environ.get("RSCRIPT"), shutil.which("Rscript"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("Rscript is required to generate a fixed R Golden Artifact")


def install_vendored_r_library(rscript: str, library: Path) -> None:
    """Install vendored sources into ``library`` without using installed gamlss packages."""
    library.mkdir(parents=True, exist_ok=True)
    sources = [REPO_ROOT / "R" / "gamlss.data", REPO_ROOT / "R" / "gamlss.dist", REPO_ROOT / "R" / "gamlss"]
    missing = [str(path) for path in sources if not (path / "DESCRIPTION").is_file()]
    if missing:
        raise RuntimeError("vendored R package source missing: " + ", ".join(missing))
    r_executable = Path(rscript).with_name("R.exe" if os.name == "nt" else "R")
    if not r_executable.is_file():
        raise RuntimeError(f"R executable required for vendored source installation: {r_executable}")
    for source in sources:
        subprocess.run(
            [str(r_executable), "CMD", "INSTALL", "--no-multiarch", "--library", str(library), str(source)],
            cwd=REPO_ROOT, text=True, capture_output=True, check=True,
        )


def _r_version(rscript: str) -> str:
    return subprocess.run(
        [rscript, "--vanilla", "-s", "-e", "cat(R.version.string)"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def build_manifest(case: GoldenCase, result_path: Path, rscript: str) -> dict[str, Any]:
    """Build the complete provenance manifest required for every artifact."""
    commit_id, dirty = git_provenance()
    manifest = {
        "schema_version": 1,
        "gamlss_source": {"version": _description_version(REPO_ROOT / "R" / "gamlss"), "path": "R/gamlss"},
        "gamlss_dist_source": {"version": _description_version(REPO_ROOT / "R" / "gamlss.dist"), "path": "R/gamlss.dist"},
        "r_version": _r_version(rscript),
        "python_version": sys.version.split()[0],
        "jax_version": _distribution_version("jax"),
        "jaxlib_version": _distribution_version("jaxlib"),
        "numpy_version": np.__version__,
        "scipy_version": _distribution_version("scipy"),
        "commit_id": commit_id,
        "dirty_worktree": dirty,
        "precision_policy": "jax-x64",
        "seed": case.seed,
        "formula": case.formula,
        "parameters": dict(case.parameters),
        "tolerances": dict(case.tolerances),
        "test_id": case.test_id,
        "result_path": str(result_path),
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError("manifest missing required fields: " + ", ".join(sorted(missing)))
    if not isinstance(manifest["seed"], int):
        raise ValueError("random case seed must be recorded as an integer")
    for source_key in ("gamlss_source", "gamlss_dist_source"):
        source = manifest[source_key]
        if not isinstance(source, Mapping) or not source.get("version") or not source.get("path"):
            raise ValueError(f"{source_key} must record both version and path")


def _run_r_no_case(rscript: str, library: Path, case: GoldenCase) -> list[float]:
    params = case.parameters
    values = params["x"]
    script = (
        ".libPaths(c(commandArgs(TRUE)[1], .libPaths())); "
        "library(gamlss.dist, lib.loc=commandArgs(TRUE)[1]); "
        f"set.seed({case.seed}); "
        f"values <- dNO(c({','.join(str(float(x)) for x in values)}), mu={float(params['mu'])}, sigma={float(params['sigma'])}); "
        "cat(jsonlite::toJSON(as.numeric(values), auto_unbox=TRUE))"
    )
    completed = subprocess.run(
        [rscript, "--vanilla", "-s", "-e", script, str(library)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    )
    return list(json.loads(completed.stdout))


def _markdown_result(result: Mapping[str, Any]) -> str:
    comparison = result["comparison"]
    return "\n".join((
        f"# R Golden Artifact: {result['case']['test_id']}", "",
        f"Status: **{result['status']}**", "",
        "## Comparison", "",
        "| Field | Value |", "| --- | ---: |",
        f"| max absolute error | {comparison['max_abs_error']:.3e} |",
        f"| tolerance | {comparison['tolerance']:.3e} |",
        f"| values compared | {comparison['count']} |", "",
        "The accompanying `manifest.json` contains full runtime provenance.", "",
    ))


def run_no_case(case: GoldenCase, output_root: Path) -> tuple[Path, Path, Path]:
    """Install vendored R packages, run a deterministic NO density case and write artifacts."""
    if case.family != "NO" or case.operation != "d":
        raise ValueError("task 0.3 runner currently supports only the fixed NO density case")
    # Precision is configured before this runner can import any JAX-dependent OmniLSS code.
    enable_jax_x64()
    rscript = _resolve_rscript()
    commit_id, dirty = git_provenance()
    result_dir = case_directory(case, output_root, commit_id, dirty)
    result_dir.mkdir(parents=True, exist_ok=True)
    library = result_dir / "r-library"
    install_vendored_r_library(rscript, library)
    manifest_path = result_dir / "manifest.json"
    result_path = result_dir / "result.json"
    markdown_path = result_dir / "result.md"
    manifest = build_manifest(case, result_path, rscript)
    r_values = np.asarray(_run_r_no_case(rscript, library, case), dtype=float)
    python_values = norm.pdf(np.asarray(case.parameters["x"], dtype=float), loc=float(case.parameters["mu"]), scale=float(case.parameters["sigma"]))
    if not np.isfinite(r_values).all() or not np.isfinite(python_values).all() or r_values.shape != python_values.shape:
        raise RuntimeError("fixed NO result is non-finite or shape-incompatible")
    tolerance = float(case.tolerances["absolute"])
    max_abs_error = float(np.max(np.abs(r_values - python_values)))
    result: dict[str, Any] = {
        "case": asdict(case), "status": "passed" if max_abs_error <= tolerance else "failed",
        "r_result": {"density": r_values.tolist()},
        "python_result": {"density": python_values.tolist()},
        "comparison": {"count": int(r_values.size), "max_abs_error": max_abs_error, "tolerance": tolerance},
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_result(result), encoding="utf-8")
    if result["status"] != "passed":
        raise RuntimeError(f"NO comparison exceeded tolerance: {max_abs_error} > {tolerance}")
    return result_path, markdown_path, manifest_path


def default_no_case() -> GoldenCase:
    return GoldenCase(
        test_id="no-density-seed-20260722", family="NO", operation="d", seed=20260722,
        formula=None, parameters={"x": [-2.0, -0.5, 0.0, 1.25, 3.0], "mu": 0.25, "sigma": 1.5},
        tolerances={"absolute": 1e-12},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "docs" / "validation" / "r-gamlss-release1")
    args = parser.parse_args(argv)
    for path in run_no_case(default_no_case(), args.output_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
