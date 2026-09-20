"""Deterministic command line runner for pytest-configured test suites."""

from __future__ import annotations

import argparse
import os
import importlib.util
import platform
import shutil
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

try:
    from tests.suites import SUITES, get_suite_modules, resolve_suite_name
except ModuleNotFoundError:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.suites import SUITES, get_suite_modules, resolve_suite_name


class RequiredRGate:
    """Track collection and runtime skips without importing test modules."""

    def __init__(self) -> None:
        self.skipped: list[str] = []

    def pytest_collectreport(self, report) -> None:
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report) -> None:
        if report.skipped:
            self.skipped.append(report.nodeid)


class CollectionReport:
    """Capture collection diagnostics for CI and local triage."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.skipped: list[str] = []
        self.items = 0

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.errors.append(str(report.nodeid))
        if report.skipped:
            self.skipped.append(str(report.nodeid))

    def pytest_collection_finish(self, session) -> None:
        self.items = len(session.items)



def configure_environment() -> None:
    existing = os.environ.get("PYTHONPATH")
    if existing:
        os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + existing
    else:
        os.environ["PYTHONPATH"] = str(SRC)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run named pytest suites for OmniLSS.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        help="Suite name to run. Repeat to combine multiple suites.",
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Run a specific test module, for example tests.test_r_consistency_zip.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available suites and exit.",
    )
    parser.add_argument("--check-env", action="store_true", help="Print the local test environment and exit.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use verbosity=2.",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure.",
    )
    parser.add_argument(
        "--require-r",
        action="store_true",
        dest="require_r",
        help=(
            "Phase gate (fail-closed): any skipped test or module is treated "
            "as a failure.  Required-R suites must not pass with skips or "
            "smoke-only results (DEVELOPMENT_PLAN 2.6 / .kiro 1.6)."
        ),
    )
    parser.add_argument("--collect-only", action="store_true", help="Collect selected modules without running tests.")
    parser.add_argument("--collect-report", type=Path, help="Write JSON collection diagnostics to this path.")
    return parser


def list_suites() -> int:
    print("Available test suites:")
    for name, modules in SUITES.items():
        print(f"- {name}: {len(modules)} modules")
    return 0


def check_environment() -> int:
    """Report actionable Python, JAX, R, and WSL environment information."""
    print(f"python: {sys.executable}")
    print(f"python_version: {platform.python_version()}")
    print(f"pytest: {__import__('pytest').__version__ if importlib.util.find_spec('pytest') else 'MISSING'}")
    try:
        import jax
        print(f"jax: {jax.__version__}")
        print(f"jax_backend: {jax.default_backend()}")
        print(f"jax_devices: {', '.join(str(device) for device in jax.devices())}")
    except Exception as exc:
        print(f"jax: ERROR ({exc})")
    rscript = shutil.which("Rscript")
    print(f"rscript: {rscript or 'MISSING'}")
    if rscript:
        expr = "cat(R.version.string, '\\n'); for (p in c('gamlss','gamlss.dist','jsonlite')) cat(p, '=', requireNamespace(p, quietly=TRUE), '\\n', sep='')"
        result = subprocess.run([rscript, "-e", expr], capture_output=True, text=True, check=False)
        print(result.stdout.rstrip())
    distro = "debianjaxgpu"
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    print(f"wsl: {wsl or 'MISSING'}")
    if not wsl:
        print("wsl_status: unavailable")
    else:
        result = subprocess.run([wsl, "-d", distro, "--", "python", "-c", "import jax; print(jax.devices())"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("wsl_jax_status: available")
            print(f"wsl_jax_devices: {result.stdout.strip()}")
        elif "not found" in (result.stderr or "").lower() or "no distribution" in (result.stderr or "").lower():
            print("wsl_jax_status: distribution-unavailable")
        else:
            print(f"wsl_jax_status: error ({result.stderr.strip() or result.returncode})")
    print(f"platform: {platform.platform()}")
    return 0


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def resolve_requested_modules(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    suites = args.suites or ([] if args.modules else ["all"])
    for suite_name in suites:
        requested.extend(get_suite_modules(suite_name))
    if args.modules:
        requested.extend(args.modules)
    return dedupe(requested)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        return list_suites()
    if args.check_env:
        return check_environment()

    configure_environment()
    try:
        modules = resolve_requested_modules(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    if not modules:
        print("ERROR: no test modules selected.")
        return 5
    missing = [module for module in modules if not ROOT.joinpath(*module.split('.')).with_suffix('.py').exists()]
    if missing:
        print("ERROR: selected test modules do not exist:")
        for module in missing:
            print(f"- {module}")
        return 4
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is required; install the project's dev dependencies.")
        return 1
    pytest_args = [str(ROOT.joinpath(*module.split('.')).with_suffix('.py')) for module in modules]
    if args.verbose:
        pytest_args.append("-v")
    if args.failfast:
        pytest_args.append("-x")
    if args.collect_only:
        pytest_args.append("--collect-only")
    if args.require_r:
        pytest_args.append("-rs")
    gate = RequiredRGate()
    collection = CollectionReport()
    plugins = [collection]
    if args.require_r:
        plugins.append(gate)
    exit_code = int(pytest.main(pytest_args, plugins=plugins))
    if args.collect_report:
        args.collect_report.parent.mkdir(parents=True, exist_ok=True)
        args.collect_report.write_text(json.dumps({
            "suite": [resolve_suite_name(name) for name in args.suites] if args.suites else None,
            "modules": modules,
            "module_count": len(modules),
            "collected_items": collection.items,
            "collection_errors": collection.errors,
            "collection_skips": collection.skipped,
            "pytest_exit_code": exit_code,
            "status": "passed" if exit_code == 0 and not collection.errors else "failed",
        }, indent=2) + "\n", encoding="utf-8")
    if args.require_r and gate.skipped:
        print("ERROR: --require-r is fail-closed: skipped tests/modules are not allowed:")
        for nodeid in gate.skipped:
            print(f"- {nodeid}")
        if exit_code in (0, 5):
            exit_code = 1
    if args.suites:
        requested_names = ", ".join(resolve_suite_name(name) for name in args.suites)
    elif args.modules:
        requested_names = "modules-only"
    else:
        requested_names = "all"
    print("")
    print(f"Executed suites/modules: {requested_names}")
    print(f"Module count: {len(modules)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
