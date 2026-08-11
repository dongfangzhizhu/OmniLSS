"""Command line runner for named unittest suites."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

try:
    from tests.suites import SUITES, get_suite_modules, resolve_suite_name
except ModuleNotFoundError:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.suites import SUITES, get_suite_modules, resolve_suite_name


@dataclass
class ModuleLoadIssue:
    module: str
    reason: str


def _is_skip_like_exception(exc: BaseException) -> bool:
    if isinstance(exc, unittest.SkipTest):
        return True
    name = exc.__class__.__name__.lower()
    return "skip" in name



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
        description="Run named unittest suites for OmniLSS.",
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
    return parser


def list_suites() -> int:
    print("Available test suites:")
    for name, modules in SUITES.items():
        print(f"- {name}: {len(modules)} modules")
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

    configure_environment()
    modules = resolve_requested_modules(args)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    skipped_modules: list[ModuleLoadIssue] = []
    failed_modules: list[ModuleLoadIssue] = []
    for module in modules:
        try:
            suite.addTests(loader.loadTestsFromName(module))
        except BaseException as exc:  # pragma: no cover - defensive for pytest skip outcomes
            issue = ModuleLoadIssue(module=module, reason=f"{exc.__class__.__name__}: {exc}")
            if _is_skip_like_exception(exc):
                skipped_modules.append(issue)
                continue
            failed_modules.append(issue)

    if failed_modules:
        print("Failed to load test modules:")
        for issue in failed_modules:
            print(f"- {issue.module}: {issue.reason}")
        return 1

    if args.require_r and skipped_modules:
        print(
            "ERROR: --require-r is fail-closed but the following modules were "
            "skipped during loading (R unavailable?):"
        )
        for issue in skipped_modules:
            print(f"- {issue.module}: {issue.reason}")
        return 1

    runner = unittest.TextTestRunner(
        verbosity=2 if args.verbose else 1,
        failfast=args.failfast,
    )
    result = runner.run(suite)
    if args.require_r and result.skipped:
        print(
            "ERROR: --require-r is fail-closed but {} test(s) were skipped "
            "(R unavailable?):".format(len(result.skipped))
        )
        for test, reason in result.skipped[:20]:
            print(f"- {test}: {reason}")
        print("Phase gate NOT passed: skips are not allowed under --require-r.")
        return 1
    if args.suites:
        requested_names = ", ".join(resolve_suite_name(name) for name in args.suites)
    elif args.modules:
        requested_names = "modules-only"
    else:
        requested_names = "all"
    print("")
    print(f"Executed suites/modules: {requested_names}")
    print(f"Module count: {len(modules)}")
    if skipped_modules:
        print("Skipped test modules during loading:")
        for issue in skipped_modules:
            print(f"- {issue.module}: {issue.reason}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
