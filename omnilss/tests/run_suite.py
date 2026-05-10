"""Command line runner for named unittest suites."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

from tests.suites import SUITES, get_suite_modules, resolve_suite_name


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        return list_suites()

    configure_environment()
    modules = resolve_requested_modules(args)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module in modules:
        suite.addTests(loader.loadTestsFromName(module))

    runner = unittest.TextTestRunner(
        verbosity=2 if args.verbose else 1,
        failfast=args.failfast,
    )
    result = runner.run(suite)
    if args.suites:
        requested_names = ", ".join(resolve_suite_name(name) for name in args.suites)
    elif args.modules:
        requested_names = "modules-only"
    else:
        requested_names = "all"
    print("")
    print(f"Executed suites/modules: {requested_names}")
    print(f"Module count: {len(modules)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
