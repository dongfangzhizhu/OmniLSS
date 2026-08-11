"""Phase 2.6: run_suite --require-r phase-gate behaviour (fail-closed)."""

from __future__ import annotations

import argparse
import unittest
import unittest.mock as mock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import run_suite  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        suites=None,
        modules=["tests.test_operations"],
        list=False,
        verbose=False,
        failfast=False,
        require_r=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _call_main(argv: list[str]) -> int:
    """Run main() with explicit argv and a stubbed unittest runner."""
    with mock.patch.object(run_suite, "unittest") as uni:
        uni.defaultTestLoader = unittest.defaultTestLoader
        uni.TestSuite = unittest.TestSuite
        uni.TextTestRunner = mock.MagicMock()
        return run_suite.main(argv)


def _fake_result(skipped=(), successful=True):
    result = mock.MagicMock()
    result.skipped = list(skipped)
    result.wasSuccessful.return_value = successful
    return result


def _stub_runner(skipped=(), successful=True):
    """Patch TextTestRunner so .run() returns a fake result."""
    fake = mock.MagicMock()
    fake.skipped = list(skipped)
    fake.wasSuccessful.return_value = successful
    return mock.patch.object(
        run_suite.unittest, "TextTestRunner",
        return_value=mock.MagicMock(run=mock.MagicMock(return_value=fake)),
    )


def test_require_r_fails_when_tests_skipped(capsys) -> None:
    """--require-r treats any skipped test as a gate failure."""
    with _stub_runner(skipped=[("test", "R unavailable")]):
        code = run_suite.main(["--module", "tests.test_operations", "--require-r"])
    assert code == 1
    out = capsys.readouterr().out
    assert "skipped" in out.lower()
    assert "not allowed" in out


def test_require_r_passes_without_skips() -> None:
    """--require-r passes when nothing is skipped and all tests succeed."""
    with _stub_runner(skipped=[]):
        code = run_suite.main(["--module", "tests.test_operations", "--require-r"])
    assert code == 0


def test_without_require_r_skips_are_allowed() -> None:
    """Without --require-r, skipped tests do not fail the run."""
    with _stub_runner(skipped=[("test", "R unavailable")]):
        code = run_suite.main(["--module", "tests.test_operations"])
    assert code == 0


def test_require_r_fails_when_module_load_skipped(capsys) -> None:
    """--require-r fails when a module is skipped during loading."""
    loader = mock.MagicMock()

    def _load_skip(name):
        raise unittest.SkipTest("R unavailable")

    loader.loadTestsFromName = _load_skip
    with mock.patch.object(run_suite.unittest, "defaultTestLoader", loader):
        with _stub_runner(skipped=[]):
            code = run_suite.main(["--module", "tests.test_operations", "--require-r"])
    assert code == 1
    out = capsys.readouterr().out
    assert "fail-closed" in out
