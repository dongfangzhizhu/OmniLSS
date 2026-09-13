"""Real pytest execution contracts for the named suite runner."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from unittest import mock

import pytest
from tests import run_suite


@pytest.mark.parametrize("source,required,expected", [
    ("def test_ok(): assert True", False, 0),
    ("def test_bad(): assert False", False, 1),
    ("import pytest\ndef test_skip(): pytest.skip('R unavailable')", False, 0),
    ("import pytest\ndef test_skip(): pytest.skip('R unavailable')", True, 1),
    ("import pytest\npytest.skip('R unavailable', allow_module_level=True)", True, 1),
    ("def test_ok(): assert True", True, 0),
    ("VALUE = 1", False, 5),
    ("raise RuntimeError('collection failed')", False, 2),
])
def test_real_execution(tmp_path, source, required, expected):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(source + "\n", encoding="utf-8")
    argv = ["--module", "tests.test_sample"]
    if required:
        argv.append("--require-r")
    code = (
        "from pathlib import Path; from tests import run_suite; "
        f"run_suite.ROOT = Path({str(tmp_path)!r}); "
        f"raise SystemExit(run_suite.main({argv!r}))"
    )
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=str(run_suite.ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if required and "skip" in source:
        assert "fail-closed" in result.stdout


def test_selection_is_ordered_and_deduplicated():
    args = argparse.Namespace(suites=["quick"], modules=["tests.extra", "tests.extra"])
    with mock.patch.object(run_suite, "get_suite_modules", return_value=["tests.b", "tests.a"]):
        assert run_suite.resolve_requested_modules(args) == ["tests.b", "tests.a", "tests.extra"]


def test_flags_and_absolute_module_paths():
    with mock.patch("pytest.main", return_value=0) as main:
        assert run_suite.main(["--module", "tests.test_operations", "--verbose", "--failfast"]) == 0
    assert main.call_args.args[0] == [str(run_suite.ROOT / "tests/test_operations.py"), "-v", "-x"]


def test_missing_module_returns_usage_error():
    assert run_suite.main(["--module", "tests.does_not_exist"]) == 4


def test_empty_selection_does_not_discover_everything():
    with mock.patch.object(run_suite, "resolve_requested_modules", return_value=[]):
        assert run_suite.main([]) == 5