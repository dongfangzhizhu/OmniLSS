"""Tests for release_check helper script."""

from __future__ import annotations

import io
import contextlib
import importlib.util
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "release_check.py"
_SPEC = importlib.util.spec_from_file_location("release_check_tool", _TOOL_PATH)
release_check = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(release_check)


def test_release_check_reports_missing_build_module(monkeypatch) -> None:
    """release_check should return non-zero and print guidance when build fails."""

    def fake_run(cmd):
        if cmd[-2:] == ["-m", "build"]:
            return 1
        return 0

    monkeypatch.setattr(release_check, "_run", fake_run)
    monkeypatch.setattr(release_check.shutil, "which", lambda _: "/usr/bin/python")

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = release_check.main()

    assert code == 1
    assert "install with: pip install build" in stderr.getvalue()
