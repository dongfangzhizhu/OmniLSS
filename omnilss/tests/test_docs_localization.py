"""Tests for docs bilingual localization policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_docs_localization.py"
_SPEC = importlib.util.spec_from_file_location("check_docs_localization", _TOOL_PATH)
check_docs_localization = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(check_docs_localization)


def test_docs_have_chinese_counterparts_and_language_switch_links() -> None:
    errors = check_docs_localization.validate_docs_localization()
    assert errors == []
