"""Tests for docs bilingual localization policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "check_docs_localization.py"
)
_SPEC = importlib.util.spec_from_file_location("check_docs_localization", _TOOL_PATH)
check_docs_localization = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(check_docs_localization)


def test_docs_have_chinese_counterparts_and_language_switch_links() -> None:
    errors = check_docs_localization.validate_docs_localization()
    assert errors == []


def test_default_mkdocs_nav_rejects_chinese_documents(tmp_path) -> None:
    english = tmp_path / "index.md"
    chinese = tmp_path / "index_cn.md"
    english.write_text("# Home\n\n[中文版本](index_cn.md)\n", encoding="utf-8")
    chinese.write_text("# 首页\n\n[English version](index.md)\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text(
        "nav:\n  - Home: index.md\n  - 首页: index_cn.md\n",
        encoding="utf-8",
    )

    errors = check_docs_localization.validate_docs_localization(tmp_path)

    assert any(
        "Chinese document listed in default MkDocs nav" in error for error in errors
    )
