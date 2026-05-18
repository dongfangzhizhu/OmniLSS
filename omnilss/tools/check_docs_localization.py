"""Validate bilingual Markdown documentation pairs.

The project documentation policy uses English as the default file and stores the
Chinese counterpart next to it with ``_cn`` appended to the stem.  Each pair must
link to the other file so readers can switch languages easily.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT
EXCLUDED_PARTS = {
    ".git",
    ".github",
    "site",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}
_CN_NAV_PATTERN = re.compile(r"[\w./-]+_cn\.md")


def iter_english_docs(docs_root: Path = DOCS_ROOT):
    """Yield English/default Markdown docs that require a ``_cn`` pair."""

    for path in sorted(docs_root.rglob("*.md")):
        if path.stem.endswith("_cn"):
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(docs_root).parts):
            continue
        yield path


def _validate_mkdocs_default_nav(docs_root: Path) -> list[str]:
    """Ensure the default MkDocs navigation remains English-only."""

    mkdocs_path = docs_root / "mkdocs.yml"
    if not mkdocs_path.exists():
        return []

    errors: list[str] = []
    for line_no, line in enumerate(
        mkdocs_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        for match in _CN_NAV_PATTERN.findall(line):
            errors.append(
                "Chinese document listed in default MkDocs nav "
                f"at mkdocs.yml:{line_no}: {match}; link from the English page instead"
            )
    return errors


def validate_docs_localization(docs_root: Path = DOCS_ROOT) -> list[str]:
    """Return localization-policy violations for project Markdown docs."""

    errors: list[str] = []
    for english_path in iter_english_docs(docs_root):
        chinese_path = english_path.with_name(f"{english_path.stem}_cn.md")
        if not chinese_path.exists():
            errors.append(
                f"missing Chinese document: {chinese_path.relative_to(docs_root)}"
            )
            continue

        english_text = english_path.read_text(encoding="utf-8")
        chinese_text = chinese_path.read_text(encoding="utf-8")
        if chinese_path.name not in english_text:
            errors.append(
                "missing Chinese switch link in "
                f"{english_path.relative_to(docs_root)} -> {chinese_path.name}"
            )
        if english_path.name not in chinese_text:
            errors.append(
                "missing English switch link in "
                f"{chinese_path.relative_to(docs_root)} -> {english_path.name}"
            )
    errors.extend(_validate_mkdocs_default_nav(docs_root))
    return errors


def main() -> int:
    errors = validate_docs_localization()
    if errors:
        print("Documentation localization check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Documentation localization check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
