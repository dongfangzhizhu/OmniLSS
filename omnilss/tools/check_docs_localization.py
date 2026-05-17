"""Validate bilingual documentation pairs under ``docs/``.

The project documentation policy uses English as the default file and stores the
Chinese counterpart next to it with ``_cn`` appended to the stem.  Each pair must
link to the other file so readers can switch languages easily.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"


def iter_english_docs(docs_root: Path = DOCS_ROOT):
    """Yield English/default Markdown docs that require a ``_cn`` pair."""

    for path in sorted(docs_root.rglob("*.md")):
        if path.stem.endswith("_cn"):
            continue
        yield path


def validate_docs_localization(docs_root: Path = DOCS_ROOT) -> list[str]:
    """Return localization-policy violations for Markdown docs."""

    errors: list[str] = []
    for english_path in iter_english_docs(docs_root):
        chinese_path = english_path.with_name(f"{english_path.stem}_cn.md")
        if not chinese_path.exists():
            errors.append(f"missing Chinese document: {chinese_path.relative_to(docs_root)}")
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
