"""Validate bilingual Markdown documentation pairs (scoped policy).

Policy (2026-09-26 structure cleanup):

1. Root user-facing docs (README/CHANGELOG/CONTRIBUTING/CODE_OF_CONDUCT) and
   every English page referenced from the default ``mkdocs.yml`` nav must have
   a ``_cn`` sibling with bidirectional language-switch links.
2. Any existing ``*_cn.md`` file must have its English counterpart and must
   cross-link both ways (orphan-pair check).
3. The default mkdocs nav must not reference Chinese documents, and every nav
   target must exist.

Internal development notes, generated reports, session notes and third-party
trees are exempt from rule 1; they are governed by the structure rules in
``docs/development/deliverable-plan-2026-09-26.md`` section 7.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BILINGUAL_ROOT_DOCS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
)

# Third-party / generated / local-only trees exempt from bilingual pairing.
EXCLUDED_PARTS = {
    ".git",
    ".github",
    ".devcontainer",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".kiro",
    ".pytest_cache",
    ".reasonix",
    ".venv",
    "R",
    "__pycache__",
    "debianjaxgpu",
    "gamlss-python",
    "mnt",
    "node_modules",
    "ondil",
    "site",
    "tmp",
    "venv",
}

# Project doc roots scanned for orphan *_cn.md files.
_PAIR_SCAN_ROOTS = ("docs", "tutorials", "benchmarks", "examples", "paper", "omnilss")

_CN_NAV_PATTERN = re.compile(r"[\w./-]+_cn\.md")
_NAV_REF_PATTERN = re.compile(r"([A-Za-z0-9_./-]+\.md)")


def _rel(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_PARTS for part in parts)


def _nav_refs(root: Path) -> list[str]:
    mkdocs = root / "mkdocs.yml"
    if not mkdocs.exists():
        return []
    return _NAV_REF_PATTERN.findall(mkdocs.read_text(encoding="utf-8"))


def _resolve_nav_target(root: Path, ref: str) -> Path | None:
    for base in (root / "docs", root):
        candidate = (base / ref).resolve()
        if candidate.exists():
            return candidate
    return None


def _check_pair(english: Path, chinese: Path, root: Path, errors: list[str]) -> None:
    if not chinese.exists():
        errors.append(f"missing Chinese document: {_rel(chinese, root)}")
        return
    english_text = english.read_text(encoding="utf-8", errors="replace")
    chinese_text = chinese.read_text(encoding="utf-8", errors="replace")
    if chinese.name not in english_text:
        errors.append(
            f"missing Chinese switch link in {_rel(english, root)} -> {chinese.name}"
        )
    if english.name not in chinese_text:
        errors.append(
            f"missing English switch link in {_rel(chinese, root)} -> {english.name}"
        )


def required_english_docs(root: Path) -> list[Path]:
    """Return English docs that must have a bilingual counterpart."""
    required: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            required.append(resolved)

    for name in BILINGUAL_ROOT_DOCS:
        candidate = root / name
        if candidate.exists():
            add(candidate)

    for ref in _nav_refs(root):
        if ref.endswith("_cn.md"):
            continue
        target = _resolve_nav_target(root, ref)
        if target is not None:
            add(target)

    return required


def _validate_mkdocs_default_nav(root: Path) -> list[str]:
    """Ensure the default MkDocs navigation remains English-only."""
    mkdocs_path = root / "mkdocs.yml"
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
        for ref in _NAV_REF_PATTERN.findall(line):
            if ref.endswith("_cn.md"):
                continue
            if _resolve_nav_target(root, ref) is None:
                errors.append(
                    f"missing mkdocs nav target at mkdocs.yml:{line_no}: {ref}"
                )
    return errors


def validate_docs_localization(docs_root: Path = ROOT) -> list[str]:
    """Return localization-policy violations for project Markdown docs."""
    root = Path(docs_root)
    errors: list[str] = []

    for english_path in required_english_docs(root):
        chinese_path = english_path.with_name(f"{english_path.stem}_cn.md")
        _check_pair(english_path, chinese_path, root, errors)

    # Orphan / broken *_cn.md pairs across project doc roots.
    for scan_root in _PAIR_SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for chinese_path in sorted(base.rglob("*_cn.md")):
            if _excluded(chinese_path, root):
                continue
            english_path = chinese_path.with_name(f"{chinese_path.stem[:-3]}.md")
            if not english_path.exists():
                errors.append(
                    f"missing English document for {_rel(chinese_path, root)}"
                )
                continue
            _check_pair(english_path, chinese_path, root, errors)

    # Root-level *_cn.md files (README_cn.md and friends).
    for chinese_path in sorted(root.glob("*_cn.md")):
        english_path = chinese_path.with_name(f"{chinese_path.stem[:-3]}.md")
        if not english_path.exists():
            errors.append(f"missing English document for {chinese_path.name}")
            continue
        _check_pair(english_path, chinese_path, root, errors)

    errors.extend(_validate_mkdocs_default_nav(root))

    # De-duplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for error in errors:
        if error not in seen:
            seen.add(error)
            unique.append(error)
    return unique


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
