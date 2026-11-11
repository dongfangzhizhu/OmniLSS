"""P0-2: legacy docs must not read as current completion / blanket R-equivalence.

The deliverable plan (section 22.6 / P0-2) requires calibrating the stale
"63% complete" / "numerically equivalent to R" wording in the old snapshot
docs so they only make validated-evidence-scoped claims, point at the
regenerable coverage artifacts, and mark ``gamlss2`` OUT OF SCOPE.  These
tests pin those calibrations in place so a later edit cannot silently
reintroduce a misleading headline number or equivalence claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.coverage_scope import OUT_OF_SCOPE_PACKAGES  # noqa: E402

DOCS = ROOT / "docs"
AUTH_ARTIFACT = "docs/development/r-symbol-coverage.json"
DIFF_ARTIFACT = "docs/development/symbol-coverage-diff.json"

#: legacy docs that used to carry a bare completion rate or blanket
#: R-equivalence claim; each must now be calibrated in place.
CALIBRATED_DOCS = (
    DOCS / "migration-progress.md",
    DOCS / "commercialization-assessment.md",
    DOCS / "tutorials" / "tutorial-r-consistency.md",
)


@pytest.mark.parametrize("path", CALIBRATED_DOCS, ids=lambda p: p.name)
def test_legacy_doc_is_calibrated(path: Path):
    text = path.read_text(encoding="utf-8")
    # points at the regenerable, authoritative artifacts instead of a
    # hard-coded headline number that would drift again
    assert AUTH_ARTIFACT in text, f"{path.name} does not cite the authoritative artifact"
    # explicitly disclaims being a current completion rate / blanket claim
    assert any(
        k in text
        for k in ("历史快照", "非当前完成率", "P0-2", "非全库", "blanket", "不是完成率", "not a completion")
    ), f"{path.name} carries no in-place calibration notice"
    # gamlss2 out-of-scope declaration (P0-2 acceptance requirement)
    assert "gamlss2" in text, f"{path.name} does not name gamlss2"
    assert "OUT OF SCOPE" in text or "out-of-scope" in text or "out_of_scope" in text, (
        f"{path.name} does not declare gamlss2 out of scope"
    )
    # must not hard-code a stale headline figure as a current rate
    assert "698 / missing 382" not in text, (
        f"{path.name} still hard-codes the 2026-11-05 count as current"
    )


def test_migration_progress_statistics_section_disclaims_63_percent():
    text = (DOCS / "migration-progress.md").read_text(encoding="utf-8")
    head = text.split("## 4. 总体进度统计", 1)[1].split("### 4.1", 1)[0]
    # the calibration banner sits between the section title and the first table
    assert "历史快照" in head
    assert "63%" in head  # names the misleading number it disclaims
    assert "不是当前完成率" in head
    assert DIFF_ARTIFACT in text or AUTH_ARTIFACT in text


def test_commercialization_no_blanket_equivalence_claim():
    text = (DOCS / "commercialization-assessment.md").read_text(encoding="utf-8")
    section = text.split("## 1. 价值主张", 1)[1].split("## 2.", 1)[0]
    # the old bare "与 R `gamlss` 数值等价" is gone; now evidence-scoped
    assert "提供与 R `gamlss` 数值等价" not in section
    assert "validated" in section or "已验证" in section
    assert "非全库 blanket 等价" in section or "非全库" in section


def test_tutorial_no_blanket_equivalence_headline():
    text = (DOCS / "tutorials" / "tutorial-r-consistency.md").read_text(encoding="utf-8")
    first_para = text.split("## 验证体系", 1)[0]
    assert "建立在**与 R `gamlss` 数值等价**的可验证承诺" not in first_para
    assert "validated" in first_para
    assert "gamlss2" in first_para


def test_core_audit_json_carries_reproducible_calibration():
    """all_ok is a path-existence result, not a completion/equivalence claim."""
    from importlib.util import module_from_spec, spec_from_file_location

    audit_path = ROOT / "omnilss" / "tools" / "audit_gamlss_migration.py"
    spec = spec_from_file_location("audit_gamlss_migration_p02", audit_path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    # register before exec: @dataclass resolves string annotations via
    # sys.modules[cls.__module__] (same as tests/test_audit_gamlss_migration.py)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = mod.build_audit()

    cal = payload["calibration"]
    assert "NOT a completion rate" in cal["note"]
    assert "NOT R numerical-equivalence" in cal["note"]
    assert AUTH_ARTIFACT in cal["authoritative_artifacts"]
    assert cal["out_of_scope_packages"] == sorted(OUT_OF_SCOPE_PACKAGES)
    assert cal["gamlss2_out_of_scope"] is True
    # plan_doc must be repo-relative + POSIX: regenerable on any host
    assert not payload["plan_doc"].startswith(("/", "D:", "C:"))
    assert "\\" not in payload["plan_doc"]

    # the checked-in artifact must match what the generator emits (no drift)
    committed = DOCS / "development" / "gamlss-core-migration-audit-2026-05-22.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == payload
