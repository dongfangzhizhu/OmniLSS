"""The coverage report must describe the families it claims to describe (D2.78).

r-export-coverage.json is the project single source of truth for capability.
Three of its fields were dead: has_dqr / has_fitting / has_smoothing were
computed with getattr(cap, "dqr", False) and friends, and FamilyCapability has
no such attributes -- its evidence lives in a features mapping keyed by
rs_fit / smooth_terms / ....  So the report said all 146 families lacked
d/q/r, including NO, GA and PO, which are the production-safe baseline.

The tests below pin the field to the family, not to a missing attribute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "omnilss" / "src"))
sys.path.insert(0, str(REPO / "tools"))

from omnilss.distribution_registry import resolve  # noqa: E402

COVERAGE = REPO / "docs" / "development" / "r-export-coverage.json"


@pytest.fixture(scope="module")
def families() -> dict:
    if not COVERAGE.exists():  # pragma: no cover - artifact is committed
        pytest.skip("coverage artifact not generated")
    return json.loads(COVERAGE.read_text(encoding="utf-8"))["families"]


def _kernels_are_real(name: str) -> bool:
    family = resolve(name)
    kernels = [getattr(family, key, None) for key in ("d", "p", "q")]
    if any(kernel is None for kernel in kernels):
        return False
    return not any(
        "placeholder" in getattr(kernel, "__name__", "") for kernel in kernels
    )


def test_the_report_does_not_say_every_family_lacks_dqr(families):
    """"All false" was the bug, not a property of the library."""
    with_dqr = [name for name, info in families.items() if info.get("has_dqr")]
    assert len(with_dqr) >= 140, len(with_dqr)


@pytest.mark.parametrize("family", ["NO", "GA", "PO", "BI", "IGA", "ZABI", "NET"])
def test_baseline_families_advertise_dqr(families, family):
    assert families[family]["has_dqr"] is True


def test_the_flag_follows_the_family_not_a_missing_attribute(families):
    """Every entry must agree with the kernels the registry hands out."""
    for name, info in families.items():
        assert info["has_dqr"] is _kernels_are_real(name), name


def test_fitting_is_advertised_for_every_registered_family(families):
    """RS fitting is available for all 146; the field used to say so for none."""
    assert all(info["has_fitting"] is True for info in families.values())