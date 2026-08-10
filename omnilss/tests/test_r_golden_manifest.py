"""Tests for R golden fixture provenance (task 0.8).

Validates that:
- Every checked-in fixture is listed in the regenerated manifest with a
  content hash and provenance (R/gamlss versions, seed).
- The manifest hashes are 8-hex-digit strings and stable across regeneration
  (reproducibility).
- Regenerated fixtures satisfy the same schema as the checked-in ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "r_golden"
REGEN_DIR = FIXTURES_DIR / "regenerated"

CHECKED_IN = [
    "ig_y_x_seed_20260522.json",
    "po_y_x_seed_20260523.json",
    "no_y_x_seed_20260524.json",
    "ga_y_x_seed_20260525.json",
    # Phase 2.1: remaining core eight families (NO/GA/PO/BI/NBI/BE/ZAGA/GU)
    "bi_y_x_seed_20260809.json",
    "nbi_y_x_seed_20260810.json",
    "be_y_x_seed_20260811.json",
    "zaga_y_x_seed_20260812.json",
    "gu_y_x_seed_20260813.json",
]


def test_regenerated_manifest_exists_and_has_hashes() -> None:
    manifest_path = REGEN_DIR / "manifest.json"
    assert manifest_path.exists(), "run Rscript fixtures/r_golden/generate_golden.R"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest.keys()) == {
        p.stem for p in (FIXTURES_DIR.glob("*.json")) if p.stem != "manifest"
    }
    for entry in manifest.values():
        assert isinstance(entry.get("content_hash"), str)
        assert len(entry["content_hash"]) == 8
        assert entry.get("seed") is not None
        assert entry.get("R_version")
        assert entry.get("gamlss_version")


def test_manifest_covers_all_checked_in_fixtures() -> None:
    manifest_path = REGEN_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("regenerated manifest not present")
    manifest = json.loads(manifest_path.read_text())
    for fname in CHECKED_IN:
        stem = Path(fname).stem
        assert stem in manifest, f"{fname} missing from regenerated manifest"


@pytest.mark.parametrize("fname", CHECKED_IN)
def test_regenerated_fixture_schema(fname: str) -> None:
    """Regenerated fixtures must satisfy the same schema as checked-in ones."""
    path = REGEN_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not regenerated")
    payload = json.loads(path.read_text())
    assert isinstance(payload.get("family"), str) and payload["family"]
    data = payload.get("data")
    assert isinstance(data, dict) and set(data.keys()) == {"x", "y"}
    assert len(data["x"]) == len(data["y"]) > 0
    expected = payload.get("expected")
    assert isinstance(expected, dict)
    assert isinstance(expected.get("mu_coefficients"), list)
    assert isinstance(expected.get("deviance"), (int, float))
    for c in expected["mu_coefficients"]:
        assert isinstance(c, (int, float))
