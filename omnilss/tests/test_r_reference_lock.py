# SPDX-License-Identifier: GPL-3.0-or-later
"""R reference lock + installer (D2.308/R1).

Pins the tarballs CI installs for Group B and verifies SHA256 before use, with
an optional mirror fallback.  These tests never touch the network.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import install_r_reference as rref  # noqa: E402

LOCK = REPO / "docs" / "development" / "r-reference-lock.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_lock_matches_reference_versions() -> None:
    lock = rref.load_lock(LOCK)
    assert lock["r_version"] == "4.5.0"
    assert [p["name"] for p in lock["packages"]] == ["gamlss.data", "gamlss.dist", "gamlss"]
    assert [p["version"] for p in lock["packages"]] == ["6.0-6", "6.1-1", "5.5-0"]
    for package in lock["packages"]:
        assert package["url"].endswith(f"{package['name']}_{package['version']}.tar.gz")
        assert len(package["sha256"]) == 64
        int(package["sha256"], 16)


def _package(path: Path, sha: str | None = None, mirror: Path | None = None) -> dict:
    return {
        "name": "demo",
        "version": "1.0",
        "url": path.as_uri(),
        "sha256": sha or _sha(path.read_bytes()),
        "mirror_url": mirror.as_uri() if mirror else None,
    }


def test_materialize_local_file(tmp_path: Path) -> None:
    src = tmp_path / "demo_1.0.tar.gz"
    src.write_bytes(b"tarball")
    paths = rref.materialize({"packages": [_package(src)]}, tmp_path / "out")
    assert [p.name for p in paths] == ["demo_1.0.tar.gz"]
    assert paths[0].read_bytes() == b"tarball"


def test_hash_mismatch_falls_back_to_mirror(tmp_path: Path) -> None:
    bad = tmp_path / "bad_1.0.tar.gz"
    good = tmp_path / "good_1.0.tar.gz"
    bad.write_bytes(b"tampered")
    good.write_bytes(b"tarball")
    paths = rref.materialize(
        {"packages": [_package(bad, sha=_sha(b"tarball"), mirror=good)]},
        tmp_path / "out",
    )
    assert paths[0].read_bytes() == b"tarball"


def test_hash_mismatch_without_mirror_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad_1.0.tar.gz"
    bad.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        rref.materialize(
            {"packages": [_package(bad, sha=_sha(b"expected"))]},
            tmp_path / "out",
        )


def test_load_lock_rejects_invalid_sha(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "packages": [
                    {"name": "x", "version": "1", "url": "https://example/x.tar.gz", "sha256": "short"}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256"):
        rref.load_lock(lock)
