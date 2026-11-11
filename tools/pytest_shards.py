"""Deterministic N-way pytest shard lists for the release gate (P0-4).

Replaces the ad-hoc, gitignored ``tmp/mk_shards.py`` used for local full-suite
runs with a committed, reviewable tool that the release workflow calls.

Design constraints (from the full-suite regression discipline in
``.workbuddy/memory/MEMORY.md``):

* **Whole-shard mode, not one-process-per-file.**  Each shard is a single pytest
  invocation over ~n/8 files so JAX / registry cold-compilation is paid once per
  shard; a single process per file repeatedly re-pays it and heavy R-bridge
  files then time out.
* **Round-robin assignment** (``files[index::n]``) spreads the few very slow
  files across shards instead of clustering them in one slow shard.
* **Repo-relative POSIX paths** so the same argfile works from the repo root on
  any OS; the workflow runs pytest from the repo root with
  ``PYTHONPATH=omnilss/src``.
* **Deterministic**: sorted file list, stable partition, verifiable union
  (no lost / duplicated / empty shards).  ``--check`` re-partitions and asserts
  the invariants without writing anything, so CI can gate on the tool itself.

Usage::

    python tools/pytest_shards.py --shards 8 --out-dir tmp/release-shards
    python tools/pytest_shards.py --shards 8 --check
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "omnilss" / "tests"
DEFAULT_SHARDS = 8
DEFAULT_OUT_DIR = ROOT / "tmp" / "release-shards"


def test_files() -> list[str]:
    """Sorted, deterministic list of every ``test_*.py`` (repo-relative POSIX)."""
    return sorted(p.relative_to(ROOT).as_posix() for p in TESTS.rglob("test_*.py"))


def build_shards(files: list[str], n: int) -> list[list[str]]:
    """Round-robin partition into ``n`` shards; ``shard[i] == files[i::n]``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [files[index::n] for index in range(n)]


def verify(files: list[str], n: int) -> list[list[str]]:
    """Return shards, asserting the union covers the suite exactly once."""
    shards = build_shards(files, n)
    flat = [f for shard in shards for f in shard]
    assert len(flat) == len(files), "shard union lost files"
    assert len(set(flat)) == len(files), "a file appears in more than one shard"
    assert set(flat) == set(files), "shard union != discovered test set"
    assert all(len(shard) > 0 for shard in shards), (
        f"empty shard: {n} shards > {len(files)} files"
    )
    return shards


def write_shards(shards: list[list[str]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, shard in enumerate(shards):
        path = out_dir / f"shard{index}.txt"
        path.write_text("\n".join(shard) + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def manifest(files: list[str], n: int) -> str:
    """Human-readable, deterministic summary (also a stability fingerprint)."""
    digest = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()[:12]
    shards = build_shards(files, n)
    sizes = ", ".join(f"s{i}={len(s)}" for i, s in enumerate(shards))
    return f"shards={n} files={len(files)} fingerprint={digest} [{sizes}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--check", action="store_true",
        help="verify partition invariants only; write nothing",
    )
    args = parser.parse_args(argv)

    files = test_files()
    if not files:
        print("error: no test_*.py discovered", file=__import__("sys").stderr)
        return 2
    try:
        shards = verify(files, args.shards)
    except (ValueError, AssertionError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1

    print(manifest(files, args.shards))
    if args.check:
        print("verify: OK (no shard written)")
        return 0
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    paths = write_shards(shards, out_dir)
    for index, (path, shard) in enumerate(zip(paths, shards)):
        print(f"  shard{index}: {len(shard):3d} files -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
