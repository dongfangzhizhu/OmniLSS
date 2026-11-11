"""P0-4: the 8-shard full-suite release gate is deterministic and wired in.

The development plan (section 22.6 / P0-4) fixes the "whole-suite 8-shard
parallel re-test" as a release gate, with independent model DBs, real-time log
output, ``git diff --check``, tiered collection, lint/type, and an isolated
performance-benchmark re-check.  These tests pin both the sharding tool and
the workflow wiring so the gate cannot silently degrade back into the fragile
single-process full-suite run.

The workflow is asserted as text (not via a YAML lib) so the acceptance test
never gets skipped just because PyYAML is absent in a given environment.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOL = ROOT / "tools" / "pytest_shards.py"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pytest_shards", TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# sharding tool: deterministic, lossless, valid paths
# ---------------------------------------------------------------------------
def test_tool_discovers_tests_as_repo_relative_posix_paths():
    mod = _load_tool()
    files = mod.test_files()
    assert files, "no test_*.py discovered"
    assert files == sorted(files), "file list is not sorted (non-deterministic)"
    for f in files:
        assert f.startswith("omnilss/tests/") and f.endswith(".py")
        assert "\\" not in f, "path is not POSIX"
        assert (ROOT / f).exists(), f"missing test file: {f}"


def test_eight_way_partition_is_lossless_and_balanced():
    mod = _load_tool()
    files = mod.test_files()
    shards = mod.verify(files, 8)
    assert len(shards) == 8
    flat = [f for s in shards for f in s]
    assert len(flat) == len(files)
    assert set(flat) == set(files)
    assert len(set(flat)) == len(files), "a file appears twice"
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1, "shards are unbalanced"
    # round-robin: shard i is exactly files[i::8]
    for i in range(8):
        assert shards[i] == files[i::8]


def test_partition_is_deterministic():
    mod = _load_tool()
    files = mod.test_files()
    assert mod.build_shards(files, 8) == mod.build_shards(files, 8)
    assert mod.manifest(files, 8) == mod.manifest(files, 8)


def test_verify_rejects_more_shards_than_files():
    mod = _load_tool()
    files = mod.test_files()
    try:
        mod.verify(files, len(files) + 1)
    except AssertionError:
        return  # expected: empty shards are rejected
    raise AssertionError("verify() should reject empty shards")


def test_cli_check_exits_zero():
    out = subprocess.run(
        [sys.executable, str(TOOL), "--shards", "8", "--check"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "verify: OK" in out.stdout
    assert "shards=8" in out.stdout


# ---------------------------------------------------------------------------
# release workflow wiring (text-level, dependency-free)
# ---------------------------------------------------------------------------
def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_gate_uses_eight_shard_matrix_not_single_process():
    text = _workflow()
    assert "sharded-suite:" in text
    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7]" in text
    assert "pytest_shards.py --shards 8 --out-dir tmp/release-shards" in text
    # whole-shard mode: one pytest process per shard via an @argfile
    assert "@tmp/release-shards/shard${{ matrix.shard }}.txt" in text
    # the old single-process full-suite run must be gone
    assert "python -m pytest tests/ -q" not in text


def test_each_shard_has_independent_db_and_realtime_output():
    text = _workflow()
    assert "OMNILSS_MODEL_DB_PATH" in text
    assert "model-db-shard-${{ matrix.shard }}.sqlite3" in text
    assert "tee tmp/shard-${{ matrix.shard }}.log" in text
    assert "set -o pipefail" in text


def test_static_release_gates_present():
    text = _workflow()
    for kw in (
        "git diff --check",
        "tiered_collection.py --mode collect --strict-skips --json",
        "pytest_shards.py --shards 8 --check",
        "ruff check src/omnilss/",
        "black --check src/omnilss/",
        "mypy src/omnilss/",
        "r_consistency",
    ):
        assert kw in text, f"release gate missing: {kw}"


def test_benchmark_recheck_is_isolated():
    text = _workflow()
    assert "benchmark-recheck:" in text
    assert "test_performance_benchmarks.py" in text
    # a separate job that publishes must wait for; no -n / xdist parallelism
    seg = text.split("benchmark-recheck:", 1)[1].split("  build:", 1)[0]
    assert "-n auto" not in seg and "-n " not in seg


def test_publish_is_gated_by_all_release_checks():
    text = _workflow()
    needs_line = [l for l in text.splitlines() if l.strip().startswith("needs:") and "publish" not in l]
    # find the publish-pypi needs line specifically
    seg = text.split("publish-pypi:", 1)[1]
    line = seg.split("needs:", 1)[1].splitlines()[0]
    for job in ("build", "test-install", "pre-release-gate", "sharded-suite", "benchmark-recheck"):
        assert job in line, f"publish-pypi not gated by {job}"
