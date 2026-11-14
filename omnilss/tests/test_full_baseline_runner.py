"""Acceptance tests for tools/run_full_baseline.py (clean 4-shard baseline).

These pin the two things that actually broke the "clean baseline" runs:

* the child env must NOT export ``JAX_ENABLE_X64`` -- doing so makes
  ``test_jax_config_isolation.py`` fail (it asserts the default is False/0),
  which is an environment artifact, not a code regression;
* each shard must get its OWN model DB (concurrent gRPC sqlite write lock),
  stdout must be real-time (log file), and timing-flaky benchmark failures
  must be classified separately from real correctness failures.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "tools" / "run_full_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_full_baseline", RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_shard_env_strips_polluting_jax_x64_and_isolates_db():
    m = _load()
    base = {"PATH": "x", "JAX_ENABLE_X64": "true", "FOO": "bar"}
    e0, removed = m.shard_env(0, ROOT / "tmp" / "full_baseline", base)
    assert removed == ["JAX_ENABLE_X64"]
    assert "JAX_ENABLE_X64" not in e0
    assert e0["FOO"] == "bar"  # unrelated env preserved
    assert base["JAX_ENABLE_X64"] == "true"  # caller's dict not mutated
    assert "shard-0" in e0["OMNILSS_MODEL_DB_PATH"]
    assert e0["MPLBACKEND"] == "Agg"  # headless backend, matches ci.yml
    e1, _ = m.shard_env(1, ROOT / "tmp" / "full_baseline", base)
    assert e0["OMNILSS_MODEL_DB_PATH"] != e1["OMNILSS_MODEL_DB_PATH"]


def test_build_plan_is_deterministic_clean_and_complete(tmp_path: Path):
    m = _load()
    shards = m._load_shards_module()
    files = shards.test_files()
    plan = m.build_plan(4, tmp_path)
    assert len(plan) == 4
    covered = sorted(f for e in plan for f in e["shard_file"].read_text(encoding="utf-8").splitlines() if f)
    assert covered == files  # lossless, no duplicates
    for i, e in enumerate(plan):
        assert "-m" in e["cmd"] and "pytest" in e["cmd"]
        assert any(a.startswith("@") and f"shard{i}.txt" in a for a in e["cmd"])
        assert "-q" in e["cmd"] and "--tb=line" in e["cmd"]
        assert "JAX_ENABLE_X64" not in e["env"]
        assert "shard-" + str(i) in e["env"]["OMNILSS_MODEL_DB_PATH"]
        assert e["log"].name == f"shard{i}.log"
        assert e["n_files"] > 0


def test_check_and_print_only_are_non_running(tmp_path: Path):
    m = _load()
    assert m.main(["--shards", "4", "--check"]) == 0
    # Use an isolated out_dir: --print-only must never touch a concurrent
    # baseline's live DB files (they are locked on Windows -> WinError 32).
    assert m.main(["--shards", "4", "--print-only", "--out-dir", str(tmp_path)]) == 0


def test_print_only_never_deletes_existing_db_files(tmp_path: Path):
    """Regression: --print-only once unlink()'d a live sqlite DB (WinError 32)."""
    m = _load()
    db = tmp_path / "model-db-shard-0.sqlite3"
    db.write_text("live", encoding="utf-8")
    assert m.main(["--shards", "4", "--print-only", "--out-dir", str(tmp_path)]) == 0
    assert db.exists() and db.read_text(encoding="utf-8") == "live"


def test_summarize_separates_timing_flaky_from_real_failures(tmp_path: Path):
    m = _load()
    good = tmp_path / "good.log"
    good.write_text("........[100%]\n", encoding="utf-8")
    timing = tmp_path / "timing.log"
    timing.write_text(
        "....[100%]\n"
        "FAILED omnilss/tests/test_performance_benchmarks.py::PerformanceBenchmarks::test_NO_small\n",
        encoding="utf-8",
    )
    real = tmp_path / "real.log"
    real.write_text(
        "...[100%]\nFAILED omnilss/tests/test_something.py::test_broken\n",
        encoding="utf-8",
    )
    plan = [
        {"index": 0, "n_files": 3, "rc": 0, "log": good},
        {"index": 1, "n_files": 3, "rc": 1, "log": timing},
        {"index": 2, "n_files": 3, "rc": 1, "log": real},
    ]
    v = m.summarize(plan)
    assert v["result"] == "REVIEW"
    assert v["total_timing_flaky"] == 1
    assert v["total_real_failures"] == 1
    assert v["shards"][1]["timing_flaky"] and not v["shards"][1]["real_failures"]
    assert v["shards"][2]["real_failures"] and not v["shards"][2]["timing_flaky"]
    assert v["shards"][0]["reached_100pct"] is True


def test_summarize_ok_when_all_shards_clean(tmp_path: Path):
    m = _load()
    logs = []
    plan = []
    for i in range(4):
        log = tmp_path / f"shard{i}.log"
        log.write_text("................[100%]\n", encoding="utf-8")
        plan.append({"index": i, "n_files": 5, "rc": 0, "log": log})
    v = m.summarize(plan)
    assert v["result"] == "OK"
    assert v["total_real_failures"] == 0 and v["total_timing_flaky"] == 0


def test_truncated_log_is_incomplete_not_a_false_green(tmp_path: Path):
    """A timeout-killed shard must NEVER be read as 'zero failures'.

    Regression for the first real run: default --timeout 7200s terminated all 4
    shards mid-run. Logs had progress F marks but no FAILED summary lines, and
    the old summarize() reported failed=[] / REVIEW -- a misleading result.
    """
    m = _load()
    truncated = tmp_path / "shard0.log"
    truncated.write_text(
        "........................................................................ [ 48%]\n"
        "...............................................................FFFFFFFFF [ 55%]\n"
        "FFF..................................................................... [ 59%]\n"
        "...........\n",  # killed, never reached 100%, no short-test-summary
        encoding="utf-8",
    )
    plan = [{"index": 0, "n_files": 138, "rc": 1, "log": truncated}]
    v = m.summarize(plan)
    assert v["result"] == "INCOMPLETE"          # not OK, not merely REVIEW
    assert v["any_incomplete"] is True
    assert v["shards"][0]["incomplete"] is True
    assert v["shards"][0]["reached_100pct"] is False
    assert v["shards"][0]["failed"] == []       # summary never printed
    assert v["shards"][0]["progress_fail_marks"] == 12  # 9 + 3 F marks
    assert v["total_progress_fail_marks"] == 12


def test_no_timeout_cap_is_default(tmp_path: Path):
    """The local-terminal runner must not self-terminate at 7200s."""
    m = _load()
    # Drive main() with --print-only (launches nothing) and capture stdout: it
    # prints "timeout=no cap" when the default (0) is in effect.
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert m.main(["--shards", "4", "--print-only", "--out-dir", str(tmp_path)]) == 0
    assert "timeout=no cap" in buf.getvalue()
    # and an explicit cap must be honored / reported
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        assert m.main(["--shards", "4", "--print-only", "--timeout", "60",
                       "--out-dir", str(tmp_path)]) == 0
    assert "timeout=60s" in buf2.getvalue()
