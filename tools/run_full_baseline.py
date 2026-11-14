"""One-shot, clean full-suite baseline runner (deterministic 4 shards).

Run it in a LOCAL TERMINAL (not through an agent turn) and keep that terminal
open until it prints the final verdict::

    python tools/run_full_baseline.py                    # no timeout cap; wait it out
    python tools/run_full_baseline.py --shards 8          # same as CI gate
    python tools/run_full_baseline.py --print-only        # show commands, run nothing
    python tools/run_full_baseline.py --check             # shard invariants only

By default there is NO wall-clock cap (4 whole-shard processes on one box can
exceed 2h).  An earlier version capped at 7200s, which terminated all four
shards mid-run and produced a misleading "zero failures / REVIEW" verdict;
the runner now reports such a run as INCOMPLETE (never a valid baseline).

What it does:

* builds a deterministic N-way split via ``tools/pytest_shards.py`` (whole-shard
  mode: one pytest process per shard, so JAX/registry cold-compilation is paid
  once, not once per file);
* launches one pytest process per shard with its OWN ``OMNILSS_MODEL_DB_PATH``
  (avoids the concurrent gRPC sqlite write lock, MEMORY.md D2.216);
* streams each shard's stdout straight into a log file (progress survives an
  interruption instead of living only in a memory buffer);
* waits IN THE FOREGROUND until every shard finishes, then prints a per-shard
  verdict and writes ``verdict.json``.

WHY A SINGLE FOREGROUND ORCHESTRATOR: an agent-turn switch / compact kills the
whole background process group, so 4 separate background jobs never survive the
>1h whole-shard run.  A local terminal has no such boundary.  This script owns
its 4 children and blocks until they are done, so there is exactly one process
to wait for.  (It also works as a single background task in a SHORT, stable new
session -- the D2.214 / 2026-11-10 recipe -- as long as no turn boundary occurs.)

ENV DISCIPLINE: the ONLY override is ``OMNILSS_MODEL_DB_PATH``.  In particular
``JAX_ENABLE_X64`` is stripped: exporting ``JAX_ENABLE_X64=true`` makes
``tests/test_jax_config_isolation.py`` fail (it asserts the default is False/0)
-- an environment artifact, not a code regression.  The successful full5 /
2026-11-10 baselines set nothing but the independent DB.

KNOWN FLAKY: ``test_performance_benchmarks.py`` / ``test_performance_snapshot_freeze.py``
use 0.5s/1.0s wall-clock thresholds that fail under parallel load (machine-load
timer, MEMORY.md).  The verdict lists them separately; re-run that file alone on
an idle machine to confirm.  This runner never edits thresholds to force a green.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tmp" / "full_baseline"
PYTEST_BASE = ("-m", "pytest")
PYTEST_FLAGS = ("-q", "-W", "ignore", "-p", "no:cacheprovider", "--tb=line")
#: env vars that must NOT be injected into a clean baseline child environment
POLLUTING_ENV = ("JAX_ENABLE_X64",)
#: test files whose failures are machine-load timing flaky, not correctness
TIMING_FLAKY = ("test_performance_benchmarks.py", "test_performance_snapshot_freeze.py")
PCT_RE = re.compile(r"\[\s*(\d+)%\]")


def _load_shards_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("_baseline_pytest_shards", here / "pytest_shards.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def shard_env(index: int, out_dir: Path, base: dict | None = None) -> tuple[dict, list[str]]:
    """Clean child env: copy base, strip polluting vars, set one private DB."""
    env = dict(os.environ if base is None else base)
    removed = [k for k in POLLUTING_ENV if k in env]
    for key in POLLUTING_ENV:
        env.pop(key, None)
    env["OMNILSS_MODEL_DB_PATH"] = str(out_dir / f"model-db-shard-{index}.sqlite3")
    return env, removed


def build_plan(shards: int, out_dir: Path, python: str | None = None) -> list[dict]:
    """Deterministic list of {index, cmd, env, log, shard_file} dicts."""
    mod = _load_shards_module()
    files = mod.test_files()
    mod.verify(files, shards)  # lossless / disjoint / non-empty
    shard_files = mod.write_shards(mod.build_shards(files, shards), out_dir)
    py = python or sys.executable
    plan = []

    def _argpath(p: Path) -> str:
        # prefer a repo-relative path (matches the D2.214 recipe & is portable);
        # fall back to absolute when out_dir lives outside the repo.
        try:
            return p.relative_to(ROOT).as_posix()
        except ValueError:
            return p.as_posix()

    for index, shard_file in enumerate(shard_files):
        env, removed = shard_env(index, out_dir)
        # python -m pytest @shardN.txt -q -W ignore -p no:cacheprovider --tb=line
        cmd = [py, *PYTEST_BASE, f"@{_argpath(shard_file)}", *PYTEST_FLAGS]
        plan.append({
            "index": index,
            "cmd": cmd,
            "env": env,
            "removed_env": removed,
            "log": out_dir / f"shard{index}.log",
            "shard_file": shard_file,
            "n_files": len(mod.build_shards(files, shards)[index]),
        })
    return plan


def _progress(log: Path) -> str:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "?"
    pcts = PCT_RE.findall(text)
    return f"{pcts[-1]}%" if pcts else "?"


def _progress_fail_marks(text: str) -> int:
    """Count ``F`` marks on progress lines (``....F...[55%]``).

    A killed/terminated pytest prints its progress dots (including failure
    ``F`` marks) but never the ``short test summary info`` FAILED lines.  These
    marks are the only evidence that failures happened at all -- without them a
    truncated log would be misread as "zero failures".
    """
    marks = 0
    for line in text.splitlines():
        if PCT_RE.search(line):
            marks += line.split("[", 1)[0].count("F")
    return marks


def summarize(plan: list[dict]) -> dict:
    """Parse each log: rc, 100% reached, FAILED/ERROR lines, timing, completeness.

    ``INCOMPLETE`` is deliberately distinct from ``REVIEW``: when a shard was
    terminated before reaching 100% (timeout / Ctrl-C / killed process), its
    failure list is NOT trustworthy -- reporting ``failed=[]`` as a clean result
    would be a false green.  Incomplete results can never be a baseline.
    """
    shards = []
    for entry in plan:
        text = entry["log"].read_text(encoding="utf-8", errors="replace") if entry["log"].exists() else ""
        failed = [l for l in text.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
        timing = [l for l in failed if any(f in l for f in TIMING_FLAKY)]
        real = [l for l in failed if l not in timing]
        reached = "[100%]" in text
        rc = entry.get("rc")
        progress_f = _progress_fail_marks(text)
        # bad rc with no summary lines => process was killed before it could
        # print "short test summary info"; not a clean zero-failure run.
        killed_without_summary = rc not in (0, None) and not failed
        incomplete = not reached or killed_without_summary
        shards.append({
            "index": entry["index"],
            "n_files": entry["n_files"],
            "rc": rc,
            "reached_100pct": reached,
            "incomplete": incomplete,
            "progress_fail_marks": progress_f,
            "failed": failed,
            "timing_flaky": timing,
            "real_failures": real,
            "log": entry["log"].as_posix(),
        })
    any_incomplete = any(s["incomplete"] for s in shards)
    ok = all(
        s["rc"] == 0 and not s["failed"] and s["reached_100pct"] and not s["incomplete"]
        for s in shards
    )
    if any_incomplete:
        result = "INCOMPLETE"
    elif ok:
        result = "OK"
    else:
        result = "REVIEW"
    return {
        "shards": shards,
        "total_real_failures": sum(len(s["real_failures"]) for s in shards),
        "total_timing_flaky": sum(len(s["timing_flaky"]) for s in shards),
        "total_progress_fail_marks": sum(s["progress_fail_marks"] for s in shards),
        "any_incomplete": any_incomplete,
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--python", default=None)
    p.add_argument(
        "--timeout", type=float, default=0.0,
        help="per-run wall-clock cap in seconds; 0 (default) = no cap. A local "
             "terminal run must NOT cap itself: 4 whole-shard processes sharing "
             "one machine routinely exceed 2h. Only set a cap when you actually "
             "want an automatic abort.",
    )
    p.add_argument("--print-only", action="store_true", help="print commands, run nothing")
    p.add_argument("--check", action="store_true", help="verify shard invariants, run nothing")
    args = p.parse_args(argv)

    mod = _load_shards_module()
    files = mod.test_files()
    if args.check:
        mod.verify(files, args.shards)
        print(mod.manifest(files, args.shards))
        print("verify: OK (no suite run)")
        return 0

    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear stale DB/log so a rerun cannot collide with a previous sqlite lock
    for old in out_dir.glob("model-db-shard-*.sqlite3*"):
        old.unlink(missing_ok=True)

    plan = build_plan(args.shards, out_dir, args.python)
    print(mod.manifest(files, args.shards))
    timeout_desc = "no cap" if args.timeout <= 0 else f"{args.timeout:.0f}s"
    print(f"cwd={ROOT.as_posix()}  out_dir={out_dir.as_posix()}  timeout={timeout_desc}")
    for e in plan:
        print(f"  [{e['index']}] {e['n_files']:3d} files -> {e['log'].as_posix()}")
        print(f"       cmd: {' '.join(e['cmd'])}")
        print(f"       env: OMNILSS_MODEL_DB_PATH={e['env']['OMNILSS_MODEL_DB_PATH']}"
              f"  stripped={e['removed_env'] or '[]'}")
    if args.print_only:
        print("print-only: no suite run")
        return 0

    # launch all shards; stdout goes straight to a file handle (real-time)
    running = []
    for e in plan:
        fh = e["log"].open("w", encoding="utf-8", newline="\n")
        proc = subprocess.Popen(
            e["cmd"], cwd=str(ROOT), env=e["env"],
            stdout=fh, stderr=subprocess.STDOUT,
        )
        running.append((e, proc, fh))
    print(f"\nlaunched {len(running)} shards; waiting in foreground (Ctrl-C aborts)...")

    deadline = (time.monotonic() + args.timeout) if args.timeout > 0 else None
    try:
        while True:
            alive = [(e, proc, fh) for (e, proc, fh) in running if proc.poll() is None]
            if not alive:
                break
            if deadline is not None and time.monotonic() > deadline:
                print("TIMEOUT: terminating remaining shards (will be reported INCOMPLETE)")
                for (e, proc, fh) in alive:
                    proc.terminate()
                break
            prog = "  ".join(f"s{e['index']}={_progress(e['log'])}" for (e, proc, fh) in running)
            print(f"  {prog}", flush=True)
            time.sleep(20)
    except KeyboardInterrupt:
        for (e, proc, fh) in running:
            proc.terminate()
        print("interrupted; partial logs kept")
        return 130

    for (e, proc, fh) in running:
        if proc.poll() is None:
            proc.wait()
        fh.close()
        e["rc"] = proc.returncode

    verdict = summarize(plan)
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n==================== full baseline verdict ====================")
    for s in verdict["shards"]:
        if s["incomplete"]:
            flag = "INCOMPLETE"
        elif s["rc"] == 0 and not s["failed"] and s["reached_100pct"]:
            flag = "OK"
        else:
            flag = "CHECK"
        print(f"  shard{s['index']}: rc={s['rc']} 100%={s['reached_100pct']} "
              f"files={s['n_files']} failed={len(s['failed'])} "
              f"progress_F={s['progress_fail_marks']} [{flag}]  {s['log']}")
        for line in s["failed"]:
            print(f"      {line}")
    print(f"  real failures={verdict['total_real_failures']}  "
          f"timing flaky={verdict['total_timing_flaky']}  "
          f"progress_F_marks={verdict['total_progress_fail_marks']}")
    print(f"  result: {verdict['result']}")
    if verdict["any_incomplete"]:
        print("  INCOMPLETE: one or more shards were terminated before 100%. This is "
              "NOT a valid baseline -- failure lines were never printed. "
              "Re-run with no timeout cap (default) and let every shard finish.")
    if verdict["total_progress_fail_marks"] and not verdict["total_real_failures"]:
        print("  NOTE: progress bars show failure (F) marks but no FAILED summary "
              "lines -- the process was killed before pytest could print details.")
    if verdict["total_timing_flaky"]:
        print("  timing flaky -> re-run alone on an idle machine:")
        print("    python -m pytest omnilss/tests/test_performance_benchmarks.py -q -W ignore -p no:cacheprovider")
    print(f"  detail: {(out_dir / 'verdict.json').as_posix()}")
    return 0 if verdict["result"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
