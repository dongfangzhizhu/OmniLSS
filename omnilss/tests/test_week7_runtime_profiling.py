from __future__ import annotations

import json

from omnilss.runtime.profiling import RuntimeProfiler, to_collapsed_stacks, write_collapsed_stacks


def test_runtime_profiler_records_timing_and_memory(tmp_path):
    profiler = RuntimeProfiler()
    with profiler.section("solver", phase="fit"):
        _ = [i * i for i in range(1000)]

    assert len(profiler.events) == 1
    event = profiler.events[0]
    assert event.name == "solver"
    assert event.runtime_seconds >= 0.0

    out = tmp_path / "profile.json"
    profiler.export_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["tags"]["phase"] == "fit"


def test_flamegraph_collapsed_stack_export(tmp_path):
    profiler = RuntimeProfiler()
    with profiler.section("kernel"):
        _ = sum(range(10))

    collapsed = to_collapsed_stacks(profiler, root="omnilss")
    assert "omnilss;kernel" in collapsed

    out = tmp_path / "collapsed.txt"
    write_collapsed_stacks(out, profiler, root="omnilss")
    assert out.read_text(encoding="utf-8") == collapsed
