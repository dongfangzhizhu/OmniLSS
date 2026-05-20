"""Flamegraph-friendly collapsed stack export."""

from __future__ import annotations

from pathlib import Path

from .runtime_profiler import RuntimeProfiler


def to_collapsed_stacks(profiler: RuntimeProfiler, root: str = "runtime") -> str:
    """Emit collapsed-stack lines compatible with flamegraph.pl input format."""
    lines: list[str] = []
    for event in profiler.events:
        stack = f"{root};{event.name}"
        samples = max(1, int(event.runtime_seconds * 1_000_000))
        lines.append(f"{stack} {samples}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_collapsed_stacks(path: str | Path, profiler: RuntimeProfiler, root: str = "runtime") -> None:
    Path(path).write_text(to_collapsed_stacks(profiler, root=root), encoding="utf-8")
