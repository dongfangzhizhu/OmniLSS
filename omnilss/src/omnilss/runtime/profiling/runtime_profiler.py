"""Runtime profiling helpers for memory and timing instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
import tracemalloc


@dataclass(frozen=True)
class ProfileEvent:
    name: str
    runtime_seconds: float
    memory_bytes_delta: int
    tags: dict[str, str]


class RuntimeProfiler:
    def __init__(self) -> None:
        self._events: list[ProfileEvent] = []

    @property
    def events(self) -> list[ProfileEvent]:
        return list(self._events)

    @contextmanager
    def section(self, name: str, **tags: str):
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            after_current, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self._events.append(
                ProfileEvent(
                    name=name,
                    runtime_seconds=elapsed,
                    memory_bytes_delta=int(after_current - before_current),
                    tags={k: str(v) for k, v in tags.items()},
                )
            )

    def export_json(self, path: str | Path) -> None:
        payload = [asdict(event) for event in self._events]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
