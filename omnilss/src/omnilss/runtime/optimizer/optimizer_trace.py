"""Iteration trace recording/export/replay utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class OptimizerTraceEntry:
    iteration: int
    deviance: float
    gradient_norm: float
    step_size: float
    condition_number: float
    runtime_seconds: float


class OptimizerTrace:
    def __init__(self) -> None:
        self._entries: list[OptimizerTraceEntry] = []

    @property
    def entries(self) -> list[OptimizerTraceEntry]:
        return list(self._entries)

    def record(self, entry: OptimizerTraceEntry) -> None:
        self._entries.append(entry)

    def to_json(self, path: str | Path) -> None:
        payload = [asdict(e) for e in self._entries]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "OptimizerTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        trace = cls()
        for item in data:
            trace.record(OptimizerTraceEntry(**item))
        return trace
