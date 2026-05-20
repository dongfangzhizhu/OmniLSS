"""Structured logging primitives for runtime/optimizer observability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeLogEvent:
    iteration: int
    family: str
    condition_number: float
    has_nan: bool
    level: str = "INFO"
    message: str = "runtime_iteration"


class StructuredRuntimeLogger:
    def __init__(self) -> None:
        self._events: list[RuntimeLogEvent] = []

    @property
    def events(self) -> list[RuntimeLogEvent]:
        return list(self._events)

    def log(self, event: RuntimeLogEvent) -> None:
        self._events.append(event)

    def as_json_lines(self) -> str:
        return "\n".join(json.dumps(asdict(event), sort_keys=True) for event in self._events) + ("\n" if self._events else "")

    def export_json_lines(self, path: str | Path) -> None:
        Path(path).write_text(self.as_json_lines(), encoding="utf-8")


def build_runtime_event(
    *,
    iteration: int,
    family: str,
    condition_number: float,
    values: dict[str, Any],
    level: str = "INFO",
    message: str = "runtime_iteration",
) -> RuntimeLogEvent:
    has_nan = any(bool(getattr(v, "dtype", None) is not None and __import__("numpy").isnan(v).any()) for v in values.values())
    return RuntimeLogEvent(
        iteration=int(iteration),
        family=str(family),
        condition_number=float(condition_number),
        has_nan=bool(has_nan),
        level=level,
        message=message,
    )
