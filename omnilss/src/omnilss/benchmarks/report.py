"""Benchmark report generation and schema versioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_type: str
    family: str
    method: str
    dataset: str
    runtime_seconds: float
    converged: bool
    deviance: float
    metadata: dict[str, Any]


def render_report(records: list[BenchmarkRecord]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "records": [asdict(r) for r in records],
    }


def write_report(path: str | Path, records: list[BenchmarkRecord]) -> None:
    Path(path).write_text(json.dumps(render_report(records), indent=2), encoding="utf-8")
