"""Benchmark runner for correctness/performance/convergence/stress suites."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .dataset_registry import BenchmarkDatasetRegistry, default_dataset_registry
from .report import BenchmarkRecord


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_type: str
    family: str
    method: str
    dataset: str
    n: int = 200


class BenchmarkRunner:
    def __init__(self, registry: BenchmarkDatasetRegistry | None = None) -> None:
        self.registry = registry or default_dataset_registry()

    def run(self, spec: BenchmarkSpec) -> BenchmarkRecord:
        dataset = self.registry.get(spec.dataset)
        payload = dataset.builder(spec.n)
        start = time.perf_counter()
        # Placeholder deterministic benchmark workload.
        deviance = float((payload["y"] ** 2).mean())
        elapsed = time.perf_counter() - start
        return BenchmarkRecord(
            benchmark_type=spec.benchmark_type,
            family=spec.family,
            method=spec.method,
            dataset=dataset.name,
            runtime_seconds=elapsed,
            converged=True,
            deviance=deviance,
            metadata={"dataset_version": dataset.version, "n": spec.n},
        )
