"""Dataset registry for reproducible benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    version: str
    builder: Callable[[int], dict[str, np.ndarray]]


class BenchmarkDatasetRegistry:
    def __init__(self) -> None:
        self._items: dict[str, BenchmarkDataset] = {}

    def register(self, dataset: BenchmarkDataset) -> None:
        self._items[dataset.name] = dataset

    def get(self, name: str) -> BenchmarkDataset:
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items.keys())


def default_dataset_registry() -> BenchmarkDatasetRegistry:
    registry = BenchmarkDatasetRegistry()

    def _normal_linear(n: int) -> dict[str, np.ndarray]:
        x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        y = 1.0 + 2.0 * x
        return {"x": x, "y": y}

    registry.register(BenchmarkDataset(name="normal_linear", version="2026.05", builder=_normal_linear))
    return registry
