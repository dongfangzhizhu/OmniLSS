from __future__ import annotations

from omnilss.find_hyper import find_hyper


def test_find_hyper_min_mode() -> None:
    out = find_hyper([0.1, 0.3, 0.2], [2.0, 1.0, 3.0], direction="min")
    assert out.best_value == 0.3
    assert out.best_score == 1.0


def test_find_hyper_max_mode() -> None:
    out = find_hyper([1, 2, 3], [0.1, 0.9, 0.2], direction="max")
    assert out.best_value == 2.0
    assert out.best_score == 0.9
