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


def test_find_hyper_returns_rank_order_and_index() -> None:
    out = find_hyper([10, 20, 30], [1.5, 0.7, 2.1], direction="min")
    assert out.best_index == 1
    assert out.rank_order.tolist() == [1, 0, 2]


def test_find_hyper_rejects_non_finite_values() -> None:
    try:
        find_hyper([1.0, 2.0], [0.1, float("nan")], direction="min")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "finite" in str(exc)
