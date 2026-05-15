"""AutoML helpers for distribution selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

from ..chooseDistParallel import choose_dist_data
from ..model import GAMLSSModel


def auto_select_family(
    model: GAMLSSModel,
    criteria: Sequence[float] = (2.0, 3.84),
    type: str = "realAll",
    extra: Sequence[str] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """High-level family selection wrapper with optional parallel criteria scoring."""
    if not parallel or len(criteria) <= 1:
        result = choose_dist_data(model, k=criteria, type=type, extra=extra)
    else:
        matrices = {}
        failed = set()
        families = None
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(choose_dist_data, model, (k,), type, extra): k for k in criteria}
            for fut in as_completed(futs):
                k = futs[fut]
                res = fut.result()
                matrices[k] = res.matrix[:, 0]
                failed.update(res.failed)
                families = res.families
        ordered = [matrices[k] for k in criteria]
        import numpy as np
        matrix = np.column_stack(ordered)
        minima = {float(k): families[int(np.nanargmin(matrix[:, i]))] for i, k in enumerate(criteria)}
        from ..chooseDistParallel import ChooseDistResult
        result = ChooseDistResult(type=type, penalties=tuple(float(k) for k in criteria), families=tuple(families), matrix=matrix, minima=minima, failed=tuple(sorted(failed)))

    ranking = {}
    for i, k in enumerate(result.penalties):
        col = result.matrix[:, i]
        order = col.argsort()
        ranking[str(k)] = [(result.families[idx], float(col[idx])) for idx in order]
    return {
        "best_by_criterion": {str(k): v for k, v in result.minima.items()},
        "ranking": ranking,
        "failed": list(result.failed),
        "families": list(result.families),
    }


__all__ = ["auto_select_family"]
