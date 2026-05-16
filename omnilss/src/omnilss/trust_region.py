"""Phase 1 trust-region and adaptive step utilities."""

from __future__ import annotations

import numpy as np


def clip_step(delta: np.ndarray, radius: float) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64)
    nrm = np.linalg.norm(d)
    if nrm <= radius or nrm == 0:
        return d
    return d * (radius / nrm)


def adaptive_step_scale(prev_loss: float, new_loss: float, step: float) -> float:
    if new_loss < prev_loss:
        return min(step * 1.2, 1.0)
    return max(step * 0.5, 1e-6)
