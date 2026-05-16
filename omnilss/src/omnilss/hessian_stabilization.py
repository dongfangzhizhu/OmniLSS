"""Phase 1 Hessian stabilization with adaptive damping and Fisher fallback."""

from __future__ import annotations

import numpy as np


def stabilize_hessian(h: np.ndarray, lam: float = 1e-6, max_cond: float = 1e10) -> tuple[np.ndarray, float, bool]:
    m = np.asarray(h, dtype=np.float64)
    n = m.shape[0]
    cond = np.linalg.cond(m) if m.size else np.nan
    use_fisher = bool(np.isfinite(cond) and cond > max_cond)
    if use_fisher:
        # Fisher fallback proxy: diagonal curvature floor
        fisher = np.diag(np.maximum(np.abs(np.diag(m)), lam))
        return fisher, cond, True
    return m + lam * np.eye(n), cond, False
