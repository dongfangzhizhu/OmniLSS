# SPDX-License-Identifier: GPL-3.0-or-later
"""Recursive O(p^2) weighted Gram matrix with exponential forgetting (P5-1c).

For forgetting factor ``f = forget in [0, 1)`` (``decay = 1 - f``), updating
with a batch of ``m`` observations ``(X_b, z_b, w_b)`` assigns relative
exponential weights ``gamma_t = decay ** (m - 1 - t)`` for ``t = 0 .. m-1`` and
decays the accumulated statistics by ``decay ** m``:

    G_new = (decay ** m) * G_old + X_b^T diag(gamma * w_b) X_b
    h_new = (decay ** m) * h_old + X_b^T (gamma * w_b * z_b)
    s_new = (decay ** m) * s_old + sum(gamma * w_b * z_b^2)
    W_new = (decay ** m) * W_old + sum(gamma * w_b)

This recurrence is algebraically batch-invariant: applying ``update(A)`` then
``update(B)`` yields the exact same sufficient statistics (up to floating-point
rounding) as ``update(concat(A, B))`` for any ``forget in [0, 1)``.
"""
from __future__ import annotations

import numpy as np


class OnlineGram:
    """Constant-memory O(p^2) recursive weighted Gram accumulator."""

    def __init__(
        self,
        n_features: int,
        *,
        forget: float = 0.0,
        ridge: float = 1e-8,
        unpenalized_first: bool = False,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if not (0.0 <= float(forget) < 1.0):
            raise ValueError(f"forget must lie in [0, 1), got {forget}")
        if float(ridge) < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        self.n_features = int(n_features)
        self.forget = float(forget)
        self.decay = 1.0 - self.forget
        self.ridge = float(ridge)
        self.unpenalized_first = bool(unpenalized_first)
        self.reset()

    def reset(self) -> None:
        """Zero all sufficient statistics."""
        self.gram = np.zeros((self.n_features, self.n_features), dtype=np.float64)
        self.rhs = np.zeros((self.n_features,), dtype=np.float64)
        self.sum_wy2 = 0.0
        self.sum_w = 0.0
        self.n_obs = 0

    def _batch_decay_weights(self, count: int) -> tuple[float, np.ndarray]:
        if count == 0:
            return 1.0, np.empty(0, dtype=np.float64)
        if self.forget == 0.0:
            return 1.0, np.ones(count, dtype=np.float64)
        exponents = np.arange(count - 1, -1, -1, dtype=np.float64)
        return float(self.decay ** count), np.power(self.decay, exponents)

    def update(
        self,
        X: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> int:
        """Incorporate a batch ``(X, z, weights)`` and return valid row count."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X_arr.shape[1]} columns, expected {self.n_features}"
            )
        z_arr = np.asarray(z, dtype=np.float64).ravel()
        if z_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"X ({X_arr.shape[0]}) and z ({z_arr.shape[0]}) row counts differ"
            )
        if weights is None:
            w_arr = np.ones(X_arr.shape[0], dtype=np.float64)
        else:
            w_arr = np.asarray(weights, dtype=np.float64).ravel()
            if w_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("weights length must match X rows")

        valid = (
            np.all(np.isfinite(X_arr), axis=1)
            & np.isfinite(z_arr)
            & np.isfinite(w_arr)
            & (w_arr > 0.0)
        )
        m_valid = int(np.sum(valid))
        if m_valid == 0:
            return 0

        Xv = X_arr[valid]
        zv = z_arr[valid]
        wv = w_arr[valid]

        scale_old, gamma = self._batch_decay_weights(m_valid)
        eff_w = wv * gamma

        if scale_old != 1.0:
            self.gram *= scale_old
            self.rhs *= scale_old
            self.sum_wy2 *= scale_old
            self.sum_w *= scale_old

        Xw = Xv * eff_w[:, None]
        self.gram += Xw.T @ Xv
        self.rhs += Xw.T @ zv
        self.sum_wy2 += float(np.dot(eff_w, zv * zv))
        self.sum_w += float(np.sum(eff_w))
        self.n_obs += m_valid
        return m_valid

    def solve(self, ridge: float | None = None) -> np.ndarray:
        """Solve ``(G + ridge * P) beta = h`` for the current coefficients."""
        reg = self.ridge if ridge is None else float(ridge)
        lhs = self.gram.copy()
        if reg > 0.0:
            diag_idx = np.arange(self.n_features)
            if self.unpenalized_first and self.n_features > 1:
                diag_idx = diag_idx[1:]
            lhs[diag_idx, diag_idx] += reg
        try:
            return np.linalg.solve(lhs, self.rhs).astype(np.float64, copy=False)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, self.rhs, rcond=None)[0].astype(
                np.float64, copy=False
            )

    def residual_sum_of_squares(self, beta: np.ndarray) -> float:
        """Weighted residual sum of squares ``sum_w (y - X beta)^2`` from O(p^2) stats."""
        b = np.asarray(beta, dtype=np.float64).ravel()
        rss = self.sum_wy2 - 2.0 * float(np.dot(b, self.rhs)) + float(b @ self.gram @ b)
        return max(rss, 0.0)

    def memory_bytes(self) -> int:
        """Exact byte size of the O(p^2) state arrays."""
        return int(self.gram.nbytes + self.rhs.nbytes + 32)
