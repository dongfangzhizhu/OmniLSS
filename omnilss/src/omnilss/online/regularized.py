# SPDX-License-Identifier: GPL-3.0-or-later
"""Online coordinate descent for LASSO, Ridge, and Elastic-Net paths (P5-2).

Operates directly on O(p^2) :class:`OnlineGram` sufficient statistics ``(G, h)``
without storing historical design matrices, minimizing:

    J(beta) = 0.5 * beta^T G beta - h^T beta
              + l1_alpha * ||beta_pen||_1
              + 0.5 * l2_alpha * ||beta_pen||_2^2

where ``beta_pen`` excludes coordinate 0 when ``unpenalized_first=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gram import OnlineGram


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def online_coordinate_descent(
    gram: np.ndarray,
    rhs: np.ndarray,
    *,
    l1_alpha: float = 0.0,
    l2_alpha: float = 0.0,
    beta_init: np.ndarray | None = None,
    unpenalized_first: bool = True,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve regularized quadratic problem directly on Gram matrix ``(G, h)``."""
    G = np.asarray(gram, dtype=np.float64)
    h = np.asarray(rhs, dtype=np.float64).ravel()
    if G.ndim != 2 or G.shape[0] != G.shape[1] or G.shape[0] != h.shape[0]:
        raise ValueError(f"Incompatible shapes: gram={G.shape}, rhs={h.shape}")
    if l1_alpha < 0.0 or l2_alpha < 0.0:
        raise ValueError("l1_alpha and l2_alpha must be non-negative")

    p = h.shape[0]
    if beta_init is None:
        beta = np.zeros(p, dtype=np.float64)
    else:
        beta = np.asarray(beta_init, dtype=np.float64).ravel().copy()
        if beta.shape[0] != p:
            raise ValueError("beta_init length must match rhs")

    # Fast exact path for pure Ridge (l1_alpha == 0)
    if l1_alpha == 0.0:
        lhs = G.copy()
        if l2_alpha > 0.0:
            idx = np.arange(p)
            if unpenalized_first and p > 1:
                idx = idx[1:]
            lhs[idx, idx] += l2_alpha
        try:
            return np.linalg.solve(lhs, h).astype(np.float64, copy=False)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, h, rcond=None)[0].astype(np.float64, copy=False)

    diag_G = np.diag(G)
    # Maintain r = G @ beta incrementally in O(p) per changed coordinate
    G_beta = G @ beta
    for _ in range(int(max_iter)):
        max_change = 0.0
        for j in range(p):
            is_penalized = not (unpenalized_first and j == 0 and p > 1)
            denom = diag_G[j] + (l2_alpha if is_penalized else 0.0)
            if denom <= 1e-15:
                continue
            old_bj = beta[j]
            # Partial residual rho_j = h_j - sum_{k != j} G_{jk} beta_k
            rho_j = h[j] - (G_beta[j] - diag_G[j] * old_bj)
            if is_penalized:
                new_bj = _soft_threshold(float(rho_j), float(l1_alpha)) / float(denom)
            else:
                new_bj = float(rho_j) / float(denom)
            diff = new_bj - old_bj
            if diff != 0.0:
                beta[j] = new_bj
                G_beta += G[:, j] * diff
                abs_diff = abs(diff)
                if abs_diff > max_change:
                    max_change = abs_diff
        if max_change <= tol:
            break
    return beta


@dataclass(frozen=True)
class RegularizedPathResult:
    """Warm-started regularization path over an online Gram accumulator."""

    alphas: np.ndarray
    coefs: np.ndarray  # shape (n_alphas, p)
    rss: np.ndarray  # shape (n_alphas,)
    df: np.ndarray  # shape (n_alphas,)
    ic: np.ndarray  # shape (n_alphas,)
    best_index: int
    criterion: str

    @property
    def best_alpha(self) -> float:
        return float(self.alphas[self.best_index])

    @property
    def best_coef(self) -> np.ndarray:
        return self.coefs[self.best_index].copy()


class OnlineRegularizedPath:
    """Streaming Elastic-Net / LASSO / Ridge regularization path via OnlineGram."""

    def __init__(
        self,
        n_features: int,
        *,
        forget: float = 0.0,
        l1_ratio: float = 1.0,
        n_alphas: int = 20,
        min_alpha_ratio: float = 1e-3,
        fit_intercept: bool = True,
        ridge_floor: float = 1e-10,
    ) -> None:
        if not (0.0 <= float(l1_ratio) <= 1.0):
            raise ValueError(f"l1_ratio must lie in [0, 1], got {l1_ratio}")
        if int(n_alphas) < 2:
            raise ValueError("n_alphas must be >= 2")
        self.n_features_in = int(n_features)
        self.fit_intercept = bool(fit_intercept)
        self.l1_ratio = float(l1_ratio)
        self.n_alphas = int(n_alphas)
        self.min_alpha_ratio = float(min_alpha_ratio)
        total_cols = self.n_features_in + (1 if self.fit_intercept else 0)
        self.gram = OnlineGram(
            total_cols,
            forget=forget,
            ridge=ridge_floor,
            unpenalized_first=self.fit_intercept,
        )

    def _design(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] != self.n_features_in:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in}"
            )
        if self.fit_intercept:
            return np.column_stack([np.ones(X_arr.shape[0], dtype=np.float64), X_arr])
        return X_arr

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None = None
    ) -> "OnlineRegularizedPath":
        self.gram.update(self._design(X), y, weights=sample_weights)
        return self

    update = partial_fit

    def alpha_grid(self, alphas: Sequence[float] | None = None) -> np.ndarray:
        if alphas is not None:
            grid = np.asarray(alphas, dtype=np.float64).ravel()
            if np.any(grid < 0.0):
                raise ValueError("alphas must be non-negative")
            return np.sort(grid)[::-1]
        G = self.gram.gram
        h = self.gram.rhs
        n_eff = max(self.gram.sum_w, 1.0)
        if self.fit_intercept and h.shape[0] > 1:
            b0 = h[0] / max(G[0, 0], 1e-12)
            grad0 = h[1:] - G[1:, 0] * b0
        else:
            grad0 = h
        l1_scale = max(self.l1_ratio, 1e-3)
        alpha_max = float(np.max(np.abs(grad0))) / (n_eff * l1_scale)
        alpha_max = max(alpha_max, 1e-8)
        return np.geomspace(alpha_max, alpha_max * self.min_alpha_ratio, self.n_alphas)

    def path(
        self,
        *,
        alphas: Sequence[float] | None = None,
        criterion: str = "bic",
        max_iter: int = 2000,
        tol: float = 1e-10,
    ) -> RegularizedPathResult:
        if self.gram.n_obs == 0:
            raise RuntimeError("OnlineRegularizedPath has no observations yet")
        crit = criterion.strip().lower()
        if crit not in {"aic", "bic"}:
            raise ValueError("criterion must be 'aic' or 'bic'")

        grid = self.alpha_grid(alphas)
        n_eff = max(self.gram.sum_w, 1.0)
        p = self.gram.n_features
        coefs = np.zeros((grid.size, p), dtype=np.float64)
        rss = np.zeros(grid.size, dtype=np.float64)
        df = np.zeros(grid.size, dtype=np.float64)
        ic = np.zeros(grid.size, dtype=np.float64)

        beta = np.zeros(p, dtype=np.float64)
        if self.fit_intercept and self.gram.gram[0, 0] > 0.0:
            beta[0] = self.gram.rhs[0] / self.gram.gram[0, 0]

        penalty_k = 2.0 if crit == "aic" else float(np.log(max(n_eff, 2.0)))
        for idx, alpha in enumerate(grid):
            l1 = float(n_eff * alpha * self.l1_ratio)
            l2 = float(n_eff * alpha * (1.0 - self.l1_ratio))
            beta = online_coordinate_descent(
                self.gram.gram,
                self.gram.rhs,
                l1_alpha=l1,
                l2_alpha=l2,
                beta_init=beta,
                unpenalized_first=self.fit_intercept,
                max_iter=max_iter,
                tol=tol,
            )
            coefs[idx] = beta
            rss_val = self.gram.residual_sum_of_squares(beta)
            rss[idx] = rss_val
            active = int(np.sum(np.abs(beta) > 1e-10))
            df[idx] = float(active)
            sigma2 = max(rss_val / n_eff, 1e-12)
            ic[idx] = float(n_eff * np.log(sigma2) + penalty_k * active)

        best_idx = int(np.argmin(ic))
        return RegularizedPathResult(
            alphas=grid,
            coefs=coefs,
            rss=rss,
            df=df,
            ic=ic,
            best_index=best_idx,
            criterion=crit,
        )
