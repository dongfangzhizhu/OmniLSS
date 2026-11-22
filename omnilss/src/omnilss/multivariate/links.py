# SPDX-License-Identifier: GPL-3.0-or-later
"""Unconstrained matrix links for covariance / precision parameterizations (P5-2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np


class MatrixLink(ABC):
    """Map an unconstrained real vector ``eta`` to a positive-definite covariance."""

    def __init__(self, dim: int) -> None:
        if int(dim) < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = int(dim)

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of unconstrained scalar parameters."""

    @abstractmethod
    def covariance(self, eta: jnp.ndarray) -> jnp.ndarray:
        """Return ``(dim, dim)`` symmetric positive-definite covariance matrix."""

    @abstractmethod
    def logdet_and_quad(
        self, eta: jnp.ndarray, residual: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(log|Sigma|, residual^T Sigma^{-1} residual)``."""

    def precision(self, eta: jnp.ndarray) -> jnp.ndarray:
        """Return ``(dim, dim)`` precision matrix ``Sigma^{-1}``."""
        return jnp.linalg.inv(self.covariance(eta))


class CholeskyCovariance(MatrixLink):
    """Lower-triangular Cholesky covariance ``Sigma = L L^T``.

    Layout of ``eta`` (length ``d*(d+1)//2``):
    - ``eta[:d]``: ``log(diag(L))``
    - ``eta[d:]``: strictly lower-triangular entries ``L[i, j]`` (``i > j``)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        tril_i, tril_j = np.tril_indices(self.dim, k=-1)
        self._tril_i = jnp.asarray(tril_i, dtype=jnp.int32)
        self._tril_j = jnp.asarray(tril_j, dtype=jnp.int32)
        self._diag_idx = jnp.arange(self.dim, dtype=jnp.int32)

    @property
    def n_params(self) -> int:
        return self.dim * (self.dim + 1) // 2

    def cholesky_factor(self, eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64).ravel()
        log_diag = jnp.clip(eta[: self.dim], -15.0, 15.0)
        L = jnp.zeros((self.dim, self.dim), dtype=jnp.float64)
        L = L.at[self._diag_idx, self._diag_idx].set(jnp.exp(log_diag))
        if self._tril_i.size > 0:
            L = L.at[self._tril_i, self._tril_j].set(eta[self.dim :])
        return L

    def covariance(self, eta: jnp.ndarray) -> jnp.ndarray:
        L = self.cholesky_factor(eta)
        return L @ L.T

    def logdet_and_quad(
        self, eta: jnp.ndarray, residual: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = jnp.asarray(eta, dtype=jnp.float64).ravel()
        log_diag = jnp.clip(eta[: self.dim], -15.0, 15.0)
        L = self.cholesky_factor(eta)
        logdet = 2.0 * jnp.sum(log_diag)
        # Solve L z = residual (supports 1D or batched (..., d))
        r = jnp.asarray(residual, dtype=jnp.float64)
        if r.ndim == 1:
            z = jnp.linalg.solve(L, r)
            quad = jnp.sum(jnp.square(z))
        else:
            z = jnp.linalg.solve(L, r.T).T
            quad = jnp.sum(jnp.square(z), axis=-1)
        return logdet, quad


class ModifiedCholeskyPrecision(MatrixLink):
    """Modified Cholesky precision ``Omega = T^T D^{-1} T`` (``Sigma = T^{-1} D T^{-T}``).

    Layout of ``eta`` (length ``d*(d+1)//2``):
    - ``eta[:d]``: ``log(D)`` (innovation variances)
    - ``eta[d:]``: strictly lower-triangular entries of unit lower-triangular ``T``
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        tril_i, tril_j = np.tril_indices(self.dim, k=-1)
        self._tril_i = jnp.asarray(tril_i, dtype=jnp.int32)
        self._tril_j = jnp.asarray(tril_j, dtype=jnp.int32)

    @property
    def n_params(self) -> int:
        return self.dim * (self.dim + 1) // 2

    def factors(self, eta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = jnp.asarray(eta, dtype=jnp.float64).ravel()
        log_d = jnp.clip(eta[: self.dim], -15.0, 15.0)
        T = jnp.eye(self.dim, dtype=jnp.float64)
        if self._tril_i.size > 0:
            T = T.at[self._tril_i, self._tril_j].set(eta[self.dim :])
        return T, log_d

    def precision(self, eta: jnp.ndarray) -> jnp.ndarray:
        T, log_d = self.factors(eta)
        inv_d = jnp.exp(-log_d)
        return (T.T * inv_d[None, :]) @ T

    def covariance(self, eta: jnp.ndarray) -> jnp.ndarray:
        T, log_d = self.factors(eta)
        T_inv = jnp.linalg.solve(T, jnp.eye(self.dim, dtype=jnp.float64))
        d_vec = jnp.exp(log_d)
        return (T_inv * d_vec[None, :]) @ T_inv.T

    def logdet_and_quad(
        self, eta: jnp.ndarray, residual: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        T, log_d = self.factors(eta)
        logdet = jnp.sum(log_d)
        inv_d = jnp.exp(-log_d)
        r = jnp.asarray(residual, dtype=jnp.float64)
        if r.ndim == 1:
            Tr = T @ r
            quad = jnp.sum(jnp.square(Tr) * inv_d)
        else:
            Tr = r @ T.T
            quad = jnp.sum(jnp.square(Tr) * inv_d[None, :], axis=-1)
        return logdet, quad


class LowRankCovariance(MatrixLink):
    """Factor / Low-Rank plus Diagonal covariance ``Sigma = diag(D) + B B^T``.

    Uses exact Matrix Determinant Lemma and Woodbury identity in O(d r^2).

    Layout of ``eta`` (length ``d + d * rank``):
    - ``eta[:d]``: ``log(D)``
    - ``eta[d:]``: factor loading matrix ``B`` of shape ``(d, rank)``
    """

    def __init__(self, dim: int, rank: int = 1) -> None:
        super().__init__(dim)
        if int(rank) < 1 or int(rank) > self.dim:
            raise ValueError(f"rank must lie in [1, {self.dim}], got {rank}")
        self.rank = int(rank)

    @property
    def n_params(self) -> int:
        return self.dim + self.dim * self.rank

    def factors(self, eta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = jnp.asarray(eta, dtype=jnp.float64).ravel()
        log_d = jnp.clip(eta[: self.dim], -15.0, 15.0)
        B = eta[self.dim :].reshape((self.dim, self.rank))
        return log_d, B

    def covariance(self, eta: jnp.ndarray) -> jnp.ndarray:
        log_d, B = self.factors(eta)
        return jnp.diag(jnp.exp(log_d)) + B @ B.T

    def logdet_and_quad(
        self, eta: jnp.ndarray, residual: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        log_d, B = self.factors(eta)
        inv_d = jnp.exp(-log_d)
        # Capacitance matrix C = I_r + B^T D^{-1} B (rank x rank SPD)
        Bt_invD = B.T * inv_d[None, :]
        cap = jnp.eye(self.rank, dtype=jnp.float64) + Bt_invD @ B
        L_cap = jnp.linalg.cholesky(cap)
        logdet = jnp.sum(log_d) + 2.0 * jnp.sum(jnp.log(jnp.diag(L_cap)))

        r = jnp.asarray(residual, dtype=jnp.float64)
        if r.ndim == 1:
            base = jnp.sum(jnp.square(r) * inv_d)
            rhs = Bt_invD @ r
            w = jnp.linalg.solve(L_cap, rhs)
            quad = base - jnp.sum(jnp.square(w))
        else:
            base = jnp.sum(jnp.square(r) * inv_d[None, :], axis=-1)
            rhs = r @ Bt_invD.T  # (n, rank)
            w = jnp.linalg.solve(L_cap, rhs.T).T
            quad = base - jnp.sum(jnp.square(w), axis=-1)
        return logdet, quad


def make_matrix_link(parameterization: str, dim: int, *, rank: int = 1) -> MatrixLink:
    """Instantiate a covariance/precision matrix link by name."""
    key = parameterization.strip().lower().replace("-", "_")
    if key == "cholesky":
        return CholeskyCovariance(dim)
    if key in {"modified_cholesky", "modifiedcholesky", "ldl", "precision"}:
        return ModifiedCholeskyPrecision(dim)
    if key in {"low_rank", "lowrank", "factor"}:
        return LowRankCovariance(dim, rank=rank)
    raise ValueError(
        f"Unknown parameterization {parameterization!r}; choose from "
        "'cholesky', 'modified_cholesky', 'low_rank'"
    )
