# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-JAX Multivariate Normal (MVN) and Multivariate Student-t (MVT) families (P5-2)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .links import MatrixLink, make_matrix_link


class MultivariateNormal:
    """JAX-differentiable Multivariate Normal distribution ``N_d(mu, Sigma(eta))``."""

    def __init__(
        self,
        dim: int,
        *,
        parameterization: str = "cholesky",
        rank: int = 1,
    ) -> None:
        self.dim = int(dim)
        self.parameterization = parameterization
        self.matrix_link: MatrixLink = make_matrix_link(
            parameterization, self.dim, rank=rank
        )

    @property
    def n_cov_params(self) -> int:
        return self.matrix_link.n_params

    def covariance(self, cov_eta: jnp.ndarray) -> jnp.ndarray:
        return self.matrix_link.covariance(cov_eta)

    def precision(self, cov_eta: jnp.ndarray) -> jnp.ndarray:
        return self.matrix_link.precision(cov_eta)

    def logpdf(
        self, y: jnp.ndarray, mu: jnp.ndarray, cov_eta: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate log-density for a single vector ``(d,)`` or batch ``(n, d)``."""
        y_arr = jnp.asarray(y, dtype=jnp.float64)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64)
        residual = y_arr - mu_arr
        logdet, quad = self.matrix_link.logdet_and_quad(cov_eta, residual)
        return -0.5 * (self.dim * math.log(2.0 * math.pi) + logdet + quad)

    def score(
        self, y: jnp.ndarray, mu: jnp.ndarray, cov_eta: jnp.ndarray
    ) -> dict[str, jnp.ndarray]:
        """Exact JAX gradients of summed log-likelihood wrt ``mu`` and ``cov_eta``."""
        y_arr = jnp.asarray(y, dtype=jnp.float64)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64)
        eta_arr = jnp.asarray(cov_eta, dtype=jnp.float64)
        grad_mu, grad_eta = jax.grad(
            lambda m, e: jnp.sum(self.logpdf(y_arr, m, e)), argnums=(0, 1)
        )(mu_arr, eta_arr)
        return {"mu": grad_mu, "cov_eta": grad_eta}

    def hessian(
        self, y: jnp.ndarray, mu: jnp.ndarray, cov_eta: jnp.ndarray
    ) -> jnp.ndarray:
        """Joint Hessian of summed log-likelihood wrt ``concat([mu, cov_eta])``."""
        y_arr = jnp.asarray(y, dtype=jnp.float64)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64).ravel()
        eta_arr = jnp.asarray(cov_eta, dtype=jnp.float64).ravel()
        packed = jnp.concatenate([mu_arr, eta_arr])
        d = self.dim

        def obj(theta: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(self.logpdf(y_arr, theta[:d], theta[d:]))

        return jax.hessian(obj)(packed)

    def sample(
        self,
        key: jax.Array,
        n: int,
        mu: jnp.ndarray,
        cov_eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Draw ``(n, dim)`` samples from ``N_d(mu, Sigma(cov_eta))``."""
        mu_arr = jnp.asarray(mu, dtype=jnp.float64).ravel()
        cov = self.covariance(cov_eta)
        L = jnp.linalg.cholesky(cov)
        z = jax.random.normal(key, shape=(int(n), self.dim), dtype=jnp.float64)
        return mu_arr[None, :] + z @ L.T


class MultivariateStudentT:
    """JAX-differentiable Multivariate Student-t distribution ``t_d(mu, Sigma(eta), df)``."""

    def __init__(
        self,
        dim: int,
        *,
        parameterization: str = "cholesky",
        rank: int = 1,
    ) -> None:
        self.dim = int(dim)
        self.parameterization = parameterization
        self.matrix_link: MatrixLink = make_matrix_link(
            parameterization, self.dim, rank=rank
        )

    @property
    def n_cov_params(self) -> int:
        return self.matrix_link.n_params

    def covariance(
        self, cov_eta: jnp.ndarray, df: float | jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Return scale matrix ``Sigma`` (or true covariance ``df/(df-2) Sigma`` if ``df`` given)."""
        scale = self.matrix_link.covariance(cov_eta)
        if df is None:
            return scale
        df_val = jnp.asarray(df, dtype=jnp.float64)
        return jnp.where(df_val > 2.0, (df_val / (df_val - 2.0)) * scale, jnp.nan)

    def logpdf(
        self,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        cov_eta: jnp.ndarray,
        log_df: float | jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate log-density where ``df = exp(log_df) > 0``."""
        y_arr = jnp.asarray(y, dtype=jnp.float64)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64)
        df = jnp.exp(jnp.clip(jnp.asarray(log_df, dtype=jnp.float64), -10.0, 12.0))
        residual = y_arr - mu_arr
        logdet, quad = self.matrix_link.logdet_and_quad(cov_eta, residual)
        d = float(self.dim)
        const = (
            gammaln(0.5 * (df + d))
            - gammaln(0.5 * df)
            - 0.5 * d * (jnp.log(df) + math.log(math.pi))
            - 0.5 * logdet
        )
        return const - 0.5 * (df + d) * jnp.log1p(quad / df)

    def score(
        self,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        cov_eta: jnp.ndarray,
        log_df: float | jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        y_arr = jnp.asarray(y, dtype=jnp.float64)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64)
        eta_arr = jnp.asarray(cov_eta, dtype=jnp.float64)
        ldf_arr = jnp.asarray(log_df, dtype=jnp.float64)
        g_mu, g_eta, g_ldf = jax.grad(
            lambda m, e, ld: jnp.sum(self.logpdf(y_arr, m, e, ld)), argnums=(0, 1, 2)
        )(mu_arr, eta_arr, ldf_arr)
        return {"mu": g_mu, "cov_eta": g_eta, "log_df": g_ldf}

    def hessian(
        self,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        cov_eta: jnp.ndarray,
        log_df: float | jnp.ndarray,
    ) -> jnp.ndarray:
        y_arr = jnp.asarray(y, dtype=jnp.float64)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64).ravel()
        eta_arr = jnp.asarray(cov_eta, dtype=jnp.float64).ravel()
        ldf_arr = jnp.asarray(log_df, dtype=jnp.float64).reshape((1,))
        packed = jnp.concatenate([mu_arr, eta_arr, ldf_arr])
        d = self.dim
        m_end = d + self.n_cov_params

        def obj(theta: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(self.logpdf(y_arr, theta[:d], theta[d:m_end], theta[m_end]))

        return jax.hessian(obj)(packed)

    def sample(
        self,
        key: jax.Array,
        n: int,
        mu: jnp.ndarray,
        cov_eta: jnp.ndarray,
        log_df: float | jnp.ndarray,
    ) -> jnp.ndarray:
        k1, k2 = jax.random.split(key)
        mu_arr = jnp.asarray(mu, dtype=jnp.float64).ravel()
        df = jnp.exp(jnp.asarray(log_df, dtype=jnp.float64))
        L = jnp.linalg.cholesky(self.matrix_link.covariance(cov_eta))
        z = jax.random.normal(k1, shape=(int(n), self.dim), dtype=jnp.float64)
        # chi^2(df) = 2 * Gamma(df / 2, 1)
        g = 2.0 * jax.random.gamma(k2, 0.5 * df, shape=(int(n), 1), dtype=jnp.float64)
        scale = jnp.sqrt(df / g)
        return mu_arr[None, :] + scale * (z @ L.T)


MVN = MultivariateNormal
MVT = MultivariateStudentT
