# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-contained OnlineGAMLSS estimator (P5-1c).

Uses :func:`omnilss.ondil_contract.as_ondil` (pure NumPy/JAX view of OmniLSS
families — zero external imports) together with per-parameter
:class:`OnlineGram` accumulators to support streaming ``partial_fit`` /
``update`` with exponential forgetting ``forget in [0, 1)``, O(p^2) constant
memory, missing-value filtering, and warm-start parity with batch fitting.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..families import FamilyDefinition
from ..ondil_contract import OndilDistributionView, as_ondil
from .gram import OnlineGram


class OnlineGAMLSS:
    """Streaming distributional regression estimator with O(p^2) state."""

    def __init__(
        self,
        family: FamilyDefinition | str = "NO",
        *,
        forget: float = 0.0,
        max_iter: int = 50,
        tol: float = 1e-6,
        ridge: float = 1e-8,
        fit_intercept: bool = True,
        equations: Mapping[str, Sequence[int] | str] | None = None,
        clip_eta: tuple[float, float] = (-12.0, 12.0),
    ) -> None:
        if not (0.0 <= float(forget) < 1.0):
            raise ValueError(f"forget must lie in [0, 1), got {forget}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if float(ridge) < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        self.family_spec = family
        self.view: OndilDistributionView = as_ondil(family)
        self.forget = float(forget)
        self.decay = 1.0 - self.forget
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.fit_intercept = bool(fit_intercept)
        self.equations = dict(equations) if equations is not None else None
        self.clip_eta = (float(clip_eta[0]), float(clip_eta[1]))

        # Fitted state
        self.is_fitted_: bool = False
        self.n_features_in_: int = 0
        self.n_obs_: int = 0
        self.n_updates_: int = 0
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.param_cols_: dict[str, tuple[int, ...]] = {}
        self.grams_: dict[str, OnlineGram] = {}
        self.coefficients_: dict[str, np.ndarray] = {}
        self._exact_normal_mode_: bool = False

    def _validate_xy(
        self,
        X: Any,
        y: Any,
        sample_weights: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"X ({X_arr.shape[0]}) and y ({y_arr.shape[0]}) row counts differ"
            )
        if sample_weights is None:
            w_arr = np.ones(X_arr.shape[0], dtype=np.float64)
        else:
            w_arr = np.asarray(sample_weights, dtype=np.float64).ravel()
            if w_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weights length must match X rows")
        return X_arr, y_arr, w_arr

    def _filter_valid_rows(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = (
            np.all(np.isfinite(X), axis=1)
            & np.isfinite(y)
            & np.isfinite(w)
            & (w > 0.0)
        )
        # Keep only observations inside the family's declared support
        support = self.view.distribution_support
        if support.lower is not None:
            if support.lower_closed:
                mask &= y >= support.lower
            else:
                mask &= y > support.lower
        if support.upper is not None:
            if support.upper_closed:
                mask &= y <= support.upper
            else:
                mask &= y < support.upper
        return X[mask], y[mask], w[mask]

    def _resolve_equations(self, n_features: int) -> dict[str, tuple[int, ...]]:
        resolved: dict[str, tuple[int, ...]] = {}
        all_cols = tuple(range(n_features))
        for idx, pname in enumerate(self.view.estimable):
            if self.equations is not None and pname in self.equations:
                spec = self.equations[pname]
                if spec == "intercept":
                    resolved[pname] = ()
                else:
                    cols = tuple(int(c) for c in spec)
                    if any(c < 0 or c >= n_features for c in cols):
                        raise ValueError(
                            f"equation columns {cols} out of bounds for n_features={n_features}"
                        )
                    resolved[pname] = cols
            else:
                resolved[pname] = all_cols if idx == 0 else ()
        return resolved

    def _design_for(self, X: np.ndarray, pname: str) -> np.ndarray:
        cols = self.param_cols_[pname]
        n = X.shape[0]
        parts: list[np.ndarray] = []
        if self.fit_intercept or not cols:
            parts.append(np.ones((n, 1), dtype=np.float64))
        if cols:
            parts.append(X[:, list(cols)])
        return np.hstack(parts)

    def _clip_eta_for(self, eta: np.ndarray, idx: int) -> np.ndarray:
        link_name = self.view.links[idx].name
        if link_name == "Identity":
            return np.clip(eta, -1e6, 1e6)
        return np.clip(eta, self.clip_eta[0], self.clip_eta[1])

    def _theta_from_betas(
        self, X: np.ndarray, betas: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        n = X.shape[0]
        theta = np.empty((n, self.view.n_params), dtype=np.float64)
        etas: list[np.ndarray] = []
        for idx, pname in enumerate(self.view.estimable):
            Xp = self._design_for(X, pname)
            eta = self._clip_eta_for(Xp @ betas[pname], idx)
            etas.append(eta)
            theta[:, idx] = self.view.links[idx].inverse(eta)
        return theta, etas

    def _working_quantities(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        eta_k: np.ndarray,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute IRLS Fisher working weights and working response for param ``idx``."""
        dl1 = self.view.dl1_dp1(y, theta, idx)
        dl2 = self.view.dl2_dp2(y, theta, idx)
        # Expected/observed information must be strictly positive for IRLS weights
        neg_hess = np.maximum(-dl2, 1e-8)
        dmu_deta = self.view.links[idx].inverse_derivative(eta_k)
        dmu_deta_safe = np.where(
            np.abs(dmu_deta) < 1e-12,
            np.where(dmu_deta >= 0.0, 1e-12, -1e-12),
            dmu_deta,
        )
        w_irls = np.clip(neg_hess * (dmu_deta_safe ** 2), 1e-8, 1e8)
        step = np.clip(dl1 / (neg_hess * dmu_deta_safe), -5.0, 5.0)
        z_irls = eta_k + step
        return w_irls, z_irls

    def _init_betas(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        betas: dict[str, np.ndarray] = {}
        y_mean = float(np.mean(y))
        y_std = max(float(np.std(y)), 1e-2)
        for idx, pname in enumerate(self.view.estimable):
            Xp = self._design_for(X, pname)
            p_k = Xp.shape[1]
            beta = np.zeros(p_k, dtype=np.float64)
            support = self.view.parameter_support[idx]
            if idx == 0:
                init_val = y_mean
                if support.lower is not None:
                    init_val = max(init_val, support.lower + 1e-2)
                if support.upper is not None:
                    init_val = min(init_val, support.upper - 1e-2)
            elif pname == "sigma":
                init_val = y_std
                if support.upper is not None:
                    init_val = min(max(init_val, 0.05), support.upper - 0.05)
                elif support.lower is not None:
                    init_val = max(init_val, support.lower + 1e-2)
            else:
                init_val = 0.5 if support.upper is not None else 2.0
            eta0 = float(self.view.links[idx].link(np.array([init_val]))[0])
            eta0 = float(self._clip_eta_for(np.array([eta0]), idx)[0])
            beta[0] = eta0
            betas[pname] = beta
        return betas

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weights: Any | None = None,
        *,
        max_iter: int | None = None,
    ) -> "OnlineGAMLSS":
        """Fit the model on an initial batch and initialize O(p^2) Gram states."""
        X_raw, y_raw, w_raw = self._validate_xy(X, y, sample_weights)
        Xv, yv, wv = self._filter_valid_rows(X_raw, y_raw, w_raw)
        if Xv.shape[0] == 0:
            raise ValueError("No valid observations after filtering missing/out-of-support rows")

        self.n_features_in_ = Xv.shape[1]
        self.param_cols_ = self._resolve_equations(self.n_features_in_)
        self._exact_normal_mode_ = (
            self.view.family.name == "NO"
            and self.view.estimable == ("mu", "sigma")
            and len(self.param_cols_["sigma"]) == 0
        )

        self.grams_ = {}
        for pname in self.view.estimable:
            p_k = self._design_for(Xv[:1], pname).shape[1]
            self.grams_[pname] = OnlineGram(
                p_k,
                forget=self.forget,
                ridge=self.ridge,
                unpenalized_first=self.fit_intercept and p_k > 1,
            )

        n_iters = self.max_iter if max_iter is None else int(max_iter)
        if self._exact_normal_mode_:
            # Exact sufficient-statistic path for Normal with constant sigma:
            # guarantees algebraic batch invariance (<= 1e-12) for both mu & sigma.
            X_mu = self._design_for(Xv, "mu")
            self.grams_["mu"].update(X_mu, yv, weights=wv)
            beta_mu = self.grams_["mu"].solve()
            rss = self.grams_["mu"].residual_sum_of_squares(beta_mu)
            sigma_hat = max(np.sqrt(rss / max(self.grams_["mu"].sum_w, 1e-12)), 1e-6)
            beta_sigma = np.array([np.log(sigma_hat)], dtype=np.float64)
            self.coefficients_ = {"mu": beta_mu, "sigma": beta_sigma}
            # Keep sigma Gram aligned for memory reporting
            self.grams_["sigma"].update(
                self._design_for(Xv, "sigma"),
                np.full(yv.shape[0], beta_sigma[0], dtype=np.float64),
                weights=wv,
            )
            self.converged_ = True
            self.n_iter_ = 1
        else:
            betas = self._init_betas(Xv, yv)
            converged = False
            it = 0
            for it in range(1, n_iters + 1):
                max_delta = 0.0
                for idx, pname in enumerate(self.view.estimable):
                    theta, etas = self._theta_from_betas(Xv, betas)
                    w_irls, z_irls = self._working_quantities(yv, theta, etas[idx], idx)
                    Xp = self._design_for(Xv, pname)
                    gram = OnlineGram(
                        Xp.shape[1],
                        forget=self.forget,
                        ridge=self.ridge,
                        unpenalized_first=self.fit_intercept and Xp.shape[1] > 1,
                    )
                    gram.update(Xp, z_irls, weights=wv * w_irls)
                    new_beta = gram.solve()
                    delta = float(np.max(np.abs(new_beta - betas[pname])))
                    max_delta = max(max_delta, delta)
                    betas[pname] = new_beta
                    self.grams_[pname] = gram
                if max_delta <= self.tol:
                    converged = True
                    break
            self.coefficients_ = betas
            self.converged_ = converged
            self.n_iter_ = it

        self.is_fitted_ = True
        self.n_obs_ = int(Xv.shape[0])
        self.n_updates_ = 1
        return self

    def update(
        self,
        X: Any,
        y: Any,
        sample_weights: Any | None = None,
    ) -> "OnlineGAMLSS":
        """Perform a constant-memory O(p^2) streaming update on a new batch."""
        if not self.is_fitted_:
            raise RuntimeError("OnlineGAMLSS must be fitted before calling update(); use partial_fit()")
        X_raw, y_raw, w_raw = self._validate_xy(X, y, sample_weights)
        if X_raw.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_raw.shape[1]} features, expected {self.n_features_in_}"
            )
        Xv, yv, wv = self._filter_valid_rows(X_raw, y_raw, w_raw)
        if Xv.shape[0] == 0:
            return self

        if self._exact_normal_mode_:
            X_mu = self._design_for(Xv, "mu")
            self.grams_["mu"].update(X_mu, yv, weights=wv)
            beta_mu = self.grams_["mu"].solve()
            rss = self.grams_["mu"].residual_sum_of_squares(beta_mu)
            sigma_hat = max(np.sqrt(rss / max(self.grams_["mu"].sum_w, 1e-12)), 1e-6)
            beta_sigma = np.array([np.log(sigma_hat)], dtype=np.float64)
            self.coefficients_["mu"] = beta_mu
            self.coefficients_["sigma"] = beta_sigma
        else:
            for idx, pname in enumerate(self.view.estimable):
                theta, etas = self._theta_from_betas(Xv, self.coefficients_)
                w_irls, z_irls = self._working_quantities(yv, theta, etas[idx], idx)
                Xp = self._design_for(Xv, pname)
                self.grams_[pname].update(Xp, z_irls, weights=wv * w_irls)
                self.coefficients_[pname] = self.grams_[pname].solve()

        self.n_obs_ += int(Xv.shape[0])
        self.n_updates_ += 1
        return self

    def partial_fit(
        self,
        X: Any,
        y: Any,
        sample_weights: Any | None = None,
        *,
        max_iter: int | None = None,
    ) -> "OnlineGAMLSS":
        """Streaming entry point: first call == ``fit``, subsequent calls == ``update``."""
        if not self.is_fitted_:
            return self.fit(X, y, sample_weights=sample_weights, max_iter=max_iter)
        return self.update(X, y, sample_weights=sample_weights)

    def predict_theta(self, X: Any) -> np.ndarray:
        """Predict distribution parameter matrix ``theta`` of shape ``(n, n_params)``."""
        if not self.is_fitted_:
            raise RuntimeError("OnlineGAMLSS is not fitted")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}"
            )
        theta, _ = self._theta_from_betas(X_arr, self.coefficients_)
        return theta

    def predict_params(self, X: Any) -> dict[str, np.ndarray]:
        """Predict distribution parameters as ``{param_name: array}``."""
        theta = self.predict_theta(X)
        return {
            pname: theta[:, idx].copy()
            for idx, pname in enumerate(self.view.estimable)
        }

    def predict_quantiles(
        self, X: Any, quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    ) -> dict[str, np.ndarray]:
        """Predict conditional quantiles for each observation in ``X``."""
        theta = self.predict_theta(X)
        out: dict[str, np.ndarray] = {}
        for q in quantiles:
            q_arr = np.full(theta.shape[0], float(q), dtype=np.float64)
            out[str(q)] = self.view.ppf(q_arr, theta)
        return out

    def log_likelihood(self, X: Any, y: Any) -> float:
        """Total log-likelihood evaluated on ``(X, y)``."""
        X_arr, y_arr, _ = self._validate_xy(X, y)
        theta = self.predict_theta(X_arr)
        return float(np.sum(self.view.logpdf(y_arr, theta)))

    def state_memory_bytes(self) -> int:
        """Exact byte footprint of stored O(p^2) sufficient statistics and coefficients."""
        gram_bytes = sum(g.memory_bytes() for g in self.grams_.values())
        coef_bytes = sum(int(b.nbytes) for b in self.coefficients_.values())
        return int(gram_bytes + coef_bytes)
