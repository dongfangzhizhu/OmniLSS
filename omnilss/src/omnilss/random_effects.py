# SPDX-License-Identifier: GPL-3.0-or-later
"""R-aligned random-effects smoother (`gamlss::re`, `gamlss::gamlss.re`).

R source reference:
- file: ``gamlss/R/re_gamlss_lme_test.R``
- functions: ``re``, ``gamlss.re``

Implements exact marginal ML and REML linear mixed-effects estimation with
observation precision weights ``w`` (matching ``nlme::lme(..., weights =
varFixed(~1/w))``):

    y = X beta + Z b + eps,    b ~ N(0, sigma^2 D),    eps ~ N(0, sigma^2 W^{-1})

Supports single-level grouping with arbitrary fixed effects (``~1``, ``~x1+x2``,
etc.) and random intercept (``~1 | g``) or random intercept + slope(s)
(``~1 + z | g``), marginal (``level=0``) and conditional (``level=1``)
prediction, and exact R ``gamlss.re`` effective degrees of freedom:

    df = N - sum(w * (y - fv)^2) / sigma^2,    nl_df = df - 1,    lambda_ = sigma.

Unsupported ``nlme`` extensions (non-null ``correlation`` structures or
multi-level nested grouping ``~1 | a/b``) raise ``NotImplementedError``
explicitly rather than silently dropping arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar


@dataclass(frozen=True)
class ReSpec:
    """Specification object returned by ``re(...)``."""

    fixed: str
    random: str
    method: str
    level: int | None
    correlation: Any
    control: dict[str, Any]
    data: Mapping[str, Any] | None = field(default=None, repr=False)
    name: str = "re"


@dataclass(frozen=True)
class GamlssReFit:
    """Backfitting return payload matching R ``gamlss.re()``."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    nl_df: float
    lambda_: float
    sigma: float
    fixed_effects: np.ndarray
    fixed_names: tuple[str, ...]
    random_effects: dict[Any, np.ndarray]
    random_names: tuple[str, ...]
    group_col: str
    D_matrix: np.ndarray
    log_lik: float
    method: str
    coefSmo: dict[str, Any]
    var: float = float("nan")
    name: str = "re"


def _parse_rhs_terms(formula_str: str) -> tuple[bool, list[str]]:
    """Parse a simple additive RHS formula like ``~1``, ``~x1 + x2``, ``~0 + x``."""
    text = str(formula_str).strip()
    if "~" in text:
        text = text.split("~", 1)[1].strip()
    if not text or text == "1":
        return True, []
    has_intercept = True
    terms: list[str] = []
    for raw in text.split("+"):
        tok = raw.strip()
        if not tok or tok == "1":
            continue
        if tok in {"0", "-1"}:
            has_intercept = False
            continue
        if tok.endswith("- 1") or tok.endswith("-1"):
            has_intercept = False
            tok = tok[:-2].strip()
            if tok:
                terms.append(tok)
            continue
        terms.append(tok)
    return has_intercept, terms


def _build_design(
    formula_str: str,
    data: Mapping[str, Any],
    n: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    has_intercept, terms = _parse_rhs_terms(formula_str)
    cols: list[np.ndarray] = []
    names: list[str] = []
    if has_intercept:
        cols.append(np.ones(n, dtype=np.float64))
        names.append("(Intercept)")
    for term in terms:
        if term not in data:
            raise KeyError(f"variable {term!r} not found in data")
        arr = np.asarray(data[term], dtype=np.float64).ravel()
        if arr.size != n:
            raise ValueError(f"variable {term!r} has length {arr.size}, expected {n}")
        cols.append(arr)
        names.append(term)
    if not cols:
        raise ValueError(f"formula {formula_str!r} produced an empty design matrix")
    return np.column_stack(cols), tuple(names)


def _parse_random_formula(random: str) -> tuple[str, str]:
    """Parse ``~ <re_terms> | <group_col>`` into (re_formula, group_col)."""
    text = str(random).strip()
    if "|" not in text:
        raise ValueError(
            f"random formula must contain '|' (e.g. '~1 | group'), got {random!r}"
        )
    lhs, rhs = text.split("|", 1)
    group_col = rhs.strip()
    if "/" in group_col or ":" in group_col:
        raise NotImplementedError(
            f"Multi-level nested random effects ({random!r}) are not yet implemented"
        )
    if not group_col:
        raise ValueError(f"missing grouping variable in random formula {random!r}")
    re_formula = lhs.strip() or "~1"
    if not re_formula.startswith("~"):
        re_formula = "~" + re_formula
    return re_formula, group_col


def _unpack_lower_tri(theta: np.ndarray, q: int) -> np.ndarray:
    """Map unconstrained parameter vector to lower-triangular Cholesky factor L of D = L L^T."""
    L = np.zeros((q, q), dtype=np.float64)
    idx = 0
    for i in range(q):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.exp(np.clip(theta[idx], -20.0, 20.0))
            else:
                L[i, j] = theta[idx]
            idx += 1
    return L


def _eval_lmm_objective(
    L: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    group_indices: list[np.ndarray],
    method: str,
) -> tuple[float, np.ndarray, float, list[np.ndarray]]:
    """Evaluate profiled -2*logLik (ML or REML), GLS beta, sigma^2, and BLUPs."""
    n, p = X.shape
    q = Z.shape[1]
    eye_q = np.eye(q, dtype=np.float64)

    XtVinvX = np.zeros((p, p), dtype=np.float64)
    XtVinvy = np.zeros(p, dtype=np.float64)
    logdet_V = 0.0
    # Cache per-group blocks for BLUP calculation
    cached_blocks: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for idx in group_indices:
        wg = w[idx]
        Xg = X[idx]
        Zg = Z[idx]
        yg = y[idx]
        # M_g = I_q + L^T (Z_g^T W_g Z_g) L
        wZg = wg[:, None] * Zg
        ZtWZ = Zg.T @ wZg
        M = eye_q + L.T @ ZtWZ @ L
        # Symmetric positive definite Cholesky
        chol_M = np.linalg.cholesky(M)
        logdet_M = 2.0 * float(np.sum(np.log(np.diag(chol_M))))
        logdet_V += -float(np.sum(np.log(wg))) + logdet_M

        # V_g^{-1} A = W_g A - (W_g Z_g L) M^{-1} (L^T Z_g^T W_g A)
        B = wZg @ L  # (n_g, q)
        BtX = B.T @ Xg  # (q, p)
        Bty = B.T @ yg  # (q,)
        # Solve chol_M U = BtX  =>  U^T U = BtX^T M^{-1} BtX
        U_X = np.linalg.solve(chol_M, BtX)
        U_y = np.linalg.solve(chol_M, Bty)

        XtVinvX += Xg.T @ (wg[:, None] * Xg) - U_X.T @ U_X
        XtVinvy += Xg.T @ (wg * yg) - U_X.T @ U_y
        cached_blocks.append((idx, wg, Zg, B, chol_M))

    # Solve GLS beta
    chol_XtVinvX = np.linalg.cholesky(XtVinvX + 1e-14 * np.eye(p))
    beta = np.linalg.solve(chol_XtVinvX.T, np.linalg.solve(chol_XtVinvX, XtVinvy))

    # Compute quadratic form r^T V^{-1} r and BLUPs b_g = L M^{-1} B^T r_g
    r = y - X @ beta
    quad = 0.0
    blups: list[np.ndarray] = []
    for idx, wg, Zg, B, chol_M in cached_blocks:
        rg = r[idx]
        Btr = B.T @ rg
        u_r = np.linalg.solve(chol_M, Btr)
        quad += float(np.sum(wg * rg * rg) - np.dot(u_r, u_r))
        # b_g = L M^{-1} B^T r_g = L (chol_M^T)^{-1} u_r
        v_r = np.linalg.solve(chol_M.T, u_r)
        blups.append(L @ v_r)

    quad = max(quad, 1e-30)
    if method == "REML":
        dof = max(n - p, 1)
        sigma2 = quad / dof
        logdet_XtVinvX = 2.0 * float(np.sum(np.log(np.diag(chol_XtVinvX))))
        neg2ll = dof * np.log(2.0 * np.pi * sigma2) + logdet_V + logdet_XtVinvX + dof
    else:
        sigma2 = quad / n
        neg2ll = n * np.log(2.0 * np.pi * sigma2) + logdet_V + n

    return float(neg2ll), beta, float(sigma2), blups


def re(
    fixed: str = "~1",
    random: str | None = None,
    correlation: Any = None,
    method: str = "ML",
    level: int | None = None,
    data: Mapping[str, Any] | None = None,
    **control: Any,
) -> ReSpec:
    """R reference: ``gamlss/R/re_gamlss_lme_test.R::re``.

    Builds a random-effects specification for ``gamlss_re``.
    """
    if not isinstance(fixed, str) or not fixed.strip().startswith("~"):
        raise ValueError(
            "fixed argument in re() needs a formula string starting with '~'"
        )
    if random is None:
        raise ValueError(
            "random argument in re() is required (e.g. random='~1 | group')"
        )
    if correlation is not None:
        raise NotImplementedError(
            "correlation structures in re() are not yet implemented"
        )
    meth = str(method).upper()
    if meth not in {"ML", "REML"}:
        raise ValueError("method must be 'ML' or 'REML'")
    _parse_random_formula(random)
    return ReSpec(
        fixed=fixed.strip(),
        random=str(random).strip(),
        method=meth,
        level=level,
        correlation=correlation,
        control=dict(control),
        data=data,
    )


def gamlss_re(
    x: ReSpec | Mapping[str, Any],
    y: Any,
    w: Any = None,
    xeval: Mapping[str, Any] | None = None,
    *,
    data: Mapping[str, Any] | None = None,
    level: int | None = None,
) -> GamlssReFit | np.ndarray:
    """R reference: ``gamlss/R/re_gamlss_lme_test.R::gamlss.re``.

    Fits a weighted linear mixed-effects model via exact marginal ML or REML.
    When ``xeval`` is ``None`` returns :class:`GamlssReFit`; when ``xeval`` is
    a mapping of new predictor/group columns, returns predictions at the
    requested ``level`` (``0`` = population/marginal ``X @ beta``,
    ``1`` or ``None`` = conditional ``X @ beta + Z @ b_g``).
    """
    if isinstance(x, ReSpec):
        spec = x
    elif isinstance(x, Mapping):
        spec = re(
            fixed=str(x.get("fixed", "~1")),
            random=x.get("random"),
            correlation=x.get("correlation"),
            method=str(x.get("method", "ML")),
            level=x.get("level"),
            data=x.get("data"),
        )
    else:
        raise TypeError(
            "gamlss_re expects a ReSpec (from re()) or configuration mapping"
        )

    if spec.correlation is not None:
        raise NotImplementedError(
            "correlation structures in gamlss_re() are not yet implemented"
        )

    fit_data = data if data is not None else spec.data
    if fit_data is None:
        raise ValueError(
            "gamlss_re() requires data (either on re(..., data=...) or gamlss_re(..., data=...))"
        )

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    if w is None:
        w_arr = np.ones(n, dtype=np.float64)
    else:
        w_arr = np.asarray(w, dtype=np.float64).ravel()
        if w_arr.size != n:
            raise ValueError(f"w has length {w_arr.size}, expected {n}")
    if np.any(w_arr < 0.0) or not np.all(np.isfinite(w_arr)):
        raise ValueError("weights w must be finite and non-negative")

    positive_mask = w_arr > 0.0
    if not np.any(positive_mask):
        raise ValueError("at least one observation must have positive weight")

    re_formula, group_col = _parse_random_formula(spec.random)
    if group_col not in fit_data:
        raise KeyError(f"grouping variable {group_col!r} not found in data")

    X_all, fixed_names = _build_design(spec.fixed, fit_data, n)
    Z_all, random_names = _build_design(re_formula, fit_data, n)
    groups_all = np.asarray(fit_data[group_col])
    if groups_all.size != n:
        raise ValueError(
            f"grouping variable {group_col!r} has length {groups_all.size}, expected {n}"
        )

    # Filter to positive-weight observations for exact likelihood
    X = X_all[positive_mask]
    Z = Z_all[positive_mask]
    y_pos = y_arr[positive_mask]
    w_pos = w_arr[positive_mask]
    groups_pos = groups_all[positive_mask]

    unique_groups: list[Any] = []
    group_indices: list[np.ndarray] = []
    seen_map: dict[Any, list[int]] = {}
    for pos_i, gval in enumerate(groups_pos.tolist()):
        if gval not in seen_map:
            seen_map[gval] = []
            unique_groups.append(gval)
        seen_map[gval].append(pos_i)
    for gval in unique_groups:
        group_indices.append(np.asarray(seen_map[gval], dtype=int))

    q = Z.shape[1]
    if q == 1:
        # 1D exact bounded optimization over log(std_ratio) in [-14, 14]
        def _obj_1d(log_s: float) -> float:
            L = np.array([[np.exp(log_s)]], dtype=np.float64)
            neg2ll, _, _, _ = _eval_lmm_objective(
                L, X, Z, y_pos, w_pos, group_indices, spec.method
            )
            return neg2ll

        res_opt = minimize_scalar(
            _obj_1d, bounds=(-14.0, 14.0), method="bounded", options={"xatol": 1e-10}
        )
        L_best = np.array([[np.exp(float(res_opt.x))]], dtype=np.float64)
        # Check boundary tau -> 0 (pure fixed effects)
        neg2ll_opt = float(res_opt.fun)
        L_zero = np.array([[0.0]], dtype=np.float64)
        neg2ll_zero, _, _, _ = _eval_lmm_objective(
            L_zero, X, Z, y_pos, w_pos, group_indices, spec.method
        )
        if neg2ll_zero <= neg2ll_opt:
            L_best = L_zero
    else:
        n_theta = q * (q + 1) // 2

        def _obj_nd(theta: np.ndarray) -> float:
            L = _unpack_lower_tri(theta, q)
            neg2ll, _, _, _ = _eval_lmm_objective(
                L, X, Z, y_pos, w_pos, group_indices, spec.method
            )
            return neg2ll

        best_res = None
        for init_diag in (0.0, -1.0, 1.0):
            theta0 = np.zeros(n_theta, dtype=np.float64)
            idx = 0
            for i in range(q):
                for j in range(i + 1):
                    if i == j:
                        theta0[idx] = init_diag
                    idx += 1
            cand = minimize(_obj_nd, theta0, method="BFGS", options={"gtol": 1e-8})
            if best_res is None or cand.fun < best_res.fun:
                best_res = cand
        L_best = _unpack_lower_tri(best_res.x, q)

    neg2ll, beta, sigma2, blups = _eval_lmm_objective(
        L_best, X, Z, y_pos, w_pos, group_indices, spec.method
    )
    sigma = float(np.sqrt(sigma2))
    D_matrix = L_best @ L_best.T
    re_map = {gval: blups[i] for i, gval in enumerate(unique_groups)}

    # Fitted values on all n rows (unseen zero-weight groups get 0 random effect)
    fv = X_all @ beta
    zero_b = np.zeros(q, dtype=np.float64)
    for row_i, gval in enumerate(groups_all.tolist()):
        fv[row_i] += float(np.dot(Z_all[row_i], re_map.get(gval, zero_b)))

    if xeval is not None:
        pred_level = spec.level if level is None else level
        first_col = next(iter(xeval.values()))
        n_new = int(np.asarray(first_col).ravel().size)
        X_new, _ = _build_design(spec.fixed, xeval, n_new)
        pred = X_new @ beta
        if pred_level is None or int(pred_level) >= 1:
            if group_col in xeval:
                Z_new, _ = _build_design(re_formula, xeval, n_new)
                g_new = np.asarray(xeval[group_col]).ravel().tolist()
                for row_i, gval in enumerate(g_new):
                    pred[row_i] += float(np.dot(Z_new[row_i], re_map.get(gval, zero_b)))
        return pred

    residuals = y_arr - fv
    N_eff = float(np.sum(positive_mask))
    # R re_gamlss_lme_test.R:153-158:
    # N <- sum(w != 0); df <- N - sum(w * (y - fv)^2) / (fit$sigma^2); nl.df = df - 1
    df = N_eff - float(np.sum(w_arr * residuals**2)) / max(sigma2, 1e-30)
    nl_df = df - 1.0
    log_lik = -0.5 * neg2ll

    coef_smo = {
        "fixed": dict(zip(fixed_names, beta.tolist())),
        "random": {g: b.tolist() for g, b in re_map.items()},
        "sigma": sigma,
        "D": D_matrix,
        "G": sigma2 * D_matrix,
        "logLik": log_lik,
        "method": spec.method,
    }
    return GamlssReFit(
        fitted_values=fv,
        residuals=residuals,
        nl_df=float(nl_df),
        lambda_=sigma,
        sigma=sigma,
        fixed_effects=beta,
        fixed_names=fixed_names,
        random_effects=re_map,
        random_names=random_names,
        group_col=group_col,
        D_matrix=D_matrix,
        log_lik=float(log_lik),
        method=spec.method,
        coefSmo=coef_smo,
    )


__all__ = [
    "GamlssReFit",
    "ReSpec",
    "gamlss_re",
    "re",
]
