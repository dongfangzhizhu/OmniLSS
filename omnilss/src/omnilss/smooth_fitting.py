"""GAMLSS fitting with smooth terms support.

This module extends the GAMLSS fitting framework to support smooth terms
using P-splines and other smoothers, leveraging JAX for efficient computation.

R source: gamlss/R/gamlss-5.R, gamlss/R/add.r
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax.numpy as jnp
import numpy as np

from .formula_parser import build_design_matrix, parse_formula
from .smoothers.pb import fit_pspline
from .smoothers.penalties import effective_df


@dataclass
class SmoothFitInfo:
    """Information about a fitted smooth term.

    Attributes
    ----------
    term_index : int
        Index of the smooth term in the formula
    variable : str
        Variable name
    smoother : str
        Smoother type (pb, ps, cs, etc.)
    lambda_ : float
        Smoothing parameter
    edf : float
        Effective degrees of freedom
    penalty : jnp.ndarray
        Penalty matrix D^T D
    basis_columns : tuple[int, int]
        (start, end) column indices in design matrix
    basis_smoother : str, optional
        Concrete basis smoother used for prediction when the formula alias is s().
    selection_method : str, optional
        Method used for lambda selection (e.g., "GCV", "REML", "AIC")
    criterion_value : float, optional
        Value of the optimization criterion
    """

    term_index: int
    variable: str
    smoother: str
    lambda_: float
    edf: float
    penalty: jnp.ndarray
    basis_columns: tuple[int, int]
    basis_smoother: Optional[str] = None
    selection_method: Optional[str] = None
    criterion_value: Optional[float] = None
    knots: Optional[np.ndarray] = None
    degree: Optional[int] = None
    order: Optional[int] = None


@dataclass
class SmoothDesignInfo:
    """Design matrix with smooth term information.

    Attributes
    ----------
    X : np.ndarray
        Full design matrix (linear + smooth basis)
    smooth_fits : list[SmoothFitInfo]
        Information about fitted smooth terms
    linear_columns : int
        Number of linear columns (including intercept)
    has_intercept : bool
        Whether design matrix includes intercept
    """

    X: np.ndarray
    smooth_fits: list[SmoothFitInfo]
    linear_columns: int
    has_intercept: bool


def build_smooth_design(
    formula: str,
    data: Mapping[str, Any],
    weights: Optional[np.ndarray] = None,
) -> SmoothDesignInfo:
    """Build design matrix with smooth terms.

    This function parses the formula, builds the design matrix including
    smooth basis functions, and fits the smooth terms to get initial
    smoothing parameters.

    Parameters
    ----------
    formula : str
        Formula string with possible smooth terms
    data : dict
        Data dictionary
    weights : np.ndarray, optional
        Observation weights

    Returns
    -------
    info : SmoothDesignInfo
        Design matrix and smooth term information

    Examples
    --------
    >>> data = {'y': y, 'x1': x1, 'x2': x2}
    >>> info = build_smooth_design("y ~ x1 + pb(x2, df=5)", data)
    >>> info.X.shape
    (100, 20)  # intercept + x1 + 18 basis functions
    """
    # Parse formula
    parsed = parse_formula(formula)

    # Build design matrix
    X, smooth_info = build_design_matrix(parsed, data, fit_smooths=True)

    # Get response
    y = np.asarray(data[parsed.response])
    n = len(y)

    if weights is None:
        weights = np.ones(n)

    # Count linear columns
    linear_columns = 1 if parsed.has_intercept else 0
    linear_columns += len(parsed.linear_terms)

    # Fit smooth terms to get initial smoothing parameters
    smooth_fits = []
    current_col = linear_columns

    for i, smooth_term in enumerate(parsed.smooth_terms):
        smooth_key = f"smooth_{i}"
        if smooth_key not in smooth_info:
            continue

        info = smooth_info[smooth_key]
        x_smooth = np.asarray(data[smooth_term.variable])

        # Get number of basis columns for this smooth term
        n_basis = info["basis"].shape[1]

        # Fit smooth term to get initial lambda
        smoother_type = smooth_term.smoother

        # For s() function, determine actual smoother to use
        if smoother_type == "s":
            # Check if smoother was specified in kwargs or as attribute
            if hasattr(smooth_term, "kwargs") and "smoother" in smooth_term.kwargs:
                smoother_type = smooth_term.kwargs["smoother"]
            else:
                smoother_type = "pb"  # Default to pb

        # Determine the method to use
        # If smooth_term has a method attribute (from s() function), use it
        if hasattr(smooth_term, "method"):
            fit_method = smooth_term.method
        else:
            fit_method = "ML"  # Default for legacy terms

        if smoother_type == "pb":
            # P-splines (pb)
            if smooth_term.df is not None:
                result = fit_pspline(
                    x_smooth,
                    y,
                    weights=weights,
                    df=smooth_term.df,
                    degree=info.get("degree", 3),
                )
            elif smooth_term.lambda_ is not None:
                result = fit_pspline(
                    x_smooth,
                    y,
                    weights=weights,
                    lambda_=smooth_term.lambda_,
                    degree=info.get("degree", 3),
                )
            else:
                # Use the method from smooth_term (auto, GCV, REML, AIC, etc.)
                result = fit_pspline(
                    x_smooth,
                    y,
                    weights=weights,
                    method=fit_method,
                    degree=info.get("degree", 3),
                )
        elif smoother_type == "ps":
            # P-splines smooth (ps)
            from omnilss.smoothers.ps import fit_pspline_smooth

            if smooth_term.df is not None:
                result = fit_pspline_smooth(
                    x_smooth,
                    y,
                    weights=weights,
                    df=smooth_term.df,
                    ps_intervals=info.get("ps_intervals", 20),
                    degree=info.get("degree", 3),
                )
            elif smooth_term.lambda_ is not None:
                result = fit_pspline_smooth(
                    x_smooth,
                    y,
                    weights=weights,
                    lambda_=smooth_term.lambda_,
                    ps_intervals=info.get("ps_intervals", 20),
                    degree=info.get("degree", 3),
                )
            else:
                # Use the method from smooth_term
                result = fit_pspline_smooth(
                    x_smooth,
                    y,
                    weights=weights,
                    method=fit_method,
                    ps_intervals=info.get("ps_intervals", 20),
                    degree=info.get("degree", 3),
                )
        elif smoother_type == "cs":
            # Cubic smoothing spline
            from omnilss.smoothers.cs import fit_cubic_spline

            # Determine method for cs
            cs_method = (
                fit_method if fit_method in ["GCV", "REML", "AIC", "auto"] else "GCV"
            )

            cs_result = fit_cubic_spline(
                x_smooth,
                y,
                weights=weights,
                df=smooth_term.df,
                method=cs_method,
            )
            n_basis_cs = info["basis"].shape[1]
            from omnilss.smoothers.penalties import penalty_matrix

            dummy_P = penalty_matrix(n_basis_cs, order=2)
            smooth_fits.append(
                SmoothFitInfo(
                    term_index=i,
                    variable=smooth_term.variable,
                    smoother=smooth_term.smoother,
                    lambda_=cs_result.lambda_,
                    edf=cs_result.edf,
                    penalty=dummy_P,
                    basis_columns=(current_col, current_col + n_basis_cs),
                    selection_method=cs_result.selection_method,
                    criterion_value=cs_result.criterion_value,
                )
            )
            current_col += n_basis_cs
            continue
        elif smoother_type == "re":
            # ── 随机效应：s(group, bs="re") ──
            # 本质：带稀疏脊惩罚的虚拟编码
            # 设计矩阵 Z = dummy-coded group indicator
            # 惩罚矩阵 S = I_k（k = 组数）

            group_var = data[smooth_term.variable]
            group_vals = np.asarray(group_var)

            # 获取唯一组别并建立索引映射
            unique_groups = np.unique(group_vals)
            n_groups = len(unique_groups)
            group_to_idx = {g: idx for idx, g in enumerate(unique_groups)}

            # one-hot 编码矩阵（每行仅一个 1，对应所属组）
            Z = np.zeros((len(group_vals), n_groups), dtype=np.float64)
            for row_i, g in enumerate(group_vals):
                Z[row_i, group_to_idx[g]] = 1.0

            # 单位惩罚矩阵（等价于随机截距的 i.i.d. 先验）
            P_re = np.eye(n_groups, dtype=np.float64)

            # 初始平滑参数 λ（随机效应方差的倒数）
            lambda_re = smooth_term.lambda_ if smooth_term.lambda_ is not None else 1.0
            # 近似有效自由度：EDF ≈ k / (1 + λ)
            edf_re = n_groups / (1.0 + lambda_re)

            # 随机效应列的范围：追加到当前设计矩阵末尾
            X_re_end = current_col + n_groups

            smooth_fits.append(
                SmoothFitInfo(
                    term_index=i,
                    variable=smooth_term.variable,
                    smoother="re",
                    lambda_=lambda_re,
                    edf=edf_re,
                    penalty=jnp.asarray(P_re, dtype=jnp.float64),
                    basis_columns=(current_col, X_re_end),
                    selection_method="ML",
                    criterion_value=None,
                )
            )

            # 将随机效应 one-hot 矩阵水平追加到设计矩阵
            X = np.hstack([X, Z])

            current_col = X_re_end
            continue

        else:
            # 未知的平滑器类型
            raise NotImplementedError(f"Smoother '{smoother_type}' not implemented")

        # Store smooth fit info (pb / ps path), including basis metadata
        # required for schema-safe prediction after JSON load.
        smooth_fits.append(
            SmoothFitInfo(
                term_index=i,
                variable=smooth_term.variable,
                smoother=smooth_term.smoother,  # Use original smoother name (could be 's')
                lambda_=result.lambda_,
                edf=result.edf,
                penalty=result.penalty,
                basis_columns=(current_col, current_col + n_basis),
                basis_smoother=smoother_type,
                selection_method=(
                    result.selection_method
                    if hasattr(result, "selection_method")
                    else None
                ),
                criterion_value=(
                    result.criterion_value
                    if hasattr(result, "criterion_value")
                    else None
                ),
                knots=np.asarray(getattr(result, "knots", []), dtype=np.float64),
                degree=int(getattr(result, "degree", info.get("degree", 3))),
                order=int(getattr(result, "order", info.get("order", 2))),
            )
        )

        # Update column counter
        current_col += n_basis

    return SmoothDesignInfo(
        X=X,
        smooth_fits=smooth_fits,
        linear_columns=linear_columns,
        has_intercept=parsed.has_intercept,
    )


def penalized_wls_no_jit(
    X: jnp.ndarray,
    z: jnp.ndarray,
    w: jnp.ndarray,
    penalties: list[tuple[int, int, jnp.ndarray, float]],
) -> jnp.ndarray:
    """Penalized weighted least squares (numpy-based, no JAX scatter overhead).

    Solves: (X^T W X + sum_i lambda_i P_i) beta = X^T W z
    """
    from scipy import linalg as sp_linalg

    X_np = np.asarray(X, dtype=np.float64)
    z_np = np.asarray(z, dtype=np.float64)
    w_np = np.asarray(w, dtype=np.float64)
    n, p = X_np.shape

    # Weighted normal equations
    sqrt_w = np.sqrt(w_np)
    Xw = X_np * sqrt_w[:, None]
    XtWX = Xw.T @ Xw

    # Add penalties using numpy block assignment (no JAX scatter)
    penalty_total = np.zeros((p, p), dtype=np.float64)
    for start, end, P, lambda_ in penalties:
        P_np = np.asarray(P, dtype=np.float64)
        penalty_total[start:end, start:end] += lambda_ * P_np

    A = XtWX + penalty_total
    b = Xw.T @ (z_np * sqrt_w)

    try:
        beta_np = sp_linalg.solve(A + np.eye(p) * 1e-10, b, assume_a="pos")
    except sp_linalg.LinAlgError:
        beta_np = sp_linalg.lstsq(A, b)[0]

    return jnp.array(beta_np)


def fit_penalized_wls(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    smooth_fits: list[SmoothFitInfo],
    auto_lambda: bool = True,
    lambda_method: str = "GCV",
) -> np.ndarray:
    """Fit penalized weighted least squares with optional automatic lambda selection.

    This is a wrapper that handles the conversion between numpy and JAX arrays.
    If auto_lambda is True and any smooth term has lambda_ = None or lambda_ <= 0,
    it will automatically select the optimal lambda using GCV or REML.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    z : np.ndarray
        Working response
    w : np.ndarray
        Weights
    smooth_fits : list[SmoothFitInfo]
        Smooth term information
    auto_lambda : bool, default=True
        Whether to automatically select lambda for terms with lambda_ = None or <= 0
    lambda_method : str, default="GCV"
        Method for automatic lambda selection: "GCV" or "REML"

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients

    Notes
    -----
    Automatic lambda selection is performed per smooth term using the specified method.
    This resolves the major pain point where users had to manually specify df.

    Examples
    --------
    >>> # Automatic lambda selection (new feature!)
    >>> beta = fit_penalized_wls(X, z, w, smooth_fits, auto_lambda=True)
    >>>
    >>> # Manual lambda (backward compatible)
    >>> beta = fit_penalized_wls(X, z, w, smooth_fits, auto_lambda=False)
    """
    # Automatic lambda selection if requested
    if auto_lambda:
        from .smooth_selection import select_lambda

        for smooth in smooth_fits:
            # Check if lambda needs to be selected
            if smooth.lambda_ is None or smooth.lambda_ <= 0:
                start, end = smooth.basis_columns

                # Extract the columns for this smooth term
                X_smooth = X[:, start:end]

                # Get the penalty matrix
                S_smooth = np.array(smooth.penalty)

                # Select optimal lambda
                try:
                    lambda_opt, _ = select_lambda(
                        X_smooth, z, w, S_smooth, method=lambda_method, verbose=False
                    )

                    # Update the smooth fit info
                    smooth.lambda_ = float(lambda_opt)

                except Exception as e:
                    # If selection fails, use a default value
                    import warnings

                    warnings.warn(
                        f"Automatic lambda selection failed for smooth term "
                        f"at columns {start}:{end}. Using default lambda=1.0. "
                        f"Error: {e}",
                        UserWarning,
                    )
                    smooth.lambda_ = 1.0

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    z_jax = jnp.array(z)
    w_jax = jnp.array(w)

    # Prepare penalties
    penalties = []
    for smooth in smooth_fits:
        start, end = smooth.basis_columns
        # Ensure lambda is positive
        lambda_val = (
            smooth.lambda_ if smooth.lambda_ is not None and smooth.lambda_ > 0 else 1.0
        )
        penalties.append(
            (
                start,
                end,
                jnp.array(smooth.penalty),
                float(lambda_val),
            )
        )

    # Fit using JAX (non-JIT version to avoid dynamic slicing issues)
    beta_jax = penalized_wls_no_jit(X_jax, z_jax, w_jax, penalties)

    # Convert back to numpy
    return np.array(beta_jax)


def update_smooth_lambdas(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    w: np.ndarray,
    smooth_fits: list[SmoothFitInfo],
    method: str = "ML",
) -> list[SmoothFitInfo]:
    """Update smoothing parameters for smooth terms.

    Uses ML, GAIC, or GCV to update lambda values.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    y : np.ndarray
        Response
    beta : np.ndarray
        Current coefficients
    w : np.ndarray
        Weights
    smooth_fits : list[SmoothFitInfo]
        Current smooth fit information
    method : str, default="ML"
        Method for updating lambda: "ML", "GAIC", or "GCV"

    Returns
    -------
    updated_fits : list[SmoothFitInfo]
        Updated smooth fit information with new lambdas
    """
    updated_fits = []

    for smooth in smooth_fits:
        start, end = smooth.basis_columns

        # Extract smooth basis columns
        X_smooth = X[:, start:end]
        beta_smooth = beta[start:end]

        # Compute fitted values and residuals
        fv = X @ beta
        residuals = y - fv

        if method == "ML":
            # ML update: lambda = sigma^2 / tau^2
            N = np.sum(w > 0)

            # Residual variance
            sig2 = np.sum(w * residuals**2) / max(N - smooth.edf, 1)
            sig2 = max(sig2, 1e-10)  # Prevent zero/negative variance

            # Penalty variance
            from .smoothers.penalties import difference_penalty

            D = difference_penalty(end - start, order=2)
            gamma = D @ beta_smooth
            tau2 = np.sum(gamma**2) / max(smooth.edf - 2, 1)
            tau2 = max(tau2, 1e-7)  # Prevent zero/negative variance

            # Update lambda
            lambda_new = float(sig2 / tau2)
            lambda_new = np.clip(lambda_new, 1e-7, 1e7)

            # Check for NaN
            if not np.isfinite(lambda_new):
                lambda_new = smooth.lambda_  # Keep old lambda if update fails

            # Compute new edf
            try:
                edf_new = effective_df(
                    jnp.array(X_smooth),
                    jnp.array(smooth.penalty),
                    lambda_new,
                    jnp.array(w),
                )
                if not np.isfinite(edf_new):
                    edf_new = smooth.edf
            except Exception:
                edf_new = smooth.edf

            updated_fits.append(
                SmoothFitInfo(
                    term_index=smooth.term_index,
                    variable=smooth.variable,
                    smoother=smooth.smoother,
                    lambda_=lambda_new,
                    edf=edf_new,
                    penalty=smooth.penalty,
                    basis_columns=smooth.basis_columns,
                    basis_smoother=smooth.basis_smoother,
                    selection_method=smooth.selection_method,  # Preserve selection method
                    criterion_value=smooth.criterion_value,  # Preserve criterion value
                    knots=smooth.knots,
                    degree=smooth.degree,
                    order=smooth.order,
                )
            )
        elif method == "GCV":
            # GCV update: minimize GCV score
            from .smoothers.gcv import optimize_lambda_gcv

            try:
                lambda_new, gcv_score = optimize_lambda_gcv(
                    y=residuals + X_smooth @ beta_smooth,  # Partial residuals
                    X=X_smooth,
                    S=smooth.penalty,
                    lambda_range=(1e-7, 1e7),
                )

                # Compute new edf
                edf_new = effective_df(
                    jnp.array(X_smooth),
                    jnp.array(smooth.penalty),
                    lambda_new,
                    jnp.array(w),
                )

                # Check for valid values
                if (
                    not np.isfinite(lambda_new)
                    or not np.isfinite(edf_new)
                    or not np.isfinite(gcv_score)
                ):
                    lambda_new = smooth.lambda_
                    edf_new = smooth.edf
                    gcv_score = 0.0  # Default value

            except Exception:
                # If optimization fails, keep current lambda
                lambda_new = smooth.lambda_
                edf_new = smooth.edf
                gcv_score = 0.0  # Default value

            updated_fits.append(
                SmoothFitInfo(
                    term_index=smooth.term_index,
                    variable=smooth.variable,
                    smoother=smooth.smoother,
                    lambda_=lambda_new,
                    edf=edf_new,
                    penalty=smooth.penalty,
                    basis_columns=smooth.basis_columns,
                    basis_smoother=smooth.basis_smoother,
                    selection_method="GCV",
                    criterion_value=gcv_score,
                    knots=smooth.knots,
                    degree=smooth.degree,
                    order=smooth.order,
                )
            )

        elif method == "REML":
            # REML update: minimize REML score
            from .smoothers.reml import optimize_lambda_reml

            try:
                lambda_new, reml_score = optimize_lambda_reml(
                    y=residuals + X_smooth @ beta_smooth,  # Partial residuals
                    X=X_smooth,
                    S=smooth.penalty,
                    lambda_range=(1e-7, 1e7),
                )

                # Compute new edf
                edf_new = effective_df(
                    jnp.array(X_smooth),
                    jnp.array(smooth.penalty),
                    lambda_new,
                    jnp.array(w),
                )

                # Check for valid values
                if (
                    not np.isfinite(lambda_new)
                    or not np.isfinite(edf_new)
                    or not np.isfinite(reml_score)
                ):
                    lambda_new = smooth.lambda_
                    edf_new = smooth.edf
                    reml_score = 0.0  # Default value

            except Exception:
                # If optimization fails, keep current lambda
                lambda_new = smooth.lambda_
                edf_new = smooth.edf
                reml_score = 0.0  # Default value

            updated_fits.append(
                SmoothFitInfo(
                    term_index=smooth.term_index,
                    variable=smooth.variable,
                    smoother=smooth.smoother,
                    lambda_=lambda_new,
                    edf=edf_new,
                    penalty=smooth.penalty,
                    basis_columns=smooth.basis_columns,
                    basis_smoother=smooth.basis_smoother,
                    selection_method="REML",
                    criterion_value=reml_score,
                    knots=smooth.knots,
                    degree=smooth.degree,
                    order=smooth.order,
                )
            )

        elif method == "GAIC":
            # GAIC update: minimize GAIC = Deviance + k * edf
            # where k is typically 2 (AIC) or log(n) (BIC)
            from scipy.optimize import minimize_scalar

            k = 2.0  # AIC penalty (can be adjusted to log(n) for BIC)

            def gaic_objective(log_lambda):
                """Compute GAIC for a given lambda."""
                lambda_val = 10**log_lambda

                try:
                    # Compute penalized fit
                    XtX = X_smooth.T @ (w[:, None] * X_smooth)
                    Xty = X_smooth.T @ (w * (residuals + X_smooth @ beta_smooth))
                    penalized_XtX = XtX + lambda_val * smooth.penalty

                    # Solve for coefficients
                    beta_new = np.linalg.solve(penalized_XtX, Xty)

                    # Compute fitted values and deviance
                    fv_new = X_smooth @ beta_new
                    residuals_new = (residuals + X_smooth @ beta_smooth) - fv_new
                    deviance = np.sum(w * residuals_new**2)

                    # Compute effective df
                    edf_val = effective_df(
                        jnp.array(X_smooth),
                        jnp.array(smooth.penalty),
                        lambda_val,
                        jnp.array(w),
                    )

                    # GAIC = Deviance + k * edf
                    gaic = deviance + k * edf_val

                    return gaic if np.isfinite(gaic) else 1e10

                except Exception:
                    return 1e10

            try:
                # Optimize in log space
                result = minimize_scalar(
                    gaic_objective,
                    bounds=(-7, 7),  # log10(lambda) in [1e-7, 1e7]
                    method="bounded",
                )

                lambda_new = 10**result.x
                gaic_score = result.fun

                # Compute new edf
                edf_new = effective_df(
                    jnp.array(X_smooth),
                    jnp.array(smooth.penalty),
                    lambda_new,
                    jnp.array(w),
                )

                # Check for valid values
                if (
                    not np.isfinite(lambda_new)
                    or not np.isfinite(edf_new)
                    or not np.isfinite(gaic_score)
                ):
                    lambda_new = smooth.lambda_
                    edf_new = smooth.edf
                    gaic_score = 0.0  # Default value

            except Exception:
                # If optimization fails, keep current lambda
                lambda_new = smooth.lambda_
                edf_new = smooth.edf
                gaic_score = 0.0  # Default value

            updated_fits.append(
                SmoothFitInfo(
                    term_index=smooth.term_index,
                    variable=smooth.variable,
                    smoother=smooth.smoother,
                    lambda_=lambda_new,
                    edf=edf_new,
                    penalty=smooth.penalty,
                    basis_columns=smooth.basis_columns,
                    basis_smoother=smooth.basis_smoother,
                    selection_method="GAIC",
                    criterion_value=gaic_score,
                    knots=smooth.knots,
                    degree=smooth.degree,
                    order=smooth.order,
                )
            )

        else:
            # Unknown method, keep current lambda
            updated_fits.append(smooth)

    return updated_fits


def compute_smooth_edf(
    X: np.ndarray,
    w: np.ndarray,
    smooth_fits: list[SmoothFitInfo],
) -> float:
    """Compute total effective degrees of freedom for smooth terms.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    w : np.ndarray
        Weights
    smooth_fits : list[SmoothFitInfo]
        Smooth fit information

    Returns
    -------
    total_edf : float
        Total effective degrees of freedom
    """
    total_edf = 0.0

    for smooth in smooth_fits:
        start, end = smooth.basis_columns
        X_smooth = X[:, start:end]

        edf = effective_df(
            jnp.array(X_smooth),
            jnp.array(smooth.penalty),
            smooth.lambda_,
            jnp.array(w),
        )

        total_edf += edf

    return total_edf
