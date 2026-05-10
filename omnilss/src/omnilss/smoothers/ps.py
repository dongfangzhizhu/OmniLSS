"""P-spline smooth (ps) smoother for GAMLSS.

ps() 与 pb() 的区别在于节点放置策略：
- pb(): 基于分位数放置节点
- ps(): 基于等间距区间放置节点（R 的 PS.R 实现）

R source: gamlss/R/PS.R
References:
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.
- Rigby & Stasinopoulos (2005). gamlss package.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar

from omnilss.smoothers.bsplines import bspline_basis, bspline_design_matrix
from omnilss.smoothers.penalties import (
    difference_penalty,
    effective_df,
    find_lambda_for_df,
    penalty_matrix,
)


class PSResult:
    """Result from ps() P-spline smooth fitting.

    Attributes
    ----------
    coefficients : np.ndarray
        Spline coefficients (beta)
    fitted_values : np.ndarray
        Fitted values (X @ beta)
    lambda_ : float
        Smoothing parameter used
    edf : float
        Effective degrees of freedom
    design_matrix : np.ndarray
        B-spline design matrix
    penalty : np.ndarray
        Penalty matrix D^T D
    knots : np.ndarray
        Knot sequence used
    degree : int
        B-spline degree
    order : int
        Penalty order
    selection_method : str, optional
        Method used for lambda selection (e.g., "GCV", "REML", "AIC")
    criterion_value : float, optional
        Value of the optimization criterion
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        fitted_values: np.ndarray,
        lambda_: float,
        edf: float,
        design_matrix: np.ndarray,
        penalty: np.ndarray,
        knots: np.ndarray,
        degree: int = 3,
        order: int = 2,
        selection_method: Optional[str] = None,
        criterion_value: Optional[float] = None,
    ):
        self.coefficients = np.asarray(coefficients)
        self.fitted_values = np.asarray(fitted_values)
        self.lambda_ = float(lambda_)
        self.edf = float(edf)
        self.design_matrix = np.asarray(design_matrix)
        self.penalty = np.asarray(penalty)
        self.knots = np.asarray(knots)
        self.degree = degree
        self.order = order
        self.selection_method = selection_method
        self.criterion_value = criterion_value
        self.ps_intervals = None  # set by fit_pspline_smooth
        self.residuals: Optional[np.ndarray] = None
        self.rss: Optional[float] = None
        self.r_squared: Optional[float] = None

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Predict at new x values."""
        X_new = np.asarray(
            bspline_basis(
                jnp.array(np.asarray(x_new).ravel(), dtype=jnp.float64),
                jnp.array(self.knots, dtype=jnp.float64),
                degree=self.degree,
            )
        )
        return X_new @ self.coefficients


def _create_ps_knots(
    x: np.ndarray,
    ps_intervals: int = 20,
    degree: int = 3,
) -> np.ndarray:
    """Create equally-spaced knots for ps() smoother.

    R source: PS.R, function ps()
    节点策略：在 [min(x), max(x)] 上等间距放置 ps_intervals+1 个内部节点，
    然后在两端各扩展 degree 个节点。

    Parameters
    ----------
    x : np.ndarray
        Predictor values
    ps_intervals : int
        Number of equal-width intervals
    degree : int
        B-spline degree

    Returns
    -------
    knots : np.ndarray
        Full knot sequence
    """
    x_min, x_max = float(np.min(x)), float(np.max(x))
    dx = (x_max - x_min) / ps_intervals

    # Interior knots: ps_intervals + 1 points from x_min to x_max
    interior = np.linspace(x_min, x_max, ps_intervals + 1)

    # Extend on both sides by degree knots
    left = x_min - dx * np.arange(degree, 0, -1)
    right = x_max + dx * np.arange(1, degree + 1)

    return np.concatenate([left, interior, right])


def _penalized_ls(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    lambda_: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Penalized weighted least squares.

    Solves: (X^T W X + lambda P) beta = X^T W y

    Returns
    -------
    beta, fitted, edf
    """
    W_sqrt = np.sqrt(w)
    XW = X * W_sqrt[:, None]
    yW = y * W_sqrt

    A = XW.T @ XW + lambda_ * P
    b = XW.T @ yW

    try:
        beta = linalg.solve(A, b, assume_a="pos")
    except linalg.LinAlgError:
        beta = linalg.lstsq(A, b)[0]

    fitted = X @ beta

    # edf = tr(H) where H = X (A)^{-1} X^T W
    # Use jax arrays for effective_df (it expects jnp arrays)
    edf_val = effective_df(
        jnp.array(X), jnp.array(P), lambda_, jnp.array(w)
    )

    return beta, fitted, float(edf_val)


def _lambda_ml(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> float:
    """Estimate lambda via fast eigendecomposition (replaces iterative ML loop)."""
    from omnilss.smoothing_selection import _fast_gcv
    result = _fast_gcv(X, y, P, w)
    return result.lambda_opt


def _lambda_gaic(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    k: float = 2.0,
) -> float:
    """Estimate lambda by GAIC minimisation."""

    def obj(log_lam: float) -> float:
        lam = float(np.exp(log_lam))
        _, fitted, edf = _penalized_ls(X, y, w, P, lam)
        rss = float(np.sum(w * (y - fitted) ** 2))
        return rss + k * edf

    res = minimize_scalar(obj, bounds=(-10.0, 15.0), method="bounded")
    return float(np.exp(res.x))


def _lambda_gcv(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
) -> float:
    """Estimate lambda by GCV minimisation."""
    n = len(y)

    def obj(log_lam: float) -> float:
        lam = float(np.exp(log_lam))
        _, fitted, edf = _penalized_ls(X, y, w, P, lam)
        rss = float(np.sum(w * (y - fitted) ** 2))
        denom = max(n - edf, 1e-6) ** 2
        return n * rss / denom

    res = minimize_scalar(obj, bounds=(-10.0, 15.0), method="bounded")
    return float(np.exp(res.x))


def fit_pspline_smooth(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    df: Optional[float] = None,
    lambda_: Optional[float] = None,
    ps_intervals: int = 20,
    degree: int = 3,
    order: int = 2,
    method: Optional[Literal["ML", "GAIC", "GCV", "REML", "AIC", "auto"]] = None,
    k: float = 2.0,
) -> PSResult:
    """Fit P-spline smooth (ps) to data with automatic lambda selection.

    R source: gamlss/R/PS.R, function ps() / gamlss.ps()

    Parameters
    ----------
    x : np.ndarray
        Predictor variable
    y : np.ndarray
        Response variable
    weights : np.ndarray, optional
        Observation weights
    df : float, optional
        Target effective degrees of freedom
    lambda_ : float, optional
        Fixed smoothing parameter (overrides df and method)
    ps_intervals : int
        Number of equal-width intervals for knot placement (default 20)
    degree : int
        B-spline degree (default 3, cubic)
    order : int
        Difference penalty order (default 2)
    method : {"ML", "GAIC", "GCV", "REML", "AIC", "auto"} or None
        Lambda selection method when df and lambda_ are both None:
        - "ML": Maximum likelihood (iterative)
        - "GAIC": Generalized AIC
        - "GCV": Generalized cross-validation (legacy)
        - "REML": Restricted maximum likelihood (more stable than GCV)
        - "AIC": Akaike information criterion
        - "auto": Automatically select using GCV (recommended)
        If None and df/lambda_ are also None, raises ValueError.
    k : float
        GAIC penalty (default 2 = AIC)

    Returns
    -------
    PSResult
        Fitted P-spline result with additional attributes:
        - selection_method: Method used for lambda selection
        - criterion_value: Value of the optimization criterion
        
    Examples
    --------
    >>> # Automatic lambda selection (recommended)
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 100)
    >>> result = fit_pspline_smooth(x, y, method="auto")
    >>> print(f"Selected lambda: {result.lambda_:.4f}, EDF: {result.edf:.2f}")
    
    >>> # Use REML for lambda selection
    >>> result = fit_pspline_smooth(x, y, method="REML")
    
    >>> # Manual df specification
    >>> result = fit_pspline_smooth(x, y, df=5)
    
    References
    ----------
    - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal
      likelihood estimation of semiparametric generalized linear models.
    - Craven, P., & Wahba, G. (1978). Smoothing noisy data with spline functions.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)

    if len(y) != n:
        raise ValueError(f"x and y must have same length: {n} vs {len(y)}")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()

    # Validate: require explicit df, lambda_, or method
    if df is None and lambda_ is None and method is None:
        raise ValueError("Either df or lambda_ must be specified (or pass method=)")

    # Negative lambda: warn and clamp
    if lambda_ is not None and lambda_ < 0:
        import warnings
        warnings.warn("lambda_ was negative; using 0.0", UserWarning, stacklevel=2)
        lambda_ = 0.0

    # Build knots using ps interval strategy
    knots = _create_ps_knots(x, ps_intervals=ps_intervals, degree=degree)

    # Build design matrix using bspline_basis (vectorised)
    X = np.asarray(
        bspline_basis(
            jnp.array(x, dtype=jnp.float64),
            jnp.array(knots, dtype=jnp.float64),
            degree=degree,
        ),
        dtype=np.float64,
    )
    p = X.shape[1]

    # Clamp order
    order = min(order, p - 1)

    # Penalty matrix
    P = np.asarray(penalty_matrix(p, order=order), dtype=np.float64)

    # Determine lambda
    if lambda_ is not None:
        lam = max(float(lambda_), 0.0)
        selection_info = {"method": "fixed_lambda", "criterion_value": None}
    elif df is not None:
        # find_lambda_for_df expects jnp arrays
        lam = float(
            find_lambda_for_df(
                jnp.array(X), jnp.array(P), float(df), jnp.array(w)
            )
        )
        selection_info = {"method": "fixed_df", "criterion_value": None}
    else:
        # Use new smoothing_selection module for auto/REML/AIC
        if method in ["auto", "REML", "AIC"]:
            try:
                from omnilss.smoothing_selection import select_smoothing_parameter
                
                # Map "auto" to "GCV"
                selection_method = "GCV" if method == "auto" else method
                
                # Use the new smoothing_selection module
                selection_result = select_smoothing_parameter(
                    X=X,
                    y=y,
                    S=P,
                    weights=w if weights is not None else None,
                    method=selection_method
                )
                
                lam = selection_result.lambda_opt
                edf_selected = selection_result.edf
                
                # Store selection info
                selection_info = {
                    "method": selection_method,
                    "criterion_value": selection_result.criterion_value,
                    "converged": selection_result.converged,
                    "n_iterations": selection_result.n_iterations
                }
                
                # Use the selected lambda for final fit
                beta, fitted, edf = _penalized_ls(X, y, w, P, lam)
                
            except ImportError:
                # Fallback to GCV if smoothing_selection not available
                import warnings
                warnings.warn(
                    f"smoothing_selection module not available, falling back to legacy GCV",
                    UserWarning
                )
                lam = _lambda_gcv(X, y, w, P)
                beta, fitted, edf = _penalized_ls(X, y, w, P, lam)
                selection_info = {"method": "GCV (legacy)", "criterion_value": None}
        
        # Use legacy methods
        elif method == "ML":
            lam = _lambda_ml(X, y, w, P)
            beta, fitted, edf = _penalized_ls(X, y, w, P, lam)
            selection_info = {"method": "ML", "criterion_value": None}
        elif method == "GAIC":
            lam = _lambda_gaic(X, y, w, P, k=k)
            beta, fitted, edf = _penalized_ls(X, y, w, P, lam)
            selection_info = {"method": "GAIC", "criterion_value": None}
        elif method == "GCV":
            lam = _lambda_gcv(X, y, w, P)
            beta, fitted, edf = _penalized_ls(X, y, w, P, lam)
            selection_info = {"method": "GCV (legacy)", "criterion_value": None}
        else:
            raise ValueError(f"Unknown method: {method!r}")
    
    # Final fit (if not already done in auto/REML/AIC branch)
    if lambda_ is not None or df is not None:
        beta, fitted, edf = _penalized_ls(X, y, w, P, lam)

    result = PSResult(
        coefficients=beta,
        fitted_values=fitted,
        lambda_=lam,
        edf=edf,
        design_matrix=X,
        penalty=P,
        knots=knots,
        degree=degree,
        order=order,
        selection_method=selection_info.get("method") if 'selection_info' in locals() else None,
        criterion_value=selection_info.get("criterion_value") if 'selection_info' in locals() else None,
    )
    result.ps_intervals = ps_intervals

    residuals = y - fitted
    result.residuals = residuals
    result.rss = float(np.sum(w * residuals**2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    result.r_squared = 1.0 - result.rss / ss_tot if ss_tot > 0 else 0.0

    return result
