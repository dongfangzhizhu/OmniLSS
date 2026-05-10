"""GAMLSS Smoothers and Special Terms.

This module provides smoothing functions and special terms for GAMLSS models,
allowing for flexible non-linear relationships between covariates and distribution parameters.

Main smoothers:
- pb: Penalized B-splines (P-splines) - IMPLEMENTED
- ps: P-splines smooth - IMPLEMENTED
- cs: Cubic splines - IMPLEMENTED
- s: Unified smooth term interface with automatic lambda selection - IMPLEMENTED
- te: Tensor product smooth - IMPLEMENTED
- ti: Tensor product interaction - IMPLEMENTED
- tps: Thin plate splines - IMPLEMENTED
- random: Random effects
- re: Random effects (alternative interface)
- lo: LOESS smoothing

R source: gamlss/R/
"""

from __future__ import annotations

from omnilss.smoothers.bsplines import (
    bspline_basis,
    bspline_design_matrix,
    create_knots,
)
from omnilss.smoothers.cs import CSResult, fit_cubic_spline
from omnilss.smoothers.pb import PSplineResult, fit_pspline
from omnilss.smoothers.penalties import (
    difference_penalty,
    effective_df,
    find_lambda_for_df,
    penalty_matrix,
)
from omnilss.smoothers.ps import PSResult, fit_pspline_smooth
from omnilss.smoothers.smooth_term import SmoothTerm, s
from omnilss.smoothers.tensor_smooth import (
    TensorProductInfo,
    create_tensor_basis,
    create_tensor_product_info,
    evaluate_tensor_smooth,
    te,
    ti,
)
from omnilss.smoothers.tps import (
    TPSResult,
    fit_tps,
    tps,
)
from omnilss.smoothers.ubre import (
    ubre_score,
    select_lambda_ubre,
    estimate_sigma2,
    ubre_vs_gcv,
)

__all__ = [
    # B-splines
    "bspline_basis",
    "bspline_design_matrix",
    "create_knots",
    # Penalties
    "difference_penalty",
    "penalty_matrix",
    "effective_df",
    "find_lambda_for_df",
    # P-splines (pb)
    "fit_pspline",
    "PSplineResult",
    # P-splines smooth (ps)
    "fit_pspline_smooth",
    "PSResult",
    # Cubic splines (cs)
    "fit_cubic_spline",
    "CSResult",
    # Unified smooth term interface
    "s",
    "SmoothTerm",
    # Tensor product smooth
    "te",
    "ti",
    "create_tensor_basis",
    "create_tensor_product_info",
    "evaluate_tensor_smooth",
    "TensorProductInfo",
    # Thin plate splines
    "fit_tps",
    "tps",
    "TPSResult",
    # UBRE
    "ubre_score",
    "select_lambda_ubre",
    "estimate_sigma2",
    "ubre_vs_gcv",
]
