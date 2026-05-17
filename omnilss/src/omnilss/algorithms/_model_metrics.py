"""Shared model metric helpers for algorithm backends."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..smooth_fitting import compute_smooth_edf


def df_fit_with_smooth_edf(
    coefficients: Mapping[str, Any],
    estimable_parameters: Iterable[str],
    design_matrices: Mapping[str, Any],
    weights: Any,
    smooth_infos: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute fitted degrees of freedom with smooth EDF adjustments.

    Penalized smooth terms should contribute their effective degrees of freedom,
    not the full nominal number of basis coefficients.  This helper centralizes
    that accounting so RS-style and CG-style backends report consistent AIC/SBC.
    """

    df_fit = 0.0
    smooth_edf: dict[str, float] = {}
    w = np.asarray(weights, dtype=np.float64)

    for parameter in estimable_parameters:
        coef_count = float(len(np.asarray(coefficients.get(parameter, [0]))))
        smooth_info = smooth_infos.get(parameter)
        smooth_fits = (
            getattr(smooth_info, "smooth_fits", None)
            if smooth_info is not None
            else None
        )
        if smooth_fits:
            smooth_columns = float(
                sum(
                    end - start
                    for start, end in (smooth.basis_columns for smooth in smooth_fits)
                )
            )
            parameter_edf = float(
                compute_smooth_edf(
                    np.asarray(design_matrices[parameter], dtype=np.float64),
                    w,
                    smooth_fits,
                )
            )
            df_fit += (coef_count - smooth_columns) + parameter_edf
            smooth_edf[parameter] = parameter_edf
        else:
            df_fit += coef_count
            smooth_edf[parameter] = 0.0

    return float(df_fit), smooth_edf
