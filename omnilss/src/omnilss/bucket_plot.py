# SPDX-License-Identifier: GPL-3.0-or-later
"""R-aligned bucket plot (`gamlss::bp`) interface.

R source reference:
- file: ``gamlss/R/bp.R``
- function: ``bp``

Delegates skewness/kurtosis coordinate calculations and 95% normal bands to
OmniLSS's live-R-verified ``moment_bucket`` / ``moment_bucket_wrap`` and
``centile_bucket`` / ``centile_bucket_wrap`` kernels in ``omnilss.ggplots``.
"""

from __future__ import annotations

from typing import Any

from .ggplots import (
    GGPlot,
    centile_bucket,
    centile_bucket_wrap,
    moment_bucket,
    moment_bucket_wrap,
)


def bp(
    obj: Any = None,
    weights: Any = None,
    type: str = "moment",
    xvar: Any = None,
    bootstrap: bool = True,
    no_bootstrap: int = 99,
    n_inter: int = 4,
    xcut_points: Any = None,
    overlap: float = 0.0,
    seed: int = 123,
    text_to_show: Any = None,
    plot: bool = False,
    **kwargs: Any,
) -> GGPlot:
    """R ``gamlss::bp`` bucket plot for skewness and kurtosis diagnostics.

    Parameters
    ----------
    obj:
        A numeric vector, a fitted ``GAMLSSModel``, or a list/tuple of them.
    type:
        One of ``"moment"``, ``"centile.central"``, or ``"centile.tail"``.
    xvar:
        Optional conditioning covariate; when given, produces a wrapped bucket
        plot split across ``n_inter`` intervals.
    bootstrap:
        Whether to compute bootstrap clouds (uses ``no_bootstrap`` resamples).
    plot:
        Whether to build a matplotlib figure in addition to the data payload.
    """
    if obj is None:
        raise ValueError(
            "bp() requires obj (a numeric vector, model, or sequence of them)"
        )

    bucket_type = str(type)
    if bucket_type not in {"moment", "centile.central", "centile.tail"}:
        raise ValueError(
            "type must be one of 'moment', 'centile.central', or 'centile.tail'"
        )

    items = tuple(obj) if isinstance(obj, (list, tuple)) else (obj,)
    if not items:
        raise ValueError("bp() requires at least one numeric vector or model")

    n_boot = int(no_bootstrap) if bootstrap else 0
    if n_boot < 0:
        raise ValueError("no_bootstrap must be non-negative")

    if bucket_type == "moment":
        if xvar is None:
            result = moment_bucket(
                *items,
                no_bootstrap=n_boot,
                seed=seed,
                plot=plot,
                **kwargs,
            )
        else:
            result = moment_bucket_wrap(
                items[0],
                *items[1:],
                weights=weights,
                xvar=xvar,
                n_inter=int(n_inter),
                no_bootstrap=n_boot,
                seed=seed,
                text_to_show=text_to_show,
                plot=plot,
                **kwargs,
            )
    else:
        centile_kind = "central" if bucket_type == "centile.central" else "tail"
        if xvar is None:
            result = centile_bucket(
                items[0],
                *items[1:],
                type=centile_kind,
                weights=weights,
                no_bootstrap=n_boot,
                seed=seed,
                text_to_show=text_to_show,
                plot=plot,
                **kwargs,
            )
        else:
            result = centile_bucket_wrap(
                items[0],
                *items[1:],
                type=centile_kind,
                weights=weights,
                xvar=xvar,
                n_inter=int(n_inter),
                no_bootstrap=n_boot,
                seed=seed,
                text_to_show=text_to_show,
                plot=plot,
                **kwargs,
            )

    payload = (
        dict(result.data) if isinstance(result.data, dict) else {"result": result.data}
    )
    payload["bp_type"] = bucket_type
    payload["bootstrap"] = bool(bootstrap)
    payload["no_bootstrap"] = n_boot
    if xcut_points is not None:
        payload["xcut_points"] = xcut_points
    if overlap:
        payload["overlap"] = float(overlap)
    return GGPlot("bp", payload, result.figure)


__all__ = ["bp"]
