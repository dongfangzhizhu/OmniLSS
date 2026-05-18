# SPDX-License-Identifier: GPL-3.0-or-later
"""Backward-compatible shim for the historical CG-named L-BFGS module.

The implementation moved to :mod:`omnilss.algorithms.lbfgs_algorithm` because
it performs joint L-BFGS optimization, not the true Cole-Green algorithm.
Use ``joint_lbfgs_fit`` or ``lbfgs_fit`` for this optimizer.  The true
Cole-Green implementation lives in :mod:`omnilss.fitting_cg`.
"""

from __future__ import annotations

import warnings

from .lbfgs_algorithm import (  # noqa: F401
    _compute_cross_derivatives,
    _irls_step_with_adjustment,
    joint_lbfgs_fit,
)

lbfgs_fit = joint_lbfgs_fit


def cg_fit(*args, **kwargs):
    """Deprecated alias for :func:`joint_lbfgs_fit`.

    This name is retained only for old imports from
    ``omnilss.algorithms.cg_algorithm``.  New code should use
    ``joint_lbfgs_fit``/``lbfgs_fit`` or the true Cole-Green API in
    ``omnilss.fitting_cg``.
    """
    warnings.warn(
        "omnilss.algorithms.cg_algorithm.cg_fit is deprecated; "
        "use joint_lbfgs_fit/lbfgs_fit for L-BFGS or omnilss.fitting_cg.fit_cg "
        "for the true Cole-Green algorithm.",
        DeprecationWarning,
        stacklevel=2,
    )
    return joint_lbfgs_fit(*args, **kwargs)


__all__ = [
    "_compute_cross_derivatives",
    "_irls_step_with_adjustment",
    "cg_fit",
    "joint_lbfgs_fit",
    "lbfgs_fit",
]
