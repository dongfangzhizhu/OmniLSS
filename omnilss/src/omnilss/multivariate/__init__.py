# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-contained JAX multivariate distributions and matrix links (P5-2).

Pure JAX/NumPy implementations of Multivariate Normal (``MVN``) and
Multivariate Student-t (``MVT``) with three unconstrained covariance /
precision parameterizations:

* ``CholeskyCovariance``: ``Sigma = L L^T`` (positive lower-triangular Cholesky)
* ``ModifiedCholeskyPrecision``: ``Precision = T^T D^{-1} T`` (LDL^T precision)
* ``LowRankCovariance``: ``Sigma = diag(D) + B B^T`` (Woodbury O(d r^2))

Zero external imports from reference repositories.
"""
from __future__ import annotations

from .distributions import (
    MVN,
    MVT,
    MultivariateNormal,
    MultivariateStudentT,
)
from .links import (
    CholeskyCovariance,
    LowRankCovariance,
    MatrixLink,
    ModifiedCholeskyPrecision,
    make_matrix_link,
)

__all__ = [
    "CholeskyCovariance",
    "LowRankCovariance",
    "MVN",
    "MVT",
    "MatrixLink",
    "ModifiedCholeskyPrecision",
    "MultivariateNormal",
    "MultivariateStudentT",
    "make_matrix_link",
]
