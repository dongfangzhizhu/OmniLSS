"""R gamlss.foreach helpers: scaled SVD and data correlation screening.

R source: R/gamlss.foreach/R/PCR-GAMLSS.R (getSVD, which.Data.Corr).

getSVD() is R's scale(x) followed by La.svd(); the centred/scaled matrix and the
centre/scale vectors are returned alongside the decomposition so PCR can map
coefficients back to the original scale.

which.Data.Corr() screens the numeric columns of a data frame for pairwise
correlations above a threshold, using the rounded correlation matrix with the
diagonal removed (exactly as R does).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SVDResult:
    """The object R's getSVD() returns."""

    d: np.ndarray
    u: np.ndarray
    vt: np.ndarray
    X: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    @property
    def v(self) -> np.ndarray:
        return self.vt.T


def getSVD(x: Any, nu: int | None = None, nv: int | None = None) -> SVDResult:
    """R getSVD: centre/scale the matrix, then take its SVD.

    R's scale() divides the centred columns by their sample standard deviation
    (n - 1 denominator); a zero-variance column yields NaN there, which is
    reproduced here rather than silently replaced.
    """

    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("x must be a matrix")
    n, p = matrix.shape
    if nu is None:
        nu = min(n, p)
    if nv is None:
        nv = min(n, p)

    center = np.mean(matrix, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.std(matrix, axis=0, ddof=1)
        scaled = (matrix - center) / scale

    u, d, vt = np.linalg.svd(scaled, full_matrices=False)
    return SVDResult(
        d=np.asarray(d[: min(nu, nv, d.shape[0])], dtype=float),
        u=np.asarray(u[:, :nu], dtype=float),
        vt=np.asarray(vt[:nv, :], dtype=float),
        X=scaled,
        center=np.asarray(center, dtype=float),
        scale=np.asarray(scale, dtype=float),
    )


def which_Data_Corr(
    data: Any,
    r: float = 0.90,
    digits: int = 3,
    print: bool = True,
):
    """R which.Data.Corr: pairs of numeric columns correlated above r.

    Returns a three-column frame (name1, name2, corrs) or None when no pair
    exceeds the threshold, matching R's cat() + invisible NULL branch.
    """

    if abs(r) >= 1 or abs(r) <= 0:
        raise ValueError("r should be greater than  0 and lass than 1")

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(dict(data))
    keep = [
        name
        for name in frame.columns
        if not isinstance(frame[name].dtype, pd.CategoricalDtype)
        and not pd.api.types.is_object_dtype(frame[name])
    ]
    numeric = frame[keep]
    matrix = numeric.corr(method="pearson").round(digits)
    values = matrix.to_numpy(dtype=float)
    np.fill_diagonal(values, values.diagonal() - 1.0)

    mask = np.abs(values) > r
    rows, columns = np.nonzero(np.triu(mask, k=1))
    if rows.size == 0:
        if print:
            print(f"no correlation above {r}")
        return None

    names = [str(name) for name in matrix.columns]
    pairs = [
        {"name1": names[i], "name2": names[j], "corrs": float(values[i, j])}
        for i, j in zip(rows, columns)
    ]
    result = pd.DataFrame(pairs, columns=["name1", "name2", "corrs"])
    if print:
        print(result.to_string(index=False))
    return result


__all__ = ["SVDResult", "getSVD", "which_Data_Corr"]
