"""R-aligned interval diagnostic interfaces.

R source reference:
- file: `gamlss/R/qstats.R`
- functions: `Q.stats`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.scipy.special import gammaincc

from .model import GAMLSSModel
from .operations import is_gamlss, residuals


@dataclass(frozen=True)
class QStatsResult:
    """Structured interval-based quantile residual normality diagnostics."""

    interval_labels: tuple[str, ...]
    z_matrix: np.ndarray
    q_matrix: np.ndarray
    totals: dict[str, np.ndarray]


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _ordered_residuals(object: GAMLSSModel) -> np.ndarray:
    values = np.asarray(residuals(object, what="z-scores"), dtype=np.float64).ravel()
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def _chi2_sf(value: float, df: float) -> float:
    if df <= 0:
        return float("nan")
    return float(gammaincc(df / 2.0, value / 2.0))


def _qtest(values: np.ndarray) -> dict[str, float]:
    """R reference: `gamlss/R/qstats.R::Q.stats` (internal Q-test helper)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return {
            "Q1": float("nan"),
            "Q2": float("nan"),
            "Q3": float("nan"),
            "Q4": float("nan"),
            "AgostinoK2": float("nan"),
            "z1": float("nan"),
            "z2": float("nan"),
            "z3": float("nan"),
            "z4": float("nan"),
            "N": float(n),
        }
    mean = float(np.mean(x))
    centered = x - mean
    m2 = float(np.mean(np.square(centered)))
    m3 = float(np.mean(np.power(centered, 3.0)))
    m4 = float(np.mean(np.power(centered, 4.0)))
    variance = max(m2, np.finfo(np.float64).eps)
    sd = math.sqrt(variance)

    q1 = n * mean**2
    q2 = ((sd ** (2.0 / 3.0) - (1.0 - 2.0 / (9.0 * n - 9.0))) ** 2) / (2.0 / (9.0 * n - 9.0))
    z1 = math.sqrt(n) * mean
    z2 = math.sqrt(q2) * math.copysign(1.0, sd ** (2.0 / 3.0) - (1.0 - 2.0 / (9.0 * n - 9.0)))

    sqrt_b1 = m3 / max(variance ** 1.5, np.finfo(np.float64).eps)
    yval = sqrt_b1 * math.sqrt((n + 1.0) * (n + 3.0) / (6.0 * max(n - 2.0, 1.0)))
    b2b1 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
    b2b1 /= max((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0), np.finfo(np.float64).eps)
    wsq = max(-1.0 + math.sqrt(max(2.0 * b2b1 - 2.0, np.finfo(np.float64).eps)), np.finfo(np.float64).eps)
    delta = 1.0 / math.sqrt(max(0.5 * math.log(wsq), np.finfo(np.float64).eps))
    alpha = math.sqrt(2.0 / max(wsq - 1.0, np.finfo(np.float64).eps))
    yalpha = yval / max(alpha, np.finfo(np.float64).eps)
    z3 = delta * math.log(yalpha + math.sqrt(yalpha * yalpha + 1.0))
    q3 = z3**2

    b2 = m4 / max(variance * variance, np.finfo(np.float64).eps)
    mean_b2 = 3.0 * (n - 1.0) / (n + 1.0)
    var_b2 = 24.0 * n * (n - 2.0) * (n - 3.0)
    var_b2 /= max((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0), np.finfo(np.float64).eps)
    xval = (b2 - mean_b2) / math.sqrt(max(var_b2, np.finfo(np.float64).eps))
    sb1b2 = 6.0 * (n * n - 5.0 * n + 2.0) / max((n + 7.0) * (n + 9.0), np.finfo(np.float64).eps)
    sb1b2 *= math.sqrt(6.0 * (n + 3.0) * (n + 5.0) / max(n * (n - 2.0) * (n - 3.0), np.finfo(np.float64).eps))
    isb1b2 = 1.0 / max(sb1b2, np.finfo(np.float64).eps)
    aval = 6.0 + 8.0 * isb1b2 * (2.0 * isb1b2 + math.sqrt(1.0 + 4.0 * isb1b2**2))
    inner = 1.0 + xval * math.sqrt(2.0 / max(aval - 4.0, np.finfo(np.float64).eps))
    inner = max(inner, np.finfo(np.float64).eps)
    z4 = (1.0 - 2.0 / (9.0 * aval)) - ((1.0 - (2.0 / aval)) / inner) ** (1.0 / 3.0)
    z4 /= math.sqrt(2.0 / (9.0 * aval))
    q4 = z4**2

    return {
        "Q1": float(q1),
        "Q2": float(q2),
        "Q3": float(q3),
        "Q4": float(q4),
        "AgostinoK2": float(q3 + q4),
        "z1": float(z1),
        "z2": float(z2),
        "z3": float(z3),
        "z4": float(z4),
        "N": float(n),
    }


def q_stats(
    object: GAMLSSModel,
    xvar: Any | None = None,
    xcut_points: np.ndarray | None = None,
    n_inter: int = 10,
) -> QStatsResult:
    """R reference: `gamlss/R/qstats.R::Q.stats`.

    Staged behavior:
    - Returns interval-wise Z/Q statistics and total chi-square summaries.
    - Uses residual z-scores and simple equal-width interval partitioning.
    """

    _require_gamlss(object)
    residx = _ordered_residuals(object)
    if residx.size == 0:
        raise ValueError("need residuals for Q statistics")
    if xvar is None:
        x_values = np.arange(1, residx.size + 1, dtype=np.float64)
    else:
        x_values = np.asarray(xvar, dtype=np.float64)
        if x_values.size != residx.size:
            raise ValueError("xvar must have the same length as residuals")

    if xcut_points is None:
        cuts = np.linspace(float(np.min(x_values)), float(np.max(x_values)), int(n_inter) + 1, dtype=np.float64)
    else:
        supplied = np.asarray(xcut_points, dtype=np.float64).ravel()
        cuts = np.concatenate(([float(np.min(x_values))], supplied, [float(np.max(x_values))]))
    cuts = np.unique(cuts)
    if cuts.size < 2:
        raise ValueError("need at least two cut points to define intervals")

    interval_labels: list[str] = []
    z_rows: list[list[float]] = []
    q_rows: list[list[float]] = []
    for index in range(cuts.size - 1):
        left = cuts[index]
        right = cuts[index + 1]
        if index == cuts.size - 2:
            mask = (x_values >= left) & (x_values <= right)
        else:
            mask = (x_values >= left) & (x_values < right)
        subset = residx[mask]
        stats = _qtest(subset)
        interval_labels.append(f"{left:.4g} to {right:.4g}")
        z_rows.append([stats["z1"], stats["z2"], stats["z3"], stats["z4"], stats["AgostinoK2"], stats["N"]])
        q_rows.append([stats["Q1"], stats["Q2"], stats["Q3"], stats["Q4"], stats["AgostinoK2"], stats["N"]])

    z_matrix = np.asarray(z_rows, dtype=np.float64)
    q_matrix = np.asarray(q_rows, dtype=np.float64)
    interval_count = z_matrix.shape[0]
    total_q = np.nansum(q_matrix[:, :5], axis=0)
    total_n = np.nansum(q_matrix[:, 5])
    dfs = np.array([interval_count, interval_count, interval_count, interval_count, 2 * interval_count], dtype=np.float64)
    p_values = np.array([_chi2_sf(total_q[i], dfs[i]) for i in range(5)], dtype=np.float64)
    totals = {
        "Q_stats": np.concatenate((total_q, [total_n])).astype(np.float64),
        "df": np.concatenate((dfs, [0.0])).astype(np.float64),
        "p_value": np.concatenate((p_values, [0.0])).astype(np.float64),
    }
    return QStatsResult(
        interval_labels=tuple(interval_labels),
        z_matrix=z_matrix,
        q_matrix=q_matrix,
        totals=totals,
    )


# R-style exact-name alias
Q_stats = q_stats

__all__ = [
    "QStatsResult",
    "Q_stats",
    "q_stats",
]
