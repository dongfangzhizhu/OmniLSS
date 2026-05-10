"""R-aligned non-nested model comparison interfaces.

R source reference:
- file: `gamlss/R/VuongClarkTest.R`
- functions: `VC.test`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from .model import GAMLSSModel
from .operations import fitted, is_gamlss


@dataclass(frozen=True)
class VuongClarkeResult:
    """Structured non-nested model comparison result."""

    vuong_statistic: float
    vuong_preferred: str
    clarke_b: int
    clarke_p_value: float
    clarke_preferred: str
    nobs: int


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _model_loglik_increments(object: GAMLSSModel) -> np.ndarray:
    kwargs: dict[str, Any] = {"y": np.asarray(object.y, dtype=np.float64)}
    for parameter in object.par:
        kwargs[parameter] = np.asarray(fitted(object, parameter), dtype=np.float64)
    dev = np.asarray(object.family.g_dev_inc(**kwargs), dtype=np.float64)
    weights = np.asarray(object.weights, dtype=np.float64) if object.weights is not None else np.ones_like(dev, dtype=np.float64)
    return -0.5 * dev * weights


def _binom_two_sided_p_value(successes: int, n: int) -> float:
    """R reference: `gamlss/R/VuongClarkTest.R::VC.test` (Clarke test p-value helper)."""
    if n < 0:
        return float("nan")
    if n == 0:
        return 1.0
    smaller_tail = min(successes, n - successes)
    prob = 0.0
    for k in range(0, smaller_tail + 1):
        prob += math.comb(n, k) * (0.5 ** n)
    return float(min(1.0, 2.0 * prob))


def vuong_clarke_test(
    model1: GAMLSSModel,
    model2: GAMLSSModel,
    sig_level: float = 0.05,
) -> VuongClarkeResult:
    """R reference: `gamlss/R/VuongClarkTest.R::VC.test`.

    Staged behavior:
    - Uses per-observation log-likelihood increments derived from `g_dev_inc`.
    - Returns structured Vuong and Clarke test summaries.
    """

    _require_gamlss(model1)
    _require_gamlss(model2)
    if not 0 < sig_level < 1:
        raise ValueError("sig_level must be between 0 and 1")

    li1 = _model_loglik_increments(model1)
    li2 = _model_loglik_increments(model2)
    if li1.shape != li2.shape:
        raise ValueError("models must be fit on the same number of observations")

    l1 = float(np.sum(li1))
    l2 = float(np.sum(li2))
    if l1 == l2:
        raise ValueError("The two competing models have identical log-likelihoods")

    p1 = float(model1.df_fit)
    p2 = float(model2.df_fit)
    n = int(li1.size)
    li12 = li1 - li2
    variance = float(np.var(li12, ddof=1)) if n > 1 else 0.0
    denom = math.sqrt(max(variance * max(n - 1, 1), np.finfo(np.float64).eps))
    vuong = (l1 - l2 - (p1 - p2) * 0.5 * math.log(max(n, 1))) / denom
    critical = NormalDist().inv_cdf(1.0 - sig_level / 2.0)
    if abs(vuong) <= critical:
        vuong_pref = "tie"
    elif vuong > 0:
        vuong_pref = "model1"
    else:
        vuong_pref = "model2"

    li12b = li12 - ((p1 - p2) / (2.0 * max(n, 1))) * math.log(max(n, 1))
    b = int(np.sum(li12b > 0.0))
    clarke_p = _binom_two_sided_p_value(b, n)
    if clarke_p <= sig_level:
        clarke_pref = "model1" if b >= n / 2 else "model2"
    else:
        clarke_pref = "tie"

    return VuongClarkeResult(
        vuong_statistic=float(vuong),
        vuong_preferred=vuong_pref,
        clarke_b=b,
        clarke_p_value=float(clarke_p),
        clarke_preferred=clarke_pref,
        nobs=n,
    )


# R-style exact-name alias
VC_test = vuong_clarke_test

__all__ = [
    "VC_test",
    "VuongClarkeResult",
    "vuong_clarke_test",
]
