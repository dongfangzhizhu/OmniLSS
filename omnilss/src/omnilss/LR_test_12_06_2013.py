"""R-aligned likelihood-ratio testing interfaces.

R source reference:
- file: `gamlss/R/LR-test-12-06-2013.R`
- functions: `LR.test`
"""

from __future__ import annotations

from dataclasses import dataclass

from jax.scipy.special import gammaincc

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class LRTestResult:
    """Structured likelihood-ratio test result for nested staged models."""

    chi_square: float
    df: float
    p_value: float
    null_deviance: float
    alternative_deviance: float
    null_df_fit: float
    alternative_df_fit: float


def _require_gamlss(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _chi2_sf(value: float, df: float) -> float:
    if df <= 0:
        return float("nan")
    return float(gammaincc(df / 2.0, value / 2.0))


def likelihood_ratio_test(
    null: GAMLSSModel,
    alternative: GAMLSSModel,
) -> LRTestResult:
    """R reference: `gamlss/R/LR-test-12-06-2013.R::LR.test`.

    Staged behavior:
    - Assumes the models are nested and does not verify that assumption.
    - Returns a structured result instead of printing.
    """

    _require_gamlss(null)
    _require_gamlss(alternative)

    d0 = float(null.g_dev)
    d1 = float(alternative.g_dev)
    if d1 > d0:
        raise ValueError("The null model has smaller deviance than the alternative")

    df0 = float(null.df_fit)
    df1 = float(alternative.df_fit)
    df = df1 - df0
    if df < 0:
        raise ValueError("The difference in fitted degrees of freedom is negative")

    chi_square = d0 - d1
    p_value = _chi2_sf(chi_square, df)
    return LRTestResult(
        chi_square=float(chi_square),
        df=float(df),
        p_value=float(p_value),
        null_deviance=d0,
        alternative_deviance=d1,
        null_df_fit=df0,
        alternative_df_fit=df1,
    )


# R-style exact-name alias
LR_test = likelihood_ratio_test

__all__ = [
    "LRTestResult",
    "LR_test",
    "likelihood_ratio_test",
]
