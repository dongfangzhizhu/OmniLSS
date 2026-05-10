"""R-aligned GAIC stepwise selection interfaces.

R source reference:
- file: `gamlss/R/stepGAIC-03-10-13..R`
- functions: `extractAIC.gamlss`, `stepGAIC`
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class ExtractAICResult:
    """Structured staged `extractAIC` payload."""

    edf: float
    aic: float


@dataclass(frozen=True)
class StepGAICStep:
    """One staged step in a `stepGAIC` search path."""

    step: int
    formula: str
    criterion: float
    change: str
    df_fit: float
    deviance: float


@dataclass(frozen=True)
class StepGAICResult:
    """Structured staged `stepGAIC` result."""

    model: GAMLSSModel
    what: str
    direction: str
    k: float
    steps: tuple[StepGAICStep, ...]


def _require_gamlss_method(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def extract_aic(
    fit: GAMLSSModel,
    k: float = 2.0,
    c: bool = False,
) -> ExtractAICResult:
    """R reference: `gamlss/R/stepGAIC-03-10-13..R::extractAIC.gamlss`."""

    _require_gamlss_method(fit)
    edf = float(fit.df_fit)
    n = max(int(fit.n), 1)
    correction = 0.0
    if k == 2.0 and c and (n - edf - 1.0) != 0.0:
        correction = (2.0 * edf * (edf + 1.0)) / (n - edf - 1.0)
    aic = float(fit.g_dev + edf * k + correction)
    return ExtractAICResult(edf=edf, aic=aic)


from .DropAddStepGAIC_Parallel import step_gaic

extractAIC = extract_aic
stepGAIC = step_gaic

__all__ = [
    "ExtractAICResult",
    "StepGAICResult",
    "StepGAICStep",
    "extractAIC",
    "extract_aic",
    "stepGAIC",
    "step_gaic",
]
