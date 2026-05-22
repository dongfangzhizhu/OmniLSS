"""R-aligned smoother extraction helpers.

R reference: `gamlss/R/getSmo-*.R::getSmo` and `edfAll`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


@dataclass(frozen=True)
class SmootherInfo:
    parameter: str
    variable: str
    smoother: str
    edf: float | None
    lambda_: float | None


def getSmo(object: GAMLSSModel, parameter: str = "mu") -> tuple[SmootherInfo, ...]:
    """Extract smoother metadata for a parameter from fitted model slots."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    if parameter not in object.par:
        raise ValueError(f"{parameter!r} is not a parameter in the object")

    infos: list[SmootherInfo] = []
    for entry in object.get_smooth_info(parameter):
        infos.append(
            SmootherInfo(
                parameter=parameter,
                variable=str(entry.get("variable", "")),
                smoother=str(entry.get("smoother", "")),
                edf=float(entry["edf"]) if entry.get("edf") is not None else None,
                lambda_=float(entry["lambda_"]) if entry.get("lambda_") is not None else None,
            )
        )
    return tuple(infos)


def edfAll(object: GAMLSSModel) -> dict[str, float]:
    """Return total effective degrees of freedom by parameter."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    out: dict[str, float] = {}
    for parameter in object.par:
        vals = [s.edf for s in getSmo(object, parameter=parameter) if s.edf is not None]
        out[parameter] = float(np.sum(vals)) if vals else 0.0
    return out


__all__ = ["getSmo", "edfAll", "SmootherInfo"]
