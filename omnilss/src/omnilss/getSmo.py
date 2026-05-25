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
    index: int


def getSmo(object: GAMLSSModel, parameter: str = "mu", require_finite: bool = True) -> tuple[SmootherInfo, ...]:
    """Extract smoother metadata for a parameter from fitted model slots."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    if parameter not in object.par:
        raise ValueError(f"{parameter!r} is not a parameter in the object")

    infos: list[SmootherInfo] = []
    for idx, entry in enumerate(object.get_smooth_info(parameter)):
        edf = float(entry["edf"]) if entry.get("edf") is not None else None
        lam = float(entry["lambda_"]) if entry.get("lambda_") is not None else None
        if require_finite:
            for name, value in (("edf", edf), ("lambda_", lam)):
                if value is not None and not np.isfinite(value):
                    raise ValueError(f"{name} for smoother #{idx} in parameter {parameter!r} must be finite")

        infos.append(
            SmootherInfo(
                parameter=parameter,
                variable=str(entry.get("variable", "")),
                smoother=str(entry.get("smoother", "")),
                edf=edf,
                lambda_=lam,
                index=idx,
            )
        )
    return tuple(infos)


def edfAll(object: GAMLSSModel, require_finite: bool = True) -> dict[str, float]:
    """Return total effective degrees of freedom by parameter."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    out: dict[str, float] = {}
    for parameter in object.par:
        vals = [s.edf for s in getSmo(object, parameter=parameter, require_finite=require_finite) if s.edf is not None]
        out[parameter] = float(np.sum(vals)) if vals else 0.0
    return out


__all__ = ["getSmo", "edfAll", "edf_summary", "SmootherInfo"]


def edf_summary(object: GAMLSSModel, require_finite: bool = True) -> dict[str, Any]:
    """Return per-parameter and global edf summary payload."""
    per_parameter = edfAll(object, require_finite=require_finite)
    total = float(np.sum(list(per_parameter.values()))) if per_parameter else 0.0
    share = {k: (v / total if total > 0.0 else 0.0) for k, v in per_parameter.items()}
    return {"per_parameter": per_parameter, "total": total, "share": share}
