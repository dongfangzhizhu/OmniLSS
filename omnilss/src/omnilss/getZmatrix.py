"""R-aligned design-matrix extraction helper.

R reference: `gamlss/R/vcov-gamlss.R::getZmatrix`.
"""

from __future__ import annotations

import numpy as np

from .model import GAMLSSModel
from .operations import is_gamlss


def getZmatrix(object: GAMLSSModel, parameter: str = "mu") -> np.ndarray:
    """Return stored design matrix for a model parameter."""
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    if parameter not in object.par:
        raise ValueError(f"{parameter!r} is not a parameter in the object")
    if parameter not in object.design_matrices:
        raise KeyError(f"no design matrix stored for parameter {parameter!r}")
    return np.asarray(object.design_matrices[parameter], dtype=np.float64)


__all__ = ["getZmatrix"]
