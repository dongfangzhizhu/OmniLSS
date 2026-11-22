"""R-aligned centile-SK plotting payload helper.

R reference: `gamlss.dist/R/momentSK.R::plotCentileSK`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .centileSK import centileSK
from .theoCentileSK import theoCentileSK


@dataclass(frozen=True)
class PlotCentileSKResult:
    centiles: np.ndarray
    empirical: np.ndarray
    theoretical_z: np.ndarray


def plotCentileSK(
    y, centiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)
) -> PlotCentileSKResult:
    """Build paired empirical centiles and theoretical-normal z references."""
    emp = centileSK(y, centiles=centiles)
    theo = theoCentileSK(centiles=centiles)
    return PlotCentileSKResult(
        centiles=np.asarray(emp.centiles, dtype=np.float64),
        empirical=np.asarray(emp.values, dtype=np.float64),
        theoretical_z=np.asarray(theo.z_values, dtype=np.float64),
    )


__all__ = ["plotCentileSK", "PlotCentileSKResult"]
