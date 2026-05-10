"""R-aligned utility operations and extra methods.

R source reference:
- file: `gamlss/R/extra.R`
- functions: `refit`, `fitted.gamlss`, `coef.gamlss`, `coefAll`,
  `residuals.gamlss`, `deviance.gamlss`, `lp`, `fv`, `model.frame.gamlss`,
  `terms.gamlss`, `model.matrix.gamlss`, `formula.gamlss`, `is.gamlss`,
  `IC`, `GAIC.gamlss`, `GAIC_table`, `GAIC_scaled`, `Rsq.gamlss`,
  `.hat.WX`, `numeric.deriv`
"""

from .methods import rsq
from .operations import (
    coef,
    coef_all,
    deviance,
    fitted,
    formula,
    fv,
    gaic,
    gaic_scaled,
    gaic_table,
    hat_wx,
    ic,
    is_gamlss,
    lp,
    lpred,
    model_frame,
    model_matrix,
    numeric_deriv,
    refit,
    residuals,
    terms,
)

coefAll = coef_all
IC = ic
GAIC = gaic
GAIC_table = gaic_table
GAIC_scaled = gaic_scaled
Rsq = rsq

__all__ = [
    "GAIC",
    "GAIC_scaled",
    "GAIC_table",
    "IC",
    "Rsq",
    "coef",
    "coefAll",
    "coef_all",
    "deviance",
    "fitted",
    "formula",
    "fv",
    "gaic",
    "gaic_scaled",
    "gaic_table",
    "hat_wx",
    "ic",
    "is_gamlss",
    "lp",
    "lpred",
    "model_frame",
    "model_matrix",
    "numeric_deriv",
    "refit",
    "residuals",
    "rsq",
    "terms",
]
