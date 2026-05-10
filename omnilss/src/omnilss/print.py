"""R-aligned print interfaces.

R source reference:
- file: `gamlss/R/print.R`
- functions: `print.gamlss`
"""

from __future__ import annotations

import numpy as np

from .model import GAMLSSModel


def print_model(object: GAMLSSModel, digits: int = 6) -> str:
    """R reference: `gamlss/R/print.R::print.gamlss`."""

    from .methods import _require_gamlss_method
    from .operations import coef, gaic
    from .SUMMARY import summary

    _require_gamlss_method(object)
    parts = [
        f"Family: {getattr(object.family, 'name', object.family)}",
        f"Fitting method: {object.additional_slots.get('method', 'RS')}",
        f"Call: {object.call}",
    ]
    for parameter in object.par:
        values = np.asarray(coef(object, parameter), dtype=np.float64).ravel()
        parts.append(f"{parameter.capitalize()} Coefficients:")
        summary_result = summary(object)
        std_error = summary_result.coefficients.get(parameter, {}).get("std_error")
        t_value = summary_result.coefficients.get(parameter, {}).get("t_value")
        p_value = summary_result.coefficients.get(parameter, {}).get("p_value")
        if std_error is None:
            rendered = ", ".join(f"{value:.{digits}g}" for value in values)
            parts.append(f"  estimate: {rendered}")
        else:
            lines = []
            for index, value in enumerate(values):
                se = np.asarray(std_error, dtype=np.float64).ravel()[index]
                tv = np.asarray(t_value, dtype=np.float64).ravel()[index]
                pv = np.asarray(p_value, dtype=np.float64).ravel()[index]
                lines.append(
                    f"  coef[{index}]: estimate={value:.{digits}g} se={se:.{digits}g} t={tv:.{digits}g} p={pv:.{digits}g}"
                )
            parts.extend(lines)
    residual_df = float(object.additional_slots.get("df.residual", object.n - object.df_fit))
    parts.append(
        f"Degrees of Freedom for the fit: {object.df_fit:.{digits}g} Residual Deg. of Freedom {residual_df:.{digits}g}"
    )
    aic = float(object.additional_slots.get("aic", gaic(object, k=2.0)))
    sbc = float(object.additional_slots.get("sbc", gaic(object, k=float(np.log(max(object.n, 1))))))
    parts.append(f"Global Deviance: {object.g_dev:.{digits}g}")
    parts.append(f"Converged: {bool(object.additional_slots.get('converged', True))}")
    parts.append(f"Cycles: {int(object.additional_slots.get('cycles', object.iter))}")
    parts.append(f"AIC: {aic:.{digits}g}")
    parts.append(f"SBC: {sbc:.{digits}g}")
    return "\n".join(parts)


print = print_model

__all__ = [
    "print",
    "print_model",
]
