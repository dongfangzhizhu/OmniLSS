"""R-aligned prediction interfaces.

R source reference:
- file: `gamlss/R/predict.gamlss_23_12_21.R`
- functions: `predict.gamlss`
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .model import GAMLSSModel


def _build_new_design_matrix(
    object: GAMLSSModel,
    what: str,
    newdata: dict[str, Any],
) -> np.ndarray:
    from .operations import coef

    term_info = object.terms.get(what)
    if term_info is None:
        raise ValueError(f"no stored terms for parameter {what}")
    coef_size = np.asarray(coef(object, what), dtype=np.float64).ravel().size
    n = len(np.asarray(next(iter(newdata.values()))))
    columns = [np.ones(n, dtype=np.float64)]
    for label in term_info.get("term_labels", []):
        if len(columns) >= coef_size:
            break
        if label not in newdata:
            raise KeyError(f"newdata is missing predictor {label!r}")
        columns.append(np.asarray(newdata[label], dtype=np.float64))
    if len(columns) != coef_size:
        raise ValueError(
            f"stored design for parameter {what!r} expects {coef_size} columns but newdata produced {len(columns)}"
        )
    return np.column_stack(columns)


def _parameter_block_covariance(object: GAMLSSModel, what: str) -> np.ndarray | None:
    from .operations import coef
    from .vcov_gamlss import vcov

    cov = vcov(object, type="vcov")
    offset = 0
    for parameter in object.par:
        values = np.asarray(coef(object, parameter), dtype=np.float64).ravel()
        size = values.size
        if parameter == what:
            local_cov = cov[offset : offset + size, offset : offset + size]
            if local_cov.size and not np.isnan(local_cov).all():
                return local_cov
            return None
        offset += size
    return None


def _predict_single_parameter(
    object: GAMLSSModel,
    what: str,
    type: str,
    terms: tuple[str, ...] | None,
    se_fit: bool,
    newdata: dict[str, Any] | None,
) -> Any:
    from .operations import coef, fitted, lp, lpred

    if newdata is not None:
        x_new = _build_new_design_matrix(object, what, newdata)
        beta = np.asarray(coef(object, what), dtype=np.float64).ravel()
        pred = x_new @ beta
        se = None
        if se_fit:
            local_cov = _parameter_block_covariance(object, what)
            if local_cov is not None:
                se = np.sqrt(np.einsum("ij,jk,ik->i", x_new, local_cov, x_new))
        if type == "terms":
            term_labels = list(object.terms.get(what, {}).get("term_labels", []))
            if not term_labels:
                result = np.zeros((x_new.shape[0], 0), dtype=np.float64)
            else:
                term_matrix = x_new[:, 1:] * beta[1:]
                if terms is not None:
                    indices = [term_labels.index(term_name) for term_name in terms]
                    term_matrix = term_matrix[:, indices]
                result = term_matrix
            if se_fit:
                return {"fit": result, "se.fit": None}
            return result

        if type == "response":
            family = object.family
            if hasattr(family, "link_inverses") and family.link_inverses and what in family.link_inverses:
                pred = np.asarray(
                    family.link_inverses[what](jnp.asarray(pred, dtype=jnp.float64)),
                    dtype=np.float64,
                )
        if se_fit:
            return {"fit": pred, "se.fit": se}
        return pred

    if type == "response":
        result = fitted(object, what=what)
        if se_fit:
            return {"fit": result, "se.fit": None}
        return result
    if type == "link" and se_fit:
        link_values = np.asarray(lp(object, what=what), dtype=np.float64)
        local_cov = _parameter_block_covariance(object, what)
        design = object.design_matrices.get(what)
        se = None
        if local_cov is not None and design is not None:
            x_design = np.asarray(design, dtype=np.float64)
            if x_design.ndim == 2 and x_design.shape[0] == link_values.shape[0]:
                se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x_design, local_cov, x_design), 0.0))
        elif local_cov is not None and local_cov.shape == (1, 1):
            se = np.repeat(np.sqrt(float(local_cov[0, 0])), link_values.shape[0]).astype(np.float64)
        return {"fit": link_values, "se.fit": se}
    return lpred(object, what=what, type=type, terms=terms, se_fit=se_fit)


def predict(
    object: GAMLSSModel,
    what: str = "mu",
    type: str = "response",
    terms: tuple[str, ...] | None = None,
    se_fit: bool = False,
    newdata: dict[str, Any] | None = None,
) -> Any:
    """Staged predict method using existing helpers and linear newdata support."""

    from .methods import _require_gamlss_method

    _require_gamlss_method(object)
    if what == "all":
        return {
            parameter: _predict_single_parameter(
                object,
                what=parameter,
                type=type,
                terms=terms,
                se_fit=se_fit,
                newdata=newdata,
            )
            for parameter in object.par
        }
    return _predict_single_parameter(
        object,
        what=what,
        type=type,
        terms=terms,
        se_fit=se_fit,
        newdata=newdata,
    )


def update_model(object: GAMLSSModel, **kwargs: Any) -> Any:
    """Thin wrapper over the staged `refit` behavior."""

    from .methods import _require_gamlss_method
    from .operations import refit

    _require_gamlss_method(object)
    return refit(object, **kwargs)


__all__ = [
    "_build_new_design_matrix",
    "_parameter_block_covariance",
    "_predict_single_parameter",
    "predict",
    "update_model",
]
