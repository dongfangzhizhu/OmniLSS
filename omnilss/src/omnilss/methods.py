"""Compatibility re-export layer for fitted-model methods.

This module re-exports implementations from R-aligned submodules.
Actual implementations have been migrated to:
- `logLik.py` for logLik.gamlss
- `vcov_gamlss.py` for vcov.gamlss
- `SUMMARY.py` for summary.gamlss
- `print.py` for print.gamlss
- `confint_gamlss_29_06_22.py` for confint.gamlss
- `predict_gamlss_23_12_21.py` for predict.gamlss
- `acfResid.py` for acfResid
- `plot.py` for plot.gamlss, qq_stats, get_rqres_samples, worm_plot_data
- `qstats.py` for Q.stats
- `Rsq.py` for Rsq.gamlss
- `LR_test_12_06_2013.py` for LR.test
- `VuongClarkTest.py` for VC.test
- `MODEL_comparison.py` for compare_models, gaic_weights
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.scipy.special import gammaincc
import numpy as np
from statistics import NormalDist

from .distributions import resolve_family
from .model import GAMLSSModel
from .operations import coef, fitted, formula, gaic, is_gamlss, lp, lpred, refit, residuals, terms

# Re-export from R-aligned modules
from .logLik import LogLikResult, log_likelihood, logLik
from .vcov_gamlss import vcov
from .SUMMARY import SummaryResult, summary
from .acfResid import ResidualACFResult, acf_residuals
from .plot import (
    PlotDiagnosticsResult,
    QQStatsResult,
    RQResSamplesResult,
    WormPlotResult,
    get_rqres_samples,
    plot_diagnostics,
    qq_stats,
    worm_plot_data,
    _kernel_density,
)
from .qstats import QStatsResult, q_stats
from .Rsq import RSquaredResult, rsq
from .LR_test_12_06_2013 import LRTestResult, likelihood_ratio_test, _chi2_sf
from .VuongClarkTest import VuongClarkeResult, vuong_clarke_test, _binom_two_sided_p_value
from .MODEL_comparison import (
    GAICWeightRow,
    GAICWeightsResult,
    ModelComparisonResult,
    ModelComparisonRow,
    compare_models,
    gaic_weights,
)
from .pdfplot import PDFPlotEntry, PDFPlotResult, pdf_plot_data, pdf_plot
from .HistDist_03_10_13 import HistDistResult, hist_dist_data, histDist
from .getQuantile import QuantileCurveEntry, QuantileCurveResult, get_quantile_data, getQuantile
from .centilesPLOT import (
    CentileCurveEntry,
    CentilesResult,
    CentilesCoverageRow,
    CentilesCoverageResult,
    CentilesSplitPanel,
    CentilesSplitResult,
    CentilesSplitCoverageResult,
    centiles_data,
    centiles_coverage_data,
    centiles_split_data,
    centiles_split_coverage_data,
    centiles,
    centiles_split,
)
from .centilesFan import CentileFanBand, CentilesFanResult, centiles_fan_data, centiles_fan
from .centilescom import (
    CentilesComparisonModel,
    CentilesComparisonResult,
    CentilesComparisonCoverageResult,
    centiles_com,
    centiles_comparison_data,
    centiles_comparison_coverage_data,
)
from .centile_pred_30_06_22 import CentilePredEntry, CentilePredResult, centiles_pred, centile_pred_data
from .term_plot_new import TermPlotEntry, TermPlotResult, term_plot_data, term_plot
from .fitted_plot import FittedPlotEntry, FittedPlotResult, fitted_plot_data, fitted_plot
from .getPEF import PartialEffectResult, get_pef_data, getPEF
from .plot2way import Plot2WayResult, plot2way_data, plot2way
from .print import print_model
from .confint_gamlss_29_06_22 import confint
from .predict_gamlss_23_12_21 import (
    _build_new_design_matrix,
    _parameter_block_covariance,
    _predict_single_parameter,
    predict,
    update_model,
)
from .stepGAICAll_B_Parallel import (
    MultiParameterScopeResult,
    MultiParameterScopeRow,
    StepGAICAllResult,
    StepGAICAllStep,
    addterm_all,
    dropterm_all,
    step_gaic_all,
)






def _require_gamlss_method(object: GAMLSSModel) -> None:
    if not is_gamlss(object):
        raise TypeError("This is not an gamlss object")


def _unwrap_validation_model(object: Any) -> GAMLSSModel:
    if isinstance(object, GAMLSSModel):
        return object
    model = getattr(object, "model", None)
    if isinstance(model, GAMLSSModel):
        return model
    raise TypeError("This is not an gamlss object")


def _normal_two_sided_pvalue(z_value: np.ndarray) -> np.ndarray:
    distribution = NormalDist()
    flattened = np.asarray(np.abs(z_value), dtype=np.float64).ravel()
    pvals = np.array(
        [2.0 * (1.0 - distribution.cdf(float(value))) for value in flattened],
        dtype=np.float64,
    )
    return pvals.reshape(np.asarray(z_value).shape)


def _ordered_residuals(object: GAMLSSModel) -> np.ndarray:
    values = np.asarray(residuals(object, what="z-scores"), dtype=np.float64).ravel()
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


# NOTE: _autocorrelation, _kernel_density, _chi2_sf, _binom_two_sided_p_value
# have been moved to their respective R-aligned modules (acfResid.py, plot.py,
# LR_test_12_06_2013.py, VuongClarkTest.py). They are re-imported at the top of
# this file for backward compatibility.


def _normalize_formula_text(value: Any) -> str:
    return str(value).strip()


def _rhs_terms(formula_text: str) -> list[str]:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        raise ValueError("formula must contain '~'")
    rhs = normalized.split("~", 1)[1].strip()
    if rhs in {"", "1"}:
        return []
    return [term.strip() for term in rhs.split("+") if term.strip() and term.strip() != "1"]


def _response_name_from_formula(formula_text: str) -> str:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        raise ValueError("formula must contain '~'")
    return normalized.split("~", 1)[0].strip()


def _compose_formula(response: str, terms_list: list[str]) -> str:
    rhs = " + ".join(terms_list) if terms_list else "1"
    return f"{response} ~ {rhs}"


def _formula_rhs_spec(formula_text: str) -> str:
    normalized = _normalize_formula_text(formula_text)
    if "~" not in normalized:
        return normalized
    return f"~ {normalized.split('~', 1)[1].strip()}"


def _broadcast_parameter(values: np.ndarray, shape: tuple[int, ...], *, fill_nan: float, fill_posinf: float, fill_neginf: float, lower: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 1 and int(np.prod(shape)) > 1:
        array = np.full(shape, array.item(), dtype=np.float64)
    else:
        array = array.reshape(shape)
    array = np.nan_to_num(array, nan=fill_nan, posinf=fill_posinf, neginf=fill_neginf)
    if lower is not None:
        array = np.maximum(array, lower)
    return array


def _boxcox_inverse_quantile(mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray, z_score: np.ndarray) -> np.ndarray:
    safe_mu = np.maximum(mu, np.finfo(np.float64).eps)
    safe_sigma = np.maximum(sigma, np.finfo(np.float64).eps)
    safe_nu = np.where(np.isfinite(nu), nu, 0.0)
    near_zero = np.abs(safe_nu) < 1e-6
    ratio = np.empty_like(safe_mu, dtype=np.float64)
    if np.any(near_zero):
        ratio[near_zero] = np.exp(safe_sigma[near_zero] * z_score[near_zero])
    if np.any(~near_zero):
        nonzero_nu = safe_nu[~near_zero]
        base = 1.0 + nonzero_nu * safe_sigma[~near_zero] * z_score[~near_zero]
        base = np.maximum(base, np.finfo(np.float64).eps)
        ratio[~near_zero] = np.power(base, 1.0 / nonzero_nu)
    ratio = np.nan_to_num(ratio, nan=np.finfo(np.float64).eps, posinf=1e12, neginf=np.finfo(np.float64).eps)
    return safe_mu * ratio


def _family_quantile_values(
    family_name: str,
    parameters: tuple[str, ...],
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    tau: np.ndarray,
    probabilities: Sequence[float],
) -> list[np.ndarray]:
    normal = NormalDist()
    positive_support_families = {"GA", "IG", "WEI", "EXP", "LOGNO", "BCCG", "BCT", "BCPE"}
    quantiles: list[np.ndarray] = []
    for probability in probabilities:
        z_score = np.full(mu.shape, normal.inv_cdf(float(probability)), dtype=np.float64)
        if family_name == "LOGNO":
            log_quantile = np.clip(mu + sigma * z_score, -700.0, 700.0)
            quantile = np.exp(log_quantile)
        elif family_name in {"BCCG", "BCT", "BCPE"}:
            quantile = _boxcox_inverse_quantile(mu, sigma, nu, z_score)
        elif family_name == "JSU":
            safe_tau = np.maximum(tau, np.finfo(np.float64).eps)
            quantile = mu + sigma * np.sinh((z_score - nu) / safe_tau)
        elif "sigma" in parameters:
            quantile = mu + sigma * z_score
        else:
            quantile = mu
        if family_name in positive_support_families:
            quantile = np.maximum(quantile, np.finfo(np.float64).eps)
        quantiles.append(np.asarray(quantile, dtype=np.float64))
    return quantiles


def _predict_parameter_frame(
    object: GAMLSSModel,
    xname: str,
    xvalues: np.ndarray,
    fixed_at: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    call_data = None if object.call is None else object.call.get("data")
    if call_data is None:
        raise ValueError("prediction data requires call['data']")
    predictor_names: set[str] = set()
    for parameter in object.par:
        formula_text = object.formulas.get(parameter)
        if formula_text is None:
            continue
        predictor_names.update(_rhs_terms(formula_text))
    predictor_names.discard(xname)
    predictor_names.discard(".")

    grid_data: dict[str, np.ndarray] = {xname: np.asarray(xvalues, dtype=np.float64).ravel()}
    chosen_fixed: dict[str, float] = {}
    explicit_fixed = {} if fixed_at is None else {str(key): float(value) for key, value in fixed_at.items()}
    for name in sorted(predictor_names):
        if name in explicit_fixed:
            value = explicit_fixed[name]
        else:
            source = np.asarray(call_data[name], dtype=np.float64).ravel()
            value = float(np.nanmedian(source))
        chosen_fixed[name] = value
        grid_data[name] = np.full(grid_data[xname].shape, value, dtype=np.float64)

    predicted = predict_all(object, newdata=grid_data, type="response", output="data.frame")
    shape = grid_data[xname].shape
    parameters = {
        "mu": _broadcast_parameter(predicted.get("mu", np.zeros(shape, dtype=np.float64)), shape, fill_nan=np.finfo(np.float64).eps, fill_posinf=1e12, fill_neginf=np.finfo(np.float64).eps, lower=np.finfo(np.float64).eps),
        "sigma": _broadcast_parameter(predicted.get("sigma", np.ones(shape, dtype=np.float64)), shape, fill_nan=1.0, fill_posinf=1e6, fill_neginf=np.finfo(np.float64).eps, lower=np.finfo(np.float64).eps),
        "nu": _broadcast_parameter(predicted.get("nu", np.zeros(shape, dtype=np.float64)), shape, fill_nan=0.0, fill_posinf=50.0, fill_neginf=-50.0),
        "tau": _broadcast_parameter(predicted.get("tau", np.ones(shape, dtype=np.float64)), shape, fill_nan=1.0, fill_posinf=1e6, fill_neginf=np.finfo(np.float64).eps, lower=np.finfo(np.float64).eps),
    }
    return parameters, chosen_fixed


def _fit_with_updated_formula(
    object: GAMLSSModel,
    formula_updates: dict[str, str],
) -> GAMLSSModel:
    from .fitting import gamlss, gamlss_ml

    if object.call is None or "data" not in object.call:
        raise ValueError("object must contain stored call data")

    mu_formula = formula_updates.get("mu", _normalize_formula_text(object.formulas.get("mu", "")))
    sigma_formula = formula_updates.get("sigma", _normalize_formula_text(object.formulas.get("sigma", "~1")))
    family = object.family
    data = object.call["data"]
    weights = object.weights
    control = object.control
    method_name = str(object.additional_slots.get("method", "RS")).upper()
    parameter_formulas = {parameter: _normalize_formula_text(value) for parameter, value in object.formulas.items()}
    parameter_formulas.update(formula_updates)

    if method_name == "ML":
        return gamlss_ml(
            formula=mu_formula,
            sigma_formula=_formula_rhs_spec(sigma_formula),
            parameter_formulas={parameter: _formula_rhs_spec(value) for parameter, value in parameter_formulas.items()},
            family=family,
            data=data,
            weights=weights,
        )
    return gamlss(
        formula=mu_formula,
        sigma_formula=_formula_rhs_spec(sigma_formula),
        parameter_formulas={parameter: _formula_rhs_spec(value) for parameter, value in parameter_formulas.items()},
        family=family,
        data=data,
        weights=weights,
        method=method_name,
        control=None if not isinstance(control, dict) else None,
    )


def _scope_terms(object: GAMLSSModel, what: str, scope: Any | None) -> list[str]:
    available_terms = list(terms(object, what=what).get("term_labels", []))
    if scope is None:
        return available_terms
    if isinstance(scope, str):
        normalized = scope.strip()
        if normalized.startswith("~"):
            candidate_terms = _rhs_terms(f"response {normalized}")
        else:
            candidate_terms = [term.strip() for term in normalized.split("+") if term.strip()]
    else:
        candidate_terms = [str(term).strip() for term in scope if str(term).strip()]
    return [term for term in candidate_terms if term in available_terms or scope is not None]


def _updated_formulas_for_parameter(
    object: GAMLSSModel,
    what: str,
    updated_formula: str,
) -> dict[str, str]:
    formulas = {parameter: _normalize_formula_text(value) for parameter, value in object.formulas.items()}
    formulas[what] = updated_formula
    return formulas


def _parse_scope_spec(scope: Any | None) -> tuple[list[str], list[str]]:
    """Parse staged lower/upper scope specifications."""

    if scope is None:
        return [], []
    if isinstance(scope, dict):
        lower_raw = scope.get("lower")
        upper_raw = scope.get("upper")
        lower = _rhs_terms(f"response {lower_raw}") if isinstance(lower_raw, str) and str(lower_raw).strip().startswith("~") else (
            [str(term).strip() for term in lower_raw if str(term).strip()] if lower_raw is not None else []
        )
        upper = _rhs_terms(f"response {upper_raw}") if isinstance(upper_raw, str) and str(upper_raw).strip().startswith("~") else (
            [str(term).strip() for term in upper_raw if str(term).strip()] if upper_raw is not None else []
        )
        return lower, upper
    if isinstance(scope, str) and scope.strip().startswith("~"):
        parsed = _rhs_terms(f"response {scope}")
        return [], parsed
    parsed = [str(term).strip() for term in scope if str(term).strip()]
    return [], parsed


# NOTE: _model_loglik_increments, _qtest, acf_residuals, qq_stats,
# get_rqres_samples, worm_plot_data, plot_diagnostics, q_stats, rsq,
# likelihood_ratio_test, vuong_clarke_test, compare_models, gaic_weights
# have been migrated to their respective R-aligned modules and are re-imported
# at the top of this file for backward compatibility.








from .predictAll_22_08_22 import PredictAllResult, predict_all
from .prodist import ProDistResult, prodist_data
from .fitDist import FitDistResult, FitDistRow, fit_dist, fit_dist_data
from .fitDistPred import (
    FitDistPredResult,
    FitDistPredRow,
    GAMLSSMLPredResult,
    fit_dist_pred,
    fit_dist_pred_data,
    gamlss_ml_pred,
    gamlss_ml_pred_data,
)
from .gamlssVGD_23_12_21 import (
    CVComparisonResult,
    CVComparisonRow,
    GAMLSSCVResult,
    GAMLSSVGDResult,
    TGDResult,
    VGDComparisonResult,
    VGDComparisonRow,
    cv,
    gamlss_cv,
    gamlss_cv_data,
    gamlss_vgd,
    gamlss_vgd_data,
    get_tgd,
    get_tgd_data,
    is_gamlss_cv,
    is_gamlss_vgd,
    vgd,
)
from .chooseDistParallel import (
    ChooseDistOrderResult,
    ChooseDistOrderRow,
    ChooseDistPredResult,
    ChooseDistResult,
    choose_dist_data,
    choose_dist_pred_data,
    get_order,
)
from .stepTGD import (
    ExtractTGDResult,
    MultiParameterTGDScopeResult,
    MultiParameterTGDScopeRow,
    StepTGDAllResult,
    StepTGDAllStep,
    StepTGDResult,
    StepTGDStep,
    TGDScopeResult,
    TGDScopeRow,
    add1_tgd,
    add1_tgd_all,
    add1_tgdp,
    drop1_tgd,
    drop1_tgd_all,
    drop1_tgdp,
    extract_tgd,
    extract_tgd_data,
    step_tgd,
    step_tgd_all,
)
from .stepGAIC_03_10_13 import ExtractAICResult, StepGAICResult, StepGAICStep, extract_aic
from .DropAddStepGAIC_Parallel import ScopeSelectionResult, ScopeSelectionRow, addterm, dropterm, step_gaic



def _subset_mapping_rows(data, mask):
    """Subset mapping-like data rows with a boolean mask."""
    import numpy as _np
    subset = {}
    for key, value in data.items():
        array = _np.asarray(value)
        if array.ndim == 0:
            subset[key] = value
            continue
        if array.shape[0] != mask.shape[0]:
            subset[key] = value
            continue
        subset[key] = array[mask]
    return subset
