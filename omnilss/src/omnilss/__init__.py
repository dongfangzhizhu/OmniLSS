"""JAX reimplementation of selected `gamlss` R package functionality."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("omnilss")
except PackageNotFoundError:
    __version__ = "0.3.0"

from . import config as _config
from . import config  # public: omnilss.config.GPU_CROSSOVER_N etc.
from .config import auto_select_method, get_config_summary
from .centilesFan import centiles_fan
from .chooseDistParallel import chooseDist, chooseDistPred, getOrder
from .controls import GAMLSSControl, GLIMControl, gamlss_control, glim_control
from .DevianceIncr import devianceIncr
from .distributions import (
    BCCG,
    BCPE,
    BCT,
    BE,
    BI,
    EXP,
    GA,
    GEOM,
    IG,
    JSU,
    LO,
    LOGNO,
    NBI,
    NO,
    PO,
    TF,
    WEI,
    ZIP,
    BetaFamily,
    BinomialFamily,
    BoxCoxColeGreenFamily,
    BoxCoxPowerExponentialFamily,
    BoxCoxTFamily,
    ExponentialFamily,
    GammaFamily,
    GeometricFamily,
    InverseGaussianFamily,
    JohnsonSUFamily,
    LogisticFamily,
    LogNormalFamily,
    NegativeBinomialFamily,
    NormalFamily,
    PoissonFamily,
    StudentTFamily,
    WeibullFamily,
    ZeroInflatedPoissonFamily,
    resolve_family,
)
from .distributions_b1 import GU, IGAMMA, NBII, PARETO2, RG
from .distributions_b2 import LOGNO2, NO2, PE, SIMPLEX, exGAUS
from .distributions_b3 import GT, SHASH, SN1, SN2
from .distributions_b4 import BEINF, BEINF0, BEINF1, BEOI, BEZI
from .distributions_b5 import ZAGA, ZAIG, ZAP, ZIP2
from .distributions_b6 import DEL, DPO, PIG, SI, SICHEL, WARING, YULE
from .distributions_b7 import BB, BNB, MN3, MN4, MN5
from .distributions_b8 import GB2, GG, LNO, NET, PARETO
from .distributions_b9 import LG, ZALG, ZIPF
from .distributions_b10_zero_variants import (
    ZABB,
    ZABI,
    ZABNB,
    ZANBI,
    ZAPIG,
    ZASICHEL,
    ZAZIPF,
    ZIBB,
    ZIBI,
    ZIBNB,
    ZIPIG,
    ZISICHEL,
)
from .distributions_b11 import GB1, GIG, JSU
from .distributions_b12 import GAF, NOF, RGE, WEI2, WEI3, BEo, GEOMo, PARETO2o
from .distributions_b13 import DELAPORT, IGA, LOGITNO, PE2, PIG2, GeneralisedPoisson
from .distributions_b14 import ST5, BCPEo, BCTo, JSUo, SHASHo, SHASHo2
from .distributions_b15 import (
    EGB2,
    GPO,
    LOGSHASH,
    LQNO,
    NBF,
    SEP1,
    SEP2,
    SEP3,
    SEP4,
    SST,
    ST3C,
    ZINBF,
)
from .distributions_b16 import DPO1, PARETO1, SEP, BCCGo, LOGSHASHo, PARETO1o
from .families import FamilyDefinition
from .fitDist import fitDist
from .fitDistPred import fitDistPred, gamlssMLpred
from .fitted_plot import fitted_plot
from .fitting import gamlss, gamlss_ml
from .gamlss_5 import gamlss_control_exact, glim_control_exact
from .gamlssML import gamlssML
from .gamlssVGD_23_12_21 import CV, VGD, gamlssCV, gamlssVGD, getTGD
from .hatvalues import hatvalues
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
    probit_derivative,
    probit_inverse,
    probit_link,
)
from .logLik import logLik
from .lpred import lpred as lpred_exact
from .methods import (
    CentileCurveEntry,
    CentileFanBand,
    CentilePredEntry,
    CentilePredResult,
    CentilesComparisonCoverageResult,
    CentilesComparisonModel,
    CentilesComparisonResult,
    CentilesCoverageResult,
    CentilesCoverageRow,
    CentilesFanResult,
    CentilesResult,
    CentilesSplitCoverageResult,
    CentilesSplitPanel,
    CentilesSplitResult,
    ChooseDistOrderResult,
    ChooseDistOrderRow,
    ChooseDistPredResult,
    ChooseDistResult,
    CVComparisonResult,
    CVComparisonRow,
    ExtractAICResult,
    ExtractTGDResult,
    FitDistPredResult,
    FitDistPredRow,
    FitDistResult,
    FitDistRow,
    FittedPlotEntry,
    FittedPlotResult,
    GAICWeightRow,
    GAICWeightsResult,
    GAMLSSCVResult,
    GAMLSSMLPredResult,
    GAMLSSVGDResult,
    HistDistResult,
    LogLikResult,
    LRTestResult,
    ModelComparisonResult,
    ModelComparisonRow,
    MultiParameterScopeResult,
    MultiParameterScopeRow,
    MultiParameterTGDScopeResult,
    MultiParameterTGDScopeRow,
    PartialEffectResult,
    PDFPlotEntry,
    PDFPlotResult,
    Plot2WayResult,
    PlotDiagnosticsResult,
    PredictAllResult,
    ProDistResult,
    QQStatsResult,
    QStatsResult,
    QuantileCurveEntry,
    QuantileCurveResult,
    ResidualACFResult,
    RQResSamplesResult,
    RSquaredResult,
    ScopeSelectionResult,
    ScopeSelectionRow,
    StepGAICAllResult,
    StepGAICAllStep,
    StepGAICResult,
    StepGAICStep,
    StepTGDAllResult,
    StepTGDAllStep,
    StepTGDResult,
    StepTGDStep,
    SummaryResult,
    TermPlotEntry,
    TermPlotResult,
    TGDResult,
    TGDScopeResult,
    TGDScopeRow,
    VGDComparisonResult,
    VGDComparisonRow,
    VuongClarkeResult,
    WormPlotResult,
    acf_residuals,
    add1_tgd,
    add1_tgd_all,
    add1_tgdp,
    addterm,
    addterm_all,
    centile_pred_data,
    centiles_comparison_coverage_data,
    centiles_comparison_data,
    centiles_coverage_data,
    centiles_data,
    centiles_fan_data,
    centiles_split_coverage_data,
    centiles_split_data,
    choose_dist_data,
    choose_dist_pred_data,
    compare_models,
    confint,
    cv,
    drop1_tgd,
    drop1_tgd_all,
    drop1_tgdp,
    dropterm,
    dropterm_all,
    extract_aic,
    extract_tgd,
    extract_tgd_data,
    fit_dist,
    fit_dist_data,
    fit_dist_pred,
    fit_dist_pred_data,
    fitted_plot_data,
    gaic_weights,
    gamlss_cv,
    gamlss_cv_data,
    gamlss_ml_pred,
    gamlss_ml_pred_data,
    gamlss_vgd,
    gamlss_vgd_data,
    get_order,
    get_pef_data,
    get_quantile_data,
    get_rqres_samples,
    get_tgd,
    get_tgd_data,
    hist_dist_data,
    is_gamlss_cv,
    is_gamlss_vgd,
    likelihood_ratio_test,
    log_likelihood,
    pdf_plot_data,
    plot2way_data,
    plot_diagnostics,
    predict,
    predict_all,
    print_model,
    prodist_data,
    q_stats,
    qq_stats,
    rsq,
    step_gaic,
    step_gaic_all,
    step_tgd,
    step_tgd_all,
    summary,
    term_plot_data,
    update_model,
    vcov,
    vgd,
    vuong_clarke_test,
    worm_plot_data,
)
from .methods import (
    choose_dist_pred_data as choose_dist_pred,
)
from .model import GAMLSSModel
from .operations import (
    coef,
    coef_all,
    deviance,
    deviance_increment,
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
from .predict_gamlss_23_12_21 import predict as predict_gamlss_exact
from .print import print as print_gamlss_exact
from .stepGAIC_03_10_13 import extractAIC, stepGAIC
from .stepGAICAll_A_parallel import stepGAICAll_A
from .stepGAICAll_B_Parallel import addtermAllP, droptermAllP, stepGAICAll_B
from .stepTGD import add1TGD, add1TGDP, drop1TGD, drop1TGDP, extractTGD, stepTGD
from .update import update
from .worm_plot import MultiPanelWormPlotData, WormPlotData, wp, wp_data, wp_interactive

# New modules: scoring rules and sklearn compatibility
from . import scoring
from .scoring import (
    crps,
    log_score,
    dss,
    interval_score,
    coverage,
    pit_histogram,
    scoring_summary,
)
from .sklearn_compat import GAMLSSRegressor

__all__ = [
    "FamilyDefinition",
    "GAMLSSControl",
    "GAMLSSModel",
    "GLIMControl",
    "ExtractAICResult",
    "ExtractTGDResult",
    "FitDistResult",
    "FitDistRow",
    "CVComparisonResult",
    "CVComparisonRow",
    "GAMLSSMLPredResult",
    "GAMLSSCVResult",
    "GAMLSSVGDResult",
    "FitDistPredResult",
    "FitDistPredRow",
    "FittedPlotEntry",
    "FittedPlotResult",
    "HistDistResult",
    "PDFPlotEntry",
    "PDFPlotResult",
    "GAICWeightsResult",
    "GAICWeightRow",
    "LRTestResult",
    "LogLikResult",
    "ModelComparisonResult",
    "ModelComparisonRow",
    "MultiParameterScopeResult",
    "MultiParameterScopeRow",
    "MultiParameterTGDScopeResult",
    "MultiParameterTGDScopeRow",
    "PartialEffectResult",
    "ProDistResult",
    "QuantileCurveEntry",
    "QuantileCurveResult",
    "PredictAllResult",
    "Plot2WayResult",
    "PlotDiagnosticsResult",
    "QStatsResult",
    "QQStatsResult",
    "RSquaredResult",
    "ScopeSelectionResult",
    "ScopeSelectionRow",
    "StepGAICAllResult",
    "StepGAICAllStep",
    "StepGAICResult",
    "StepGAICStep",
    "StepTGDAllResult",
    "StepTGDAllStep",
    "StepTGDResult",
    "StepTGDStep",
    "TGDResult",
    "TGDScopeResult",
    "TGDScopeRow",
    "TermPlotEntry",
    "TermPlotResult",
    "VuongClarkeResult",
    "BB",
    "BCCG",
    "BCCGo",
    "BCPE",
    "BCPEo",
    "BCT",
    "BCTo",
    "BE",
    "BEINF",
    "BEINF0",
    "BEINF1",
    "BEo",
    "BEOI",
    "BEZI",
    "BI",
    "BNB",
    "BetaFamily",
    "BinomialFamily",
    "BoxCoxColeGreenFamily",
    "BoxCoxPowerExponentialFamily",
    "BoxCoxTFamily",
    "DEL",
    "DELAPORT",
    "DPO",
    "DPO1",
    "EGB2",
    "EXP",
    "ExponentialFamily",
    "exGAUS",
    "GA",
    "GAF",
    "GammaFamily",
    "GB1",
    "GB2",
    "GeneralisedPoisson",
    "GEOM",
    "GEOMo",
    "GeometricFamily",
    "GG",
    "GIG",
    "GPO",
    "GT",
    "GU",
    "IG",
    "IGA",
    "InverseGaussianFamily",
    "IGAMMA",
    "JSU",
    "JSUo",
    "JohnsonSUFamily",
    "LG",
    "LNO",
    "LO",
    "LOGITNO",
    "LOGNO",
    "LOGNO2",
    "LOGSHASH",
    "LOGSHASHo",
    "LQNO",
    "LogNormalFamily",
    "LogisticFamily",
    "MN3",
    "MN4",
    "MN5",
    "NBI",
    "NBII",
    "NBF",
    "NegativeBinomialFamily",
    "NET",
    "NO",
    "NO2",
    "NOF",
    "NormalFamily",
    "PARETO",
    "PARETO1",
    "PARETO1o",
    "PARETO2",
    "PARETO2o",
    "PE",
    "PE2",
    "PIG",
    "PIG2",
    "PO",
    "PoissonFamily",
    "ResidualACFResult",
    "RG",
    "RGE",
    "RQResSamplesResult",
    "SEP",
    "SEP1",
    "SEP2",
    "SEP3",
    "SEP4",
    "SHASH",
    "SHASHo",
    "SHASHo2",
    "SI",
    "SICHEL",
    "SIMPLEX",
    "SN1",
    "SN2",
    "SST",
    "ST3C",
    "ST5",
    "SummaryResult",
    "CV",
    "TF",
    "StudentTFamily",
    "VGDComparisonResult",
    "VGDComparisonRow",
    "VGD",
    "WARING",
    "WEI",
    "WEI2",
    "WEI3",
    "WeibullFamily",
    "YULE",
    "ZABNB",
    "ZABB",
    "ZAGA",
    "ZAIG",
    "ZALG",
    "ZANBI",
    "ZABI",
    "ZAP",
    "ZAPIG",
    "ZASICHEL",
    "ZAZIPF",
    "ZIP",
    "ZIP2",
    "ZIBB",
    "ZIBNB",
    "ZIBI",
    "ZINBI",
    "ZINBF",
    "ZIPF",
    "ZIPIG",
    "ZISICHEL",
    "ZeroInflatedPoissonFamily",
    "acf_residuals",
    "add1TGD",
    "add1TGDP",
    "add1_tgd",
    "add1_tgd_all",
    "add1_tgdp",
    "addterm",
    "addtermAllP",
    "addterm_all",
    "coef",
    "coef_all",
    "centiles_data",
    "centiles_coverage_data",
    "centiles_fan_data",
    "centiles_comparison_data",
    "centiles_comparison_coverage_data",
    "centiles_fan",
    "cv",
    "choose_dist_data",
    "choose_dist_pred",
    "choose_dist_pred_data",
    "chooseDist",
    "chooseDistPred",
    "get_order",
    "CentilesCoverageRow",
    "CentilesCoverageResult",
    "centile_pred_data",
    "centiles_split_data",
    "centiles_split_coverage_data",
    "CentileFanBand",
    "CentilesFanResult",
    "CentilePredEntry",
    "CentilePredResult",
    "CentileCurveEntry",
    "CentilesComparisonModel",
    "CentilesComparisonCoverageResult",
    "CentilesComparisonResult",
    "CentilesResult",
    "CentilesSplitPanel",
    "CentilesSplitCoverageResult",
    "CentilesSplitResult",
    "ChooseDistOrderResult",
    "ChooseDistOrderRow",
    "ChooseDistResult",
    "ChooseDistPredResult",
    "compare_models",
    "confint",
    "DELAPORT",
    "deviance",
    "devianceIncr",
    "deviance_increment",
    "dropterm",
    "droptermAllP",
    "dropterm_all",
    "drop1TGD",
    "drop1TGDP",
    "drop1_tgd",
    "drop1_tgd_all",
    "drop1_tgdp",
    "EGB2",
    "extract_aic",
    "extractAIC",
    "extractTGD",
    "extract_tgd",
    "extract_tgd_data",
    "fitted",
    "formula",
    "fv",
    "gaic_weights",
    "gaic",
    "gaic_scaled",
    "gaic_table",
    "get_pef_data",
    "getOrder",
    "get_quantile_data",
    "getTGD",
    "get_tgd",
    "get_tgd_data",
    "gamlssCV",
    "gamlssMLpred",
    "gamlssVGD",
    "gamlss_vgd",
    "gamlss_vgd_data",
    "hist_dist_data",
    "fitDist",
    "fit_dist",
    "fit_dist_data",
    "fitDistPred",
    "fit_dist_pred",
    "fit_dist_pred_data",
    "gamlss_ml_pred",
    "gamlss_ml_pred_data",
    "gamlss_cv",
    "gamlss_cv_data",
    "fitted_plot",
    "fitted_plot_data",
    "pdf_plot_data",
    "get_rqres_samples",
    "gamlss",
    "gamlss_control",
    "gamlss_control_exact",
    "gamlss_ml",
    "glim_control",
    "glim_control_exact",
    "hatvalues",
    "hat_wx",
    "ic",
    "identity_derivative",
    "identity_inverse",
    "identity_link",
    "is_gamlss_cv",
    "is_gamlss_vgd",
    "is_gamlss",
    "likelihood_ratio_test",
    "logLik",
    "log_derivative",
    "log_inverse",
    "log_likelihood",
    "log_link",
    "lp",
    "lpred",
    "lpred_exact",
    "model_frame",
    "model_matrix",
    "numeric_deriv",
    "plot_diagnostics",
    "plot2way_data",
    "predict",
    "predict_gamlss_exact",
    "predict_all",
    "prodist_data",
    "print_gamlss_exact",
    "print_model",
    "q_stats",
    "qq_stats",
    "refit",
    "resolve_family",
    "residuals",
    "rsq",
    "summary",
    "stepGAIC",
    "stepGAICAll_A",
    "stepGAICAll_B",
    "stepTGD",
    "step_gaic",
    "step_gaic_all",
    "step_tgd",
    "step_tgd_all",
    "term_plot_data",
    "terms",
    "update",
    "update_model",
    "vcov",
    "vgd",
    "vuong_clarke_test",
    "worm_plot_data",
    "WormPlotResult",
    # ── Model selection ──
    "compare_distributions",
    "select_best_distribution",
    "stepwise_distribution_selection",
    "quick_distribution_search",
    # ── Prediction API ──
    "predict_params",
    "predict_quantiles",
    "centiles",
    "predict_response",
    # ── Diagnostics API ──
    "quantile_residuals",
    "comprehensive_diagnostics",
    "calibration_check",
    "cooks_distance",
    "ComprehensiveDiagnostics",
    # ── Links registry ──
    "LINK_REGISTRY",
    "get_link",
    "get_link_fn",
    "save_model",
    "load_model",
    "save_model_json",
    "load_model_json",
    "save_model_pickle",
    "load_model_pickle",
]


# Bootstrap methods
from .bootstrap import (
    BootstrapResult,
    compute_confidence_intervals,
    generate_bootstrap_indices,
    nonparametric_bootstrap,
    parametric_bootstrap,
    residual_bootstrap,
)

# Regularization methods
from .regularization import (
    RegularizationResult,
    cross_validate_lambda,
    elastic_net_penalty,
    fit_elastic_net,
    fit_lasso_coordinate_descent,
    fit_regularized,
    fit_ridge,
    l1_penalty,
    l2_penalty,
    soft_threshold,
)

# Model selection
try:
    from .model_selection import (
        compare_distributions,
        quick_distribution_search,
        select_best_distribution,
        stepwise_distribution_selection,
    )
except ImportError as _e:
    import warnings as _warnings

    _warnings.warn(
        f"omnilss.model_selection not available: {_e}. Install optional dependencies or check installation.",
        ImportWarning,
        stacklevel=2,
    )
    compare_distributions = None
    quick_distribution_search = None
    select_best_distribution = None
    stepwise_distribution_selection = None

# Prediction API
try:
    from .prediction import (
        centiles,
        predict_params,
        predict_quantiles,
        predict_response,
    )
except ImportError as _e:
    import warnings as _warnings

    _warnings.warn(
        f"omnilss.prediction not available: {_e}. Install optional dependencies or check installation.",
        ImportWarning,
        stacklevel=2,
    )
    centiles = None
    predict_params = None
    predict_quantiles = None
    predict_response = None

# Diagnostics API
try:
    from .diagnostics import (
        ComprehensiveDiagnostics,
        calibration_check,
        comprehensive_diagnostics,
        cooks_distance,
        quantile_residuals,
    )
except ImportError as _e:
    import warnings as _warnings

    _warnings.warn(
        f"omnilss.diagnostics not available: {_e}. Install optional dependencies or check installation.",
        ImportWarning,
        stacklevel=2,
    )
    ComprehensiveDiagnostics = None
    calibration_check = None
    comprehensive_diagnostics = None
    cooks_distance = None
    quantile_residuals = None

# ── Links 注册表及辅助函数（新增）──
try:
    from .links import (
        LINK_REGISTRY,
        cloglog_derivative,
        cloglog_inverse,
        cloglog_link,
        get_inverse_link_fn,
        get_link,
        get_link_derivative_fn,
        get_link_fn,
        sqrt_derivative,
        sqrt_inverse,
        sqrt_link,
    )
except ImportError as _e:
    import warnings as _warnings

    _warnings.warn(
        f"omnilss.links not available: {_e}. Install optional dependencies or check installation.",
        ImportWarning,
        stacklevel=2,
    )
    LINK_REGISTRY = None
    cloglog_derivative = None
    cloglog_inverse = None
    cloglog_link = None
    get_inverse_link_fn = None
    get_link = None
    get_link_derivative_fn = None
    get_link_fn = None
    sqrt_derivative = None
    sqrt_inverse = None
    sqrt_link = None


# Serialization API
try:
    from .serialization import (
        load_model,
        save_model,
        load_model_json,
        save_model_json,
        load_model_pickle,
        save_model_pickle,
    )
except ImportError as _e:
    import warnings as _warnings

    _warnings.warn(
        f"omnilss.serialization not available: {_e}. Install optional dependencies or check installation.",
        ImportWarning,
        stacklevel=2,
    )
    load_model = None
    save_model = None
    load_model_json = None
    save_model_json = None
    load_model_pickle = None
    save_model_pickle = None


def check_installation() -> dict:
    """Return module availability diagnostics for OmniLSS installation."""
    import importlib

    status: dict[str, str] = {}
    modules = {
        "core": ("omnilss.fitting", "gamlss"),
        "distributions": ("omnilss.distributions", "NO"),
        "smoothers": ("omnilss.smoothers.pb", "pb_smoother"),
        "diagnostics": ("omnilss.diagnostics", "quantile_residuals"),
        "prediction": ("omnilss.prediction", "predict_params"),
        "serialization": ("omnilss.serialization", "save_model_json"),
        "deep": ("omnilss.deep.deep_gamlss", "fit_deep_gamlss"),
        "sklearn": ("omnilss.sklearn_compat", "GAMLSSRegressor"),
        "grpc": ("omnilss.api.grpc.server", "serve"),
    }
    for name, (mod, attr) in modules.items():
        try:
            m = importlib.import_module(mod)
            getattr(m, attr)
            status[name] = "ok"
        except ImportError as e:
            status[name] = f"missing: {e}"
        except Exception as e:  # pragma: no cover - diagnostic path
            status[name] = f"error: {e}"
    return status
