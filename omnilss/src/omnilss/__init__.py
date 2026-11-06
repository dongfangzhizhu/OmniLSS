"""JAX reimplementation of selected `gamlss` R package functionality."""

# Enable JAX float64 globally before anything else touches JAX.
from . import _precision as _precision  # noqa: F401  (side-effect import)

from ._version import PACKAGE_VERSION

__version__ = PACKAGE_VERSION

from . import config as _config
from . import config  # public: omnilss.config.GPU_CROSSOVER_N etc.
from .config import auto_select_method, get_config_summary
from .centilesFan import centiles_fan
from .chooseDistParallel import chooseDist, chooseDistPred, getOrder
from .controls import GAMLSSControl, GLIMControl, gamlss_control, glim_control
from .DevianceIncr import devianceIncr
from .dtop import dtop, DTopResult
from .ECDF import ECDF, ECDFResult
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
    TF2,
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
from .distributions_b3 import GT, SHASH, SN1, SN2, SHASHo
from .distributions_b4 import BEINF, BEINF0, BEINF1, BEOI, BEZI
from .distributions_b5 import ZAGA, ZAIG, ZAP, ZIP2, ZINBI
from .distributions_b6 import DEL, DPO, PIG, SI, SICHEL, WARING, YULE
from .distributions_b7 import BB, BNB, MN3, MN4, MN5
from .distributions_b8 import GB2, GG, LNO, NET, PARETO
from .distributions_b9 import LG, ZALG, ZILG, ZIPF
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
from .distributions_b11 import GB1, GIG
from .distributions_b12 import GAF, NOF, RGE, WEI2, WEI3, BEo, GEOMo, PARETO2o
from .distributions_b13 import DELAPORT, IGA, LOGITNO, PE2, PIG2, GeneralisedPoisson
from .distributions_b14 import ST5, BCPEo, BCTo, JSUo, SHASHo2
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
from .dpqr_functions import fittedMN, meanBEINF, meanBEINF0, meanBEINF1, meanZAGA, meanZAIG
from .distributions_st_compat import ST1, ST2, ST3, ST4
from .distributions_dbi import DBI
from .distributions_dburr12 import DBURR12
from .distributions_boxcox_untr import BCCGuntr, BCPEuntr, BCTuntr
from .distributions3 import (
    pdf,
    log_pdf,
    cdf,
    quantile,
    random,
    is_continuous,
    is_discrete,
    support,
    moment_availability,
    mean,
    variance,
    skewness,
    kurtosis,
)
from .families import FamilyDefinition
from .edf import edf, smooth_term_edf
from .data_prep import (
    data_dim,
    data_distinct,
    data_exclude_class,
    data_fac2num,
    data_int2num,
    data_names,
    data_na_obs,
    data_na_vars,
    data_omit,
    data_only_continuous,
    data_rm,
    data_rm1val,
    data_rmNAvars,
    data_rename,
    data_select,
    data_shorter_names,
    data_cut,
    data_part,
    data_part_list,
    data_Kfold_index,
    data_Kfold_weights,
    data_boot_index,
    data_boot_weights,
    data_factor,
    data_cha2fac,
    data_few2fac,
    data_str,
    data_formulae,
    high_val,
    low_val,
    y_distinct,
    y_factor,
)
from .family_means import meanBEOI, meanBEZI
from .pcr import SVDResult, getSVD, which_Data_Corr
from .smoother_fits import SmootherFit, gamlss_cs, gamlss_cy, gamlss_pb, gamlss_ps
from .fractional_poly import FPFit, FPSpec, bfp, fp, gamlss_fp
from .pspline_variants import (
    PSplineVariant,
    gamlss_pbc,
    gamlss_pbm,
    gamlss_pbo,
    gamlss_pbp,
    gamlss_pbz,
    gamlss_pvc,
    pbc,
    pbm,
    pbo,
    pbp,
    pbz,
    pvc,
)
from .mixtures import MX_control, NP_control, dMX, dmx, mx_control, np_control, pMX, pmx
from .pcategorical import PCatFit, PCatSpec, gamlss_pcat, pcat, pairwise_differences
from .powerpoly import PPFit, PPSpec, gamlss_pp, pp, scs
from .ridge_random import (
    RIFit,
    RISpec,
    RandomFit,
    RandomSpec,
    gamlss_random,
    gamlss_ri,
    random_spec,
    ri,
)
from .smoother_specs import (
    CSSpline,
    CYSpline,
    PBSpline,
    PSSpline,
    bbase,
    cbase,
    cdiff,
    cs,
    cy,
    difference_matrix,
    pb,
    ps,
)
from .smoother_controls import (
    cs_control,
    cy_control,
    pbc_control,
    pbm_control,
    pbo_control,
    pbp_control,
    pb_control,
    pbz_control,
    pvc_control,
)
from .family_groups import (
    BINOMIAL,
    BI_LIST,
    COUNTS,
    MULTIN_LIST,
    REAL_0_TO_1,
    REAL_ALL,
    REAL_LINE,
    REAL_PLUS,
    SM_LIST,
    _hat_WX,
)
from .family_generator import (
    Family,
    Family_d,
    Family_p,
    Family_q,
    Family_r,
    GAMLSS,
    GAMLSSDistribution,
    GeneratedFamily,
    family_d,
    family_p,
    family_q,
    family_r,
    gen_Family,
)
from .MULTIN import MULTIN
from .gen_family_builder import build_gen_family_skeleton
from .family_capabilities import (
    CapabilityStatus,
    CAPABILITY_MATRIX_VERSION,
    FEATURES as FAMILY_CAPABILITY_FEATURES,
    ImplementationStatus,
    METHOD_CAPABILITY_FEATURES,
    METHOD_ROUTE_FEATURES,
    FamilyCapability,
    FamilyCapabilityError,
    capability_matrix,
    family_capability_names,
    family_supports,
    get_family_capability,
    implementation_status_for_family,
    method_route_feature,
    list_family_capabilities,
    method_capability_features,
    method_route_capability_report,
    require_family_capability,
    require_method_route,
    validate_capability_matrix_payload,
)
from .fitDist import fitDist
from .fitDistPred import fitDistPred, gamlssMLpred
from .Formulae2data import Formulae2data, Formulae2DataResult
from .FitTail import logSurv, loglogSurv, TailCurveResult
from .fitted_plot import fitted_plot
from .flexDist import FlexDistResult, flexDist
from .find_hyper import find_hyper, FindHyperResult
from .fitting import gamlss, gamlss_ml
from .gamlss_5 import gamlss_control_exact, glim_control_exact
from .gamlssML import gamlssML
from .gamlssVGD_23_12_21 import CV, VGD, gamlssCV, gamlssVGD, getTGD
from .gen_likelihood import gen_likelihood, GenLikelihoodResult
from .hatvalues import hatvalues
from .histSmo import histSmo, histSmoC, histSmoO, histSmoP, HistSmoResult
from .hazardFun import HazardResult, gen_hazard, hazardFun
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
from .gamlss_family import (
    LinkGamlss,
    as_family,
    as_gamlss_family,
    checklink,
    family_summary,
    gamlss_family,
    gamlss_family_default,
    make_link_gamlss,
    show_link,
)
from .centile_skew import centile_skew, centileSkew
from .o_variant_distributions import (
    dBCCGo,
    dBCPEo,
    dBCTo,
    dBEo,
    dGEOMo,
    dJSUo,
    dPARETO2o,
    dSHASHo,
    pBCCGo,
    pBCPEo,
    pBCTo,
    pBEo,
    pGEOMo,
    pJSUo,
    pPARETO2o,
    pSHASHo,
    qBCCGo,
    qBCPEo,
    qBCTo,
    qBEo,
    qGEOMo,
    qJSUo,
    qPARETO2o,
    qSHASHo,
    rBCCGo,
    rBCPEo,
    rBCTo,
    rBEo,
    rGEOMo,
    rJSUo,
    rPARETO2o,
    rSHASHo,
)
from .gamlss_cens import CensoredFamily, cens, cens_d, cens_p, cens_q, gen_cens
from .gamlss_scope import gamlss_scope
from .gamlss_tr import (
    TruncatedFamily,
    gen_trun,
    trun,
    trun_d,
    trun_p,
    trun_q,
    trun_r,
)
from .logLik import logLik
from .loglogSurv import logSurv0
from .loglogplot import loglogplot, LogLogPlotResult
from .plotCentileSK import PlotCentileSKResult, plotCentileSK
from .pdfplot import PDFPlotEntry, PDFPlotResult, pdf_plot, pdf_plot_data, plotBEINF, plotBEINF0, plotBEINF1
from .lpred import lpred as lpred_exact
from .momentSK import MomentSKResult, momentSK, momentSkew
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
    AIC,
    aic,
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
# R-name aliases for the information criteria (R exports AIC / GAIC / IC).
GAIC = gaic
IC = ic
from .predict_gamlss_23_12_21 import predict as predict_gamlss_exact
from .print import print as print_gamlss_exact
from .stepGAIC_03_10_13 import extractAIC, stepGAIC
from .stepwise_vr import StepwisePathResult, StepwisePathRow, stepGAIC_VR, step_gaic_vr
from .stepwise_all import add1All, drop1All
from .stepGAICAll_A_parallel import stepGAICAll_A
from .stepGAICAll_B_Parallel import addtermAllP, droptermAllP, stepGAICAll_B
from .stepTGD import add1TGD, add1TGDP, drop1TGD, drop1TGDP, extractTGD, stepTGD
from .update import update
from .worm_plot import MultiPanelWormPlotData, WormPlotData, wp, wp_data, wp_interactive

# ── New Phase-A diagnostic / helper imports ───────────────────────────────────
from .par_plot import par_plot, ParPlotResult, ParPlotSeries
from .prof_dev import prof_dev, prof_term, ProfDevResult
from .lms import lms, z_scores, LMSResult
from .getSmo import getSmo, edfAll, SmootherInfo
from .plot_smoothers import plotDF, plotLambda, SmootherPlotRow
from .getZmatrix import getZmatrix
from .quantSheets import quantSheets, findPower, z_scoresQS, QuantSheetsResult
from .vis_lo import vis_lo, VisLoResult
from .lo import lo, LoResult
from .quantSheets_control import quantSheets_control, QuantSheetsControl
from .centileSK import centileSK, CentileSKResult
from .centileKurt import centileKurt, CentileKurtResult
from .checkMomentSK import checkMomentSK, CheckMomentSKResult
from .checkCentileSK import checkCentileSK, CheckCentileSKResult
from .theoCentileSK import theoCentileSK, TheoCentileSKResult
from .vcov_gamlss import get_K, rvcov

# Optional modules are resolved lazily via __getattr__ below so missing optional
# dependencies do not make ``import omnilss`` fail.

__all__ = [
    "FamilyDefinition",
    "GAMLSSControl",
    "GAMLSSModel",
    "GLIMControl",
    "CapabilityStatus",
    "CAPABILITY_MATRIX_VERSION",
    "FAMILY_CAPABILITY_FEATURES",
    "ImplementationStatus",
    "METHOD_CAPABILITY_FEATURES",
    "METHOD_ROUTE_FEATURES",
    "FamilyCapability",
    "FamilyCapabilityError",
    "capability_matrix",
    "family_capability_names",
    "family_supports",
    "get_family_capability",
    "implementation_status_for_family",
    "list_family_capabilities",
    "method_capability_features",
    "method_route_capability_report",
    "method_route_feature",
    "require_family_capability",
    "require_method_route",
    "validate_capability_matrix_payload",
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
    "BCTuntr",
    "build_gen_family_skeleton",
    "BCPEuntr",
    "BCCGuntr",
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
    "DBI",
    "DBURR12",
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
    "MULTIN",
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
    "ST4",
    "ST3",
    "ST2",
    "ST1",
    "ST3C",
    "ST5",
    "SummaryResult",
    "CV",
    "TF",
    "TF2",
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
    "AIC",
    "GAIC",
    "IC",
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
    "support",
    "moment_availability",
    "kurtosis",
    "skewness",
    "variance",
    "mean",
    "is_discrete",
    "is_continuous",
    "random",
    "quantile",
    "cdf",
    "log_pdf",
    "pdf",
    "residuals",
    "rsq",
    "summary",
    "stepGAIC",
    "stepGAICAll_A",
    "stepGAICAll_B",
    "stepTGD",
    # ── gamlss drop1All / add1All single-term tables ──
    "drop1All",
    "add1All",
    # ── gamlss stepGAIC.VR path selection ──
    "stepGAIC_VR",
    "step_gaic_vr",
    "StepwisePathResult",
    "StepwisePathRow",
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
    "PredictionSchemaError",
    "build_prediction_design_matrix",
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
    # ── gamlss family / link framework (R gamlss.dist) ──
    "LinkGamlss",
    "make_link_gamlss",
    "checklink",
    "show_link",
    "as_family",
    "as_gamlss_family",
    "gamlss_family",
    "gamlss_family_default",
    "family_summary",
    "centile_skew",
    "centileSkew",
    "gamlss_scope",
    # ── gen.Family / Family(.d/.p/.q/.r) / GAMLSS (genFamily.R, distributions3.R) ──
    "Family",
    "Family_d",
    "Family_p",
    "Family_q",
    "Family_r",
    "GeneratedFamily",
    "family_d",
    "family_p",
    "family_q",
    "family_r",
    "gen_Family",
    "GAMLSS",
    "GAMLSSDistribution",
    # ── gamlss edf() + internal support groups (edf.R, chooseDistParallel.R) ──
    "edf",
    "smooth_term_edf",
    "REAL_LINE",
    "REAL_PLUS",
    "REAL_0_TO_1",
    "REAL_ALL",
    "COUNTS",
    "BINOMIAL",
    "BI_LIST",
    "MULTIN_LIST",
    "SM_LIST",
    # ── gamlss.prepdata data shaping ──
    "data_dim",
    "data_names",
    "data_na_vars",
    "data_na_obs",
    "data_omit",
    "data_distinct",
    "data_rm1val",
    "data_rm",
    "data_exclude_class",
    "data_only_continuous",
    "data_fac2num",
    "data_int2num",
    "data_rmNAvars",
    "data_rename",
    "data_select",
    "data_shorter_names",
    "data_cut",
    "data_part",
    "data_part_list",
    "data_Kfold_index",
    "data_Kfold_weights",
    "data_boot_index",
    "data_boot_weights",
    # ── gamlss.dist inflated-beta means ──
    "meanBEZI",
    "meanBEOI",
    "y_distinct",
    "y_factor",
    "data_cha2fac",
    "data_few2fac",
    "data_str",
    "data_formulae",
    "data_factor",
    "high_val",
    "low_val",
    # ── gamlss smoother control objects ──
    "pb_control",
    "cs_control",
    "cy_control",
    "pbc_control",
    "pbm_control",
    "pbo_control",
    "pbp_control",
    "pbz_control",
    "pvc_control",
    # ── gamlss smoother specification (pb basis + penalty) ──
    "pb",
    "PBSpline",
    "bbase",
    "difference_matrix",
    "ps",
    "PSSpline",
    "cs",
    "CSSpline",
    # ── gamlss smoother backfitting steps ──
    "gamlss_pb",
    "gamlss_ps",
    "gamlss_cs",
    "gamlss_cy",
    "SmootherFit",
    # ── gamlss cyclic P-spline spec ──
    "cy",
    "CYSpline",
    "cbase",
    "cdiff",
    # ── gamlss fractional-polynomial smoother ──
    "fp",
    "bfp",
    "gamlss_fp",
    "FPSpec",
    "FPFit",
    # ── gamlss P-spline variant specs and fitters ──
    "pbm",
    "pbo",
    "pbp",
    "pbz",
    "pbc",
    "pvc",
    "PSplineVariant",
    "gamlss_pbm",
    "gamlss_pbo",
    "gamlss_pbp",
    "gamlss_pbz",
    "gamlss_pbc",
    "gamlss_pvc",
    # ── gamlss ridge-type smoothers ──
    "ri",
    "RISpec",
    "RIFit",
    "gamlss_ri",
    "random_spec",
    "RandomSpec",
    "RandomFit",
    "gamlss_random",
    # ── gamlss power-polynomial smoother ──
    "pp",
    "PPSpec",
    "PPFit",
    "gamlss_pp",
    "scs",
    # ── gamlss penalised categorical smoother ──
    "pcat",
    "PCatSpec",
    "PCatFit",
    "gamlss_pcat",
    "pairwise_differences",
    # ── gamlss.mx mixture densities and control lists ──
    "dmx",
    "pmx",
    "dMX",
    "pMX",
    "mx_control",
    "np_control",
    "MX_control",
    "NP_control",
    # ── gamlss.foreach PCR helpers ──
    "getSVD",
    "SVDResult",
    "which_Data_Corr",
    # ── gamlss.cens censored-family helpers ──
    "CensoredFamily",
    "cens",
    "cens_d",
    "cens_p",
    "cens_q",
    "gen_cens",
    # ── gamlss.tr truncation helpers ──
    "TruncatedFamily",
    "trun",
    "gen_trun",
    "trun_d",
    "trun_p",
    "trun_q",
    "trun_r",
    # ── R 'o' (original) parameterisation densities ──
    "dBCCGo",
    "dBCPEo",
    "dBCTo",
    "dBEo",
    "dGEOMo",
    "dJSUo",
    "dPARETO2o",
    "dSHASHo",
    "pBCCGo",
    "pBCPEo",
    "pBCTo",
    "pBEo",
    "pGEOMo",
    "pJSUo",
    "pPARETO2o",
    "pSHASHo",
    "qBCCGo",
    "qBCPEo",
    "qBCTo",
    "qBEo",
    "qGEOMo",
    "qJSUo",
    "qPARETO2o",
    "qSHASHo",
    "rBCCGo",
    "rBCPEo",
    "rBCTo",
    "rBEo",
    "rGEOMo",
    "rJSUo",
    "rPARETO2o",
    "rSHASHo",
    "save_model",
    "load_model",
    "save_model_json",
    "load_model_json",
    "validate_model_json",
    "artifact_schema_policy",
    "compare_model_capability_snapshot",
    "save_model_pickle",
    "load_model_pickle",    "Formulae2DataResult",
    "Formulae2data",
    "GenLikelihoodResult",
    "gen_likelihood",
    "DTopResult",
    "dtop",
    "ECDF",
    "ECDFResult",
    "loglogplot",
    "LogLogPlotResult",
    "par_plot",
    "ParPlotResult",
    "ParPlotSeries",
    "find_hyper",
    "FindHyperResult",
    "get_K",
    "rvcov",
    "prof_dev",
    "prof_term",
    "ProfDevResult",
    "lms",
    "z_scores",
    "LMSResult",
    "getSmo",
    "edfAll",
    "plotDF",
    "plotLambda",
    "SmootherInfo",
    "getZmatrix",
    "logSurv",
    "loglogSurv",
    "TailCurveResult",
    "histSmo",
    "histSmoO",
    "histSmoC",
    "histSmoP",
    "HistSmoResult",
    "quantSheets",
    "findPower",
    "z_scoresQS",
    "QuantSheetsResult",
    "vis_lo",
    "lo",
    "VisLoResult",
    "LoResult",
    "quantSheets_control",
    "QuantSheetsControl",
    "logSurv0",
    "centileSK",
    "CentileSKResult",
    "hazardFun",
    "gen_hazard",
    "HazardResult",
    "momentSK",
    "momentSkew",
    "MomentSKResult",
    "flexDist",
    "FlexDistResult",
    "checkMomentSK",
    "CheckMomentSKResult",
    "checkCentileSK",
    "CheckCentileSKResult",
    "theoCentileSK",
    "TheoCentileSKResult",
    "centileKurt",
    "CentileKurtResult",
    "plotCentileSK",
    "PlotCentileSKResult",

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
        PredictionSchemaError,
        build_prediction_design_matrix,
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
    PredictionSchemaError = None
    build_prediction_design_matrix = None
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
        validate_model_json,
        artifact_schema_policy,
        compare_model_capability_snapshot,
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
    validate_model_json = None
    artifact_schema_policy = None
    compare_model_capability_snapshot = None
    load_model_pickle = None
    save_model_pickle = None


_OPTIONAL_EXPORTS = {
    "crps": ("omnilss.scoring", "crps", "scoring"),
    "log_score": ("omnilss.scoring", "log_score", "scoring"),
    "dss": ("omnilss.scoring", "dss", "scoring"),
    "interval_score": ("omnilss.scoring", "interval_score", "scoring"),
    "coverage": ("omnilss.scoring", "coverage", "scoring"),
    "pit_histogram": ("omnilss.scoring", "pit_histogram", "scoring"),
    "scoring_summary": ("omnilss.scoring", "scoring_summary", "scoring"),
    "GAMLSSRegressor": ("omnilss.sklearn_compat", "GAMLSSRegressor", "sklearn"),
}


def __getattr__(name: str):
    """Lazily load optional top-level exports.

    This keeps the core package importable when optional extras such as
    scikit-learn or future scoring-rule backends are not installed.
    """
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib
    import warnings

    module_name, attr_name, extra = _OPTIONAL_EXPORTS[name]
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
    except ImportError as exc:
        warnings.warn(
            f"Optional OmniLSS export {name!r} requires the {extra!r} extra: "
            f"pip install 'omnilss[{extra}]'. Error: {exc}",
            ImportWarning,
            stacklevel=2,
        )
        value = None
    globals()[name] = value
    return value


def check_installation() -> dict:
    """Return module availability diagnostics for OmniLSS installation."""
    import importlib

    status: dict[str, str] = {}
    modules = {
        "core": ("omnilss.fitting", "gamlss"),
        "distributions": ("omnilss.distributions", "NO"),
        "smoothers": ("omnilss.smoothers.pb", "fit_pspline"),
        "diagnostics": ("omnilss.diagnostics", "quantile_residuals"),
        "prediction": ("omnilss.prediction", "predict_params"),
        "serialization": ("omnilss.serialization", "save_model_json"),
        "deep": ("omnilss.deep.deep_gamlss", "fit_deep_gamlss"),
        "sklearn": ("omnilss.sklearn_compat", "GAMLSSRegressor"),
        "grpc": ("omnilss.api.grpc.generated.fit_pb2_grpc", "FitServiceStub"),
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
