"""Performance testing configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Base directories
PERFORMANCE_DIR = Path(__file__).parent
RESULTS_DIR = PERFORMANCE_DIR / "results"
DATA_DIR = PERFORMANCE_DIR / "data"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, DATA_DIR, REPORTS_DIR,
                 RESULTS_DIR / "raw", RESULTS_DIR / "processed",
                 DATA_DIR / "synthetic", DATA_DIR / "real"]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataSize:
    """Data size configuration."""
    name: str
    n_obs: int
    description: str


# Data sizes to test
DATA_SIZES = [
    DataSize("tiny", 100, "Tiny dataset for quick tests"),
    DataSize("small", 500, "Small dataset"),
    DataSize("medium", 5_000, "Medium dataset"),
    DataSize("large", 50_000, "Large dataset"),
    DataSize("xlarge", 100_000, "Extra large dataset"),
]

QUICK_DATA_SIZES = [
    DataSize("tiny", 100, "Tiny dataset"),
    DataSize("small", 500, "Small dataset"),
    DataSize("medium", 5_000, "Medium dataset"),
]


@dataclass
class DistributionConfig:
    """Distribution family configuration."""
    name: str
    r_name: str
    python_class: str
    type: str  # "continuous", "discrete", "mixed"
    parameters: list[str]
    requires_fixed: list[str] = field(default_factory=list)
    data_generator: str | None = None
    notes: str = ""


# All distributions to test
DISTRIBUTIONS = [
    # Continuous - Basic
    DistributionConfig("NO", "NO", "NO", "continuous", ["mu", "sigma"]),
    DistributionConfig("GA", "GA", "GA", "continuous", ["mu", "sigma"]),
    DistributionConfig("LOGNO", "LOGNO", "LOGNO", "continuous", ["mu", "sigma"]),
    DistributionConfig("WEI", "WEI", "WEI", "continuous", ["mu", "sigma"]),
    DistributionConfig("EXP", "EXP", "EXP", "continuous", ["mu"]),
    DistributionConfig("IG", "IG", "IG", "continuous", ["mu", "sigma"]),
    DistributionConfig("LO", "LO", "LO", "continuous", ["mu", "sigma"]),
    DistributionConfig("TF", "TF", "TF", "continuous", ["mu", "sigma", "nu"]),
    
    # Continuous - Extended
    DistributionConfig("NO2", "NO2", "NO2", "continuous", ["mu", "sigma"]),
    DistributionConfig("LOGNO2", "LOGNO2", "LOGNO2", "continuous", ["mu", "sigma"]),
    DistributionConfig("PE", "PE", "PE", "continuous", ["mu", "sigma", "nu"]),
    DistributionConfig("SIMPLEX", "SIMPLEX", "SIMPLEX", "continuous", ["mu", "sigma"]),
    DistributionConfig("exGAUS", "exGAUS", "exGAUS", "continuous", ["mu", "sigma", "nu"]),
    
    # Continuous - Skewed
    DistributionConfig("SHASH", "SHASH", "SHASH", "continuous", ["mu", "sigma", "nu", "tau"]),
    DistributionConfig("SN1", "SN1", "SN1", "continuous", ["mu", "sigma", "nu"]),
    DistributionConfig("SN2", "SN2", "SN2", "continuous", ["mu", "sigma", "nu"]),
    DistributionConfig("GT", "GT", "GT", "continuous", ["mu", "sigma", "nu", "tau"]),
    
    # Continuous - Advanced
    DistributionConfig("GG", "GG", "GG", "continuous", ["mu", "sigma", "nu"]),
    DistributionConfig("GB2", "GB2", "GB2", "continuous", ["mu", "sigma", "nu", "tau"]),
    DistributionConfig("NET", "NET", "NET", "continuous", ["mu", "sigma", "nu", "tau"]),
    
    # Discrete - Basic
    DistributionConfig("PO", "PO", "PO", "discrete", ["mu"]),
    DistributionConfig("BI", "BI", "BI", "discrete", ["mu"], requires_fixed=["bd"]),
    DistributionConfig("GEOM", "GEOM", "GEOM", "discrete", ["mu"]),
    DistributionConfig("NBI", "NBI", "NBI", "discrete", ["mu", "sigma"]),
    DistributionConfig("NBII", "NBII", "NBII", "discrete", ["mu", "sigma"]),
    
    # Discrete - Zero-Inflated
    DistributionConfig("ZIP", "ZIP", "ZIP", "discrete", ["mu", "sigma"]),
    DistributionConfig("ZIP2", "ZIP2", "ZIP2", "discrete", ["mu", "sigma"]),
    DistributionConfig("ZINBI", "ZINBI", "ZINBI", "discrete", ["mu", "sigma", "nu"]),
    DistributionConfig("ZAP", "ZAP", "ZAP", "discrete", ["mu", "sigma"]),
    
    # Discrete - Advanced
    DistributionConfig("BB", "BB", "BB", "discrete", ["mu", "sigma"], requires_fixed=["bd"]),
    DistributionConfig("BNB", "BNB", "BNB", "discrete", ["mu", "sigma"], requires_fixed=["bd"]),
    DistributionConfig("PIG", "PIG", "PIG", "discrete", ["mu", "sigma"]),
    DistributionConfig("SICHEL", "SICHEL", "SICHEL", "discrete", ["mu", "sigma", "nu"]),
    DistributionConfig("DPO", "DPO", "DPO", "discrete", ["mu", "sigma"]),
    DistributionConfig("DEL", "DEL", "DEL", "discrete", ["mu", "sigma", "nu"]),
    DistributionConfig("YULE", "YULE", "YULE", "discrete", ["mu"]),
    DistributionConfig("WARING", "WARING", "WARING", "discrete", ["mu", "sigma"]),
    
    # Beta and Zero-Altered
    DistributionConfig("BE", "BE", "BE", "continuous", ["mu", "sigma"]),
    DistributionConfig("BEINF", "BEINF", "BEINF", "mixed", ["mu", "sigma", "nu", "tau"]),
    DistributionConfig("ZAGA", "ZAGA", "ZAGA", "mixed", ["mu", "sigma", "nu"]),
    DistributionConfig("ZAIG", "ZAIG", "ZAIG", "mixed", ["mu", "sigma", "nu"]),
    
    # Batch 1 remaining
    DistributionConfig("GU", "GU", "GU", "continuous", ["mu", "sigma"]),
    DistributionConfig("RG", "RG", "RG", "continuous", ["mu", "sigma"]),
    DistributionConfig("IGAMMA", "IGAMMA", "IGAMMA", "continuous", ["mu", "sigma"]),
    DistributionConfig("PARETO2", "PARETO2", "PARETO2", "continuous", ["mu", "sigma"]),
]

# Quick test subset (most common distributions)
QUICK_DISTRIBUTIONS = [
    "NO", "GA", "LOGNO", "PO", "BI", "NBI", "BE", "ZIP", "ZAGA"
]


@dataclass
class ModelConfig:
    """Model complexity configuration."""
    name: str
    mu_formula: str
    sigma_formula: str = "~1"
    nu_formula: str | None = None
    tau_formula: str | None = None
    description: str = ""
    n_predictors: int = 0


# Model complexities to test
MODEL_CONFIGS = [
    ModelConfig("intercept", "y ~ 1", description="Intercept only", n_predictors=0),
    ModelConfig("linear1", "y ~ x1", description="One predictor", n_predictors=1),
    ModelConfig("linear2", "y ~ x1 + x2", description="Two predictors", n_predictors=2),
    ModelConfig("linear5", "y ~ x1 + x2 + x3 + x4 + x5", description="Five predictors", n_predictors=5),
    ModelConfig("smooth1", "y ~ pb(x1)", description="One smooth term", n_predictors=1),
    ModelConfig("smooth2", "y ~ pb(x1) + ps(x2)", description="Two smooth terms", n_predictors=2),
    ModelConfig("mixed", "y ~ x1 + pb(x2)", description="Mixed linear and smooth", n_predictors=2),
    ModelConfig("sigma_model", "y ~ x1", "~x2", description="Sigma modeling", n_predictors=2),
]

QUICK_MODEL_CONFIGS = [
    ModelConfig("intercept", "y ~ 1", description="Intercept only"),
    ModelConfig("linear1", "y ~ x1", description="One predictor", n_predictors=1),
    ModelConfig("linear2", "y ~ x1 + x2", description="Two predictors", n_predictors=2),
]


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    n_repeats: int = 5  # Number of times to repeat each test
    warmup_runs: int = 2  # Warmup runs before timing
    timeout: float = 300.0  # Timeout in seconds
    memory_profiling: bool = True
    save_raw_results: bool = True
    random_seed: int = 42


# Default configurations
DEFAULT_CONFIG = BenchmarkConfig()
QUICK_CONFIG = BenchmarkConfig(n_repeats=3, warmup_runs=1, timeout=60.0)


@dataclass
class ReportConfig:
    """Report generation configuration."""
    title: str = "Omni GAMLSS Performance Report"
    author: str = "Omni GAMLSS Team"
    include_plots: bool = True
    include_tables: bool = True
    include_raw_data: bool = False
    plot_format: str = "png"  # "png", "svg", "pdf"
    plot_dpi: int = 300
    table_format: str = "github"  # "github", "html", "latex"


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "speedup_good": 1.5,  # JAX is 1.5x faster than R
    "speedup_acceptable": 0.8,  # JAX is at least 80% of R speed
    "speedup_poor": 0.5,  # JAX is less than 50% of R speed
    "memory_ratio_good": 1.2,  # JAX uses at most 1.2x R memory
    "memory_ratio_acceptable": 2.0,  # JAX uses at most 2x R memory
    "convergence_rate_good": 0.95,  # 95% convergence rate
    "convergence_rate_acceptable": 0.85,  # 85% convergence rate
}


def get_distribution_config(name: str) -> DistributionConfig | None:
    """Get distribution configuration by name."""
    for dist in DISTRIBUTIONS:
        if dist.name == name or dist.r_name == name:
            return dist
    return None


def get_quick_distributions() -> list[DistributionConfig]:
    """Get quick test distribution configurations."""
    return [d for d in DISTRIBUTIONS if d.name in QUICK_DISTRIBUTIONS]


def get_all_distributions() -> list[DistributionConfig]:
    """Get all distribution configurations."""
    return DISTRIBUTIONS


def get_model_config(name: str) -> ModelConfig | None:
    """Get model configuration by name."""
    for model in MODEL_CONFIGS:
        if model.name == name:
            return model
    return None
