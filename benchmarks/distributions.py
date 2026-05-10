"""Distribution family performance benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "omnilss" / "src"))

from .base import Benchmark, BenchmarkResult
from .data_generators import generate_data_for_distribution


class DistributionBenchmark(Benchmark):
    """Benchmark for a specific distribution family."""
    
    def __init__(
        self,
        distribution: str,
        model_formula: str = "y ~ x1",
        sigma_formula: str = "~1",
        n_obs: int = 1000,
        n_predictors: int = 1,
        seed: int = 42,
    ):
        name = f"{distribution}_{n_obs}obs"
        description = f"Benchmark {distribution} distribution with {n_obs} observations"
        super().__init__(name, description)
        
        self.distribution = distribution
        self.model_formula = model_formula
        self.sigma_formula = sigma_formula
        self.n_obs = n_obs
        self.n_predictors = n_predictors
        self.seed = seed
    
    def setup(self, **kwargs: Any) -> dict[str, Any]:
        """Generate test data."""
        data = generate_data_for_distribution(
            self.distribution,
            self.n_obs,
            self.n_predictors,
            self.seed,
        )
        
        return {
            "data": data,
            "distribution": self.distribution,
            "model_config": self.model_formula,
            "n_obs": self.n_obs,
        }
    
    def run_python(self, setup_data: dict[str, Any]) -> BenchmarkResult:
        """Run Python/JAX implementation."""
        from omnilss.distributions import resolve_family
        from omnilss.fitting import gamlss
        
        data = setup_data["data"]
        
        # Get family
        family = resolve_family(self.distribution)
        
        # Run with timing
        def fit_model():
            return gamlss(
                formula=self.model_formula,
                sigma_formula=self.sigma_formula,
                family=family,
                data=data,
            )
        
        model, fit_time, memory_increase = self.run_with_timing(fit_model)
        
        # Extract results
        deviance = float(model.deviance) if hasattr(model, "deviance") else None
        aic = float(model.aic) if hasattr(model, "aic") else None
        bic = float(model.bic) if hasattr(model, "bic") else None
        n_iterations = int(model.n_iterations) if hasattr(model, "n_iterations") else None
        converged = bool(model.converged) if hasattr(model, "converged") else None
        
        return BenchmarkResult(
            benchmark_name=self.name,
            distribution=self.distribution,
            model_config=self.model_formula,
            data_size=self.n_obs,
            implementation="python",
            total_time=fit_time,
            fit_time=fit_time,
            memory_increase_mb=memory_increase,
            deviance=deviance,
            aic=aic,
            bic=bic,
            n_iterations=n_iterations,
            converged=converged,
            coefficients=dict(model.coefficients) if hasattr(model, "coefficients") else None,
            fitted_values=dict(model.fitted_values) if hasattr(model, "fitted_values") else None,
        )
    
    def run_r(self, setup_data: dict[str, Any]) -> BenchmarkResult:
        """Run R implementation."""
        # Import R bridge
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "omnilss" / "tests"))
        
        try:
            from rbus.r_bridge import RBridge
        except ImportError:
            return BenchmarkResult(
                benchmark_name=self.name,
                distribution=self.distribution,
                model_config=self.model_formula,
                data_size=self.n_obs,
                implementation="r",
                total_time=0.0,
                success=False,
                error_message="R bridge not available",
            )
        
        data = setup_data["data"]
        
        # Run with timing
        def fit_model():
            bridge = RBridge()
            # Use call_r_gamlss instead of fit_gamlss
            return bridge.call_r_gamlss(
                data=data,
                formula=self.model_formula,
                family=self.distribution,
                sigma_formula=self.sigma_formula,
            )
        
        try:
            result, fit_time, memory_increase = self.run_with_timing(fit_model)
            
            return BenchmarkResult(
                benchmark_name=self.name,
                distribution=self.distribution,
                model_config=self.model_formula,
                data_size=self.n_obs,
                implementation="r",
                total_time=fit_time,
                fit_time=fit_time,
                memory_increase_mb=memory_increase,
                deviance=result.get("deviance"),
                aic=result.get("aic"),
                bic=result.get("bic"),
                n_iterations=result.get("n_iterations"),
                converged=result.get("converged", True),
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark_name=self.name,
                distribution=self.distribution,
                model_config=self.model_formula,
                data_size=self.n_obs,
                implementation="r",
                total_time=0.0,
                success=False,
                error_message=str(e),
            )


def create_distribution_benchmarks(
    distributions: list[str],
    data_sizes: list[int],
    model_configs: list[tuple[str, str]] | None = None,
) -> list[DistributionBenchmark]:
    """Create benchmarks for multiple distributions and data sizes.
    
    Parameters
    ----------
    distributions : list[str]
        List of distribution names
    data_sizes : list[int]
        List of data sizes to test
    model_configs : list[tuple[str, str]], optional
        List of (mu_formula, sigma_formula) tuples
        Default is [("y ~ x1", "~1")]
    
    Returns
    -------
    benchmarks : list[DistributionBenchmark]
        List of benchmark objects
    """
    if model_configs is None:
        model_configs = [("y ~ x1", "~1")]
    
    benchmarks = []
    
    for dist in distributions:
        for n_obs in data_sizes:
            for mu_formula, sigma_formula in model_configs:
                # Determine number of predictors from formula
                n_predictors = mu_formula.count("x")
                
                benchmark = DistributionBenchmark(
                    distribution=dist,
                    model_formula=mu_formula,
                    sigma_formula=sigma_formula,
                    n_obs=n_obs,
                    n_predictors=max(1, n_predictors),
                )
                benchmarks.append(benchmark)
    
    return benchmarks
