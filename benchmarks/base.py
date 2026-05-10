"""Base classes for benchmarking."""

from __future__ import annotations

import gc
import os
import platform
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import psutil

# Detect JAX device
try:
    import jax
    JAX_AVAILABLE = True
    JAX_DEVICES = jax.devices()
    JAX_DEFAULT_DEVICE = JAX_DEVICES[0] if JAX_DEVICES else None
    JAX_HAS_GPU = any(d.platform in ('gpu', 'cuda', 'rocm') for d in JAX_DEVICES)
except ImportError:
    JAX_AVAILABLE = False
    JAX_DEVICES = []
    JAX_DEFAULT_DEVICE = None
    JAX_HAS_GPU = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    # Identification
    benchmark_name: str
    distribution: str
    model_config: str
    data_size: int
    implementation: str  # "python" or "r"
    
    # Timing
    total_time: float  # seconds
    fit_time: float | None = None
    setup_time: float | None = None
    
    # Memory
    peak_memory_mb: float | None = None
    memory_increase_mb: float | None = None
    
    # Model quality
    deviance: float | None = None
    aic: float | None = None
    bic: float | None = None
    n_iterations: int | None = None
    converged: bool | None = None
    
    # Numerical results
    coefficients: dict[str, np.ndarray] | None = None
    fitted_values: dict[str, np.ndarray] | None = None
    
    # Error information
    success: bool = True
    error_message: str | None = None
    error_traceback: str | None = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # System information
    platform: str = field(default_factory=lambda: platform.system())
    device: str = field(default_factory=lambda: str(JAX_DEFAULT_DEVICE) if JAX_AVAILABLE else "cpu")
    has_gpu: bool = field(default_factory=lambda: JAX_HAS_GPU)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "benchmark_name": self.benchmark_name,
            "distribution": self.distribution,
            "model_config": self.model_config,
            "data_size": self.data_size,
            "implementation": self.implementation,
            "total_time": self.total_time,
            "fit_time": self.fit_time,
            "setup_time": self.setup_time,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_increase_mb": self.memory_increase_mb,
            "deviance": self.deviance,
            "aic": self.aic,
            "bic": self.bic,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "platform": self.platform,
            "device": self.device,
            "has_gpu": self.has_gpu,
        }
        # Don't include large arrays in dict
        return result


@dataclass
class ComparisonResult:
    """Comparison between Python and R implementations."""
    
    benchmark_name: str
    distribution: str
    model_config: str
    data_size: int
    
    # Performance comparison
    python_time: float
    r_time: float
    speedup: float  # r_time / python_time
    
    # Memory comparison
    python_memory_mb: float | None = None
    r_memory_mb: float | None = None
    memory_ratio: float | None = None  # python_memory / r_memory
    
    # Numerical comparison
    deviance_diff: float | None = None
    deviance_rel_diff: float | None = None
    coefficient_max_diff: float | None = None
    coefficient_rmse: float | None = None
    
    # Convergence comparison
    python_converged: bool | None = None
    r_converged: bool | None = None
    both_converged: bool | None = None
    
    # Success
    python_success: bool = True
    r_success: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "distribution": self.distribution,
            "model_config": self.model_config,
            "data_size": self.data_size,
            "python_time": self.python_time,
            "r_time": self.r_time,
            "speedup": self.speedup,
            "python_memory_mb": self.python_memory_mb,
            "r_memory_mb": self.r_memory_mb,
            "memory_ratio": self.memory_ratio,
            "deviance_diff": self.deviance_diff,
            "deviance_rel_diff": self.deviance_rel_diff,
            "coefficient_max_diff": self.coefficient_max_diff,
            "coefficient_rmse": self.coefficient_rmse,
            "python_converged": self.python_converged,
            "r_converged": self.r_converged,
            "both_converged": self.both_converged,
            "python_success": self.python_success,
            "r_success": self.r_success,
        }


class Benchmark(ABC):
    """Base class for benchmarks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.process = psutil.Process()
    
    @abstractmethod
    def setup(self, **kwargs: Any) -> dict[str, Any]:
        """Setup benchmark (generate data, etc.)."""
        pass
    
    @abstractmethod
    def run_python(self, setup_data: dict[str, Any]) -> BenchmarkResult:
        """Run Python/JAX implementation."""
        pass
    
    @abstractmethod
    def run_r(self, setup_data: dict[str, Any]) -> BenchmarkResult:
        """Run R implementation."""
        pass
    
    def measure_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def run_with_timing(
        self,
        func: Callable[[], Any],
        measure_memory: bool = True,
    ) -> tuple[Any, float, float | None]:
        """Run function with timing and optional memory measurement.
        
        Returns:
            (result, time_seconds, peak_memory_mb)
        """
        # Force garbage collection
        gc.collect()
        
        # Measure initial memory
        if measure_memory:
            initial_memory = self.measure_memory()
            peak_memory = initial_memory
        else:
            initial_memory = None
            peak_memory = None
        
        # Run function
        start_time = time.perf_counter()
        try:
            result = func()
            success = True
        except Exception as e:
            result = None
            success = False
            raise
        finally:
            end_time = time.perf_counter()
            
            # Measure final memory
            if measure_memory:
                final_memory = self.measure_memory()
                peak_memory = max(peak_memory, final_memory)
        
        elapsed_time = end_time - start_time
        memory_increase = peak_memory - initial_memory if measure_memory else None
        
        return result, elapsed_time, memory_increase
    
    def run_single(
        self,
        implementation: str,
        setup_data: dict[str, Any],
        timeout: float | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        if implementation == "python":
            runner = self.run_python
        elif implementation == "r":
            runner = self.run_r
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        try:
            result = runner(setup_data)
            return result
        except Exception as e:
            # Create error result
            return BenchmarkResult(
                benchmark_name=self.name,
                distribution=setup_data.get("distribution", "unknown"),
                model_config=setup_data.get("model_config", "unknown"),
                data_size=setup_data.get("n_obs", 0),
                implementation=implementation,
                total_time=0.0,
                success=False,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
    
    def compare(
        self,
        setup_data: dict[str, Any],
        n_repeats: int = 5,
        warmup_runs: int = 2,
    ) -> ComparisonResult:
        """Compare Python and R implementations."""
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                self.run_single("python", setup_data)
            except:
                pass
            try:
                self.run_single("r", setup_data)
            except:
                pass
        
        # Actual benchmark runs
        python_results = []
        r_results = []
        
        for _ in range(n_repeats):
            py_result = self.run_single("python", setup_data)
            python_results.append(py_result)
            
            r_result = self.run_single("r", setup_data)
            r_results.append(r_result)
        
        # Aggregate results
        py_times = [r.total_time for r in python_results if r.success]
        r_times = [r.total_time for r in r_results if r.success]
        
        if not py_times or not r_times:
            # At least one implementation failed
            return ComparisonResult(
                benchmark_name=self.name,
                distribution=setup_data.get("distribution", "unknown"),
                model_config=setup_data.get("model_config", "unknown"),
                data_size=setup_data.get("n_obs", 0),
                python_time=np.mean(py_times) if py_times else float('inf'),
                r_time=np.mean(r_times) if r_times else float('inf'),
                speedup=0.0,
                python_success=bool(py_times),
                r_success=bool(r_times),
            )
        
        # Calculate statistics
        python_time = np.median(py_times)
        r_time = np.median(r_times)
        speedup = r_time / python_time if python_time > 0 else 0.0
        
        # Memory comparison
        py_memories = [r.memory_increase_mb for r in python_results if r.success and r.memory_increase_mb is not None]
        r_memories = [r.memory_increase_mb for r in r_results if r.success and r.memory_increase_mb is not None]
        
        python_memory = np.median(py_memories) if py_memories else None
        r_memory = np.median(r_memories) if r_memories else None
        memory_ratio = python_memory / r_memory if (python_memory and r_memory and r_memory > 0) else None
        
        # Numerical comparison (use last successful run)
        py_last = next((r for r in reversed(python_results) if r.success), None)
        r_last = next((r for r in reversed(r_results) if r.success), None)
        
        deviance_diff = None
        deviance_rel_diff = None
        if py_last and r_last and py_last.deviance is not None and r_last.deviance is not None:
            deviance_diff = abs(py_last.deviance - r_last.deviance)
            deviance_rel_diff = deviance_diff / abs(r_last.deviance) if r_last.deviance != 0 else None
        
        return ComparisonResult(
            benchmark_name=self.name,
            distribution=setup_data.get("distribution", "unknown"),
            model_config=setup_data.get("model_config", "unknown"),
            data_size=setup_data.get("n_obs", 0),
            python_time=python_time,
            r_time=r_time,
            speedup=speedup,
            python_memory_mb=python_memory,
            r_memory_mb=r_memory,
            memory_ratio=memory_ratio,
            deviance_diff=deviance_diff,
            deviance_rel_diff=deviance_rel_diff,
            python_converged=py_last.converged if py_last else None,
            r_converged=r_last.converged if r_last else None,
            both_converged=(py_last.converged and r_last.converged) if (py_last and r_last) else None,
            python_success=bool(py_times),
            r_success=bool(r_times),
        )


class BenchmarkSuite:
    """Collection of benchmarks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.benchmarks: list[Benchmark] = []
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
    
    def run_all(
        self,
        n_repeats: int = 5,
        warmup_runs: int = 2,
        verbose: bool = True,
    ) -> list[ComparisonResult]:
        """Run all benchmarks in the suite."""
        results = []
        
        for i, benchmark in enumerate(self.benchmarks, 1):
            if verbose:
                print(f"[{i}/{len(self.benchmarks)}] Running {benchmark.name}...")
            
            try:
                setup_data = benchmark.setup()
                comparison = benchmark.compare(setup_data, n_repeats, warmup_runs)
                results.append(comparison)
                
                if verbose:
                    print(f"  Speedup: {comparison.speedup:.2f}x")
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
        
        return results
