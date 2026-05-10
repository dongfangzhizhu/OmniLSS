"""Performance monitoring utilities for GAMLSS fitting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceMonitor:
    """Monitor performance during GAMLSS fitting.
    
    Attributes
    ----------
    use_jit : bool
        Whether JIT compilation is used
    jit_compile_time : float
        Time spent on JIT compilation (seconds)
    iteration_times : List[float]
        Time for each iteration (seconds)
    parameter_times : Dict[str, List[float]]
        Time for each parameter update per iteration
    total_time : float
        Total fitting time (seconds)
    n_observations : int
        Number of observations
    family_name : str
        Distribution family name
    """
    
    use_jit: bool = False
    jit_compile_time: float = 0.0
    iteration_times: List[float] = field(default_factory=list)
    parameter_times: Dict[str, List[float]] = field(default_factory=dict)
    total_time: float = 0.0
    n_observations: int = 0
    family_name: str = ""
    
    _start_time: Optional[float] = field(default=None, init=False, repr=False)
    _iter_start_time: Optional[float] = field(default=None, init=False, repr=False)
    
    def start(self):
        """Start timing total fitting."""
        self._start_time = time.time()
    
    def start_iteration(self):
        """Start timing an iteration."""
        self._iter_start_time = time.time()
    
    def record_jit_compile(self, compile_time: float):
        """Record JIT compilation time.
        
        Parameters
        ----------
        compile_time : float
            Time spent on JIT compilation (seconds)
        """
        self.jit_compile_time = compile_time
    
    def record_parameter_update(self, param_name: str, update_time: float):
        """Record parameter update time.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        update_time : float
            Time spent updating this parameter (seconds)
        """
        if param_name not in self.parameter_times:
            self.parameter_times[param_name] = []
        self.parameter_times[param_name].append(update_time)
    
    def finish_iteration(self):
        """Finish timing an iteration."""
        if self._iter_start_time is not None:
            iter_time = time.time() - self._iter_start_time
            self.iteration_times.append(iter_time)
            self._iter_start_time = None
    
    def finish(self):
        """Finish timing total fitting."""
        if self._start_time is not None:
            self.total_time = time.time() - self._start_time
            self._start_time = None
    
    def summary(self) -> Dict[str, Any]:
        """Get performance summary.
        
        Returns
        -------
        summary : dict
            Dictionary containing performance statistics
        """
        summary = {
            "family": self.family_name,
            "n_observations": self.n_observations,
            "use_jit": self.use_jit,
            "jit_compile_time": self.jit_compile_time,
            "n_iterations": len(self.iteration_times),
            "total_time": self.total_time,
        }
        
        if self.iteration_times:
            summary.update({
                "mean_iteration_time": sum(self.iteration_times) / len(self.iteration_times),
                "min_iteration_time": min(self.iteration_times),
                "max_iteration_time": max(self.iteration_times),
                "first_iteration_time": self.iteration_times[0],
                "last_iteration_time": self.iteration_times[-1],
            })
        
        # Parameter-specific timing
        if self.parameter_times:
            param_summary = {}
            for param, times in self.parameter_times.items():
                param_summary[param] = {
                    "mean_time": sum(times) / len(times),
                    "total_time": sum(times),
                }
            summary["parameter_times"] = param_summary
        
        return summary
    
    def print_summary(self, verbose: bool = True):
        """Print performance summary.
        
        Parameters
        ----------
        verbose : bool, default True
            If True, print detailed statistics
        """
        print("\n" + "=" * 70)
        print("GAMLSS Fitting Performance Summary")
        print("=" * 70)
        
        print(f"Family: {self.family_name}")
        print(f"Observations: {self.n_observations:,}")
        print(f"JIT Optimization: {'Enabled' if self.use_jit else 'Disabled'}")
        
        if self.use_jit and self.jit_compile_time > 0:
            print(f"JIT Compile Time: {self.jit_compile_time:.3f}s")
        
        print(f"\nIterations: {len(self.iteration_times)}")
        
        if self.iteration_times:
            mean_time = sum(self.iteration_times) / len(self.iteration_times)
            print(f"Mean Iteration Time: {mean_time:.4f}s")
            
            if verbose:
                print(f"  First Iteration: {self.iteration_times[0]:.4f}s")
                print(f"  Last Iteration: {self.iteration_times[-1]:.4f}s")
                print(f"  Min: {min(self.iteration_times):.4f}s")
                print(f"  Max: {max(self.iteration_times):.4f}s")
        
        print(f"\nTotal Fitting Time: {self.total_time:.3f}s")
        
        # Calculate throughput
        if self.total_time > 0 and self.n_observations > 0:
            throughput = self.n_observations / self.total_time
            print(f"Throughput: {throughput:,.0f} obs/sec")
        
        # Parameter timing breakdown
        if verbose and self.parameter_times:
            print("\nParameter Update Times:")
            for param, times in self.parameter_times.items():
                mean_time = sum(times) / len(times)
                total_time = sum(times)
                pct = (total_time / self.total_time * 100) if self.total_time > 0 else 0
                print(f"  {param}: {mean_time:.4f}s/iter (total: {total_time:.3f}s, {pct:.1f}%)")
        
        print("=" * 70)
    
    def print_iteration(self, iteration: int, deviance: float, converged: bool = False):
        """Print iteration progress.
        
        Parameters
        ----------
        iteration : int
            Current iteration number (1-indexed)
        deviance : float
            Current deviance value
        converged : bool, default False
            Whether the algorithm has converged
        """
        if self.iteration_times:
            iter_time = self.iteration_times[-1]
            status = "✓ Converged" if converged else ""
            print(f"  Iteration {iteration:3d}: deviance={deviance:12.4f}, time={iter_time:.4f}s {status}")


class PerformanceStats:
    """Container for performance statistics in fitted model.
    
    Attributes
    ----------
    use_jit : bool
        Whether JIT compilation was used
    jit_compile_time : float
        Time spent on JIT compilation (seconds)
    n_iterations : int
        Number of iterations
    total_time : float
        Total fitting time (seconds)
    mean_iteration_time : float
        Mean time per iteration (seconds)
    throughput : float
        Observations processed per second
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize from PerformanceMonitor.
        
        Parameters
        ----------
        monitor : PerformanceMonitor
            Performance monitor instance
        """
        summary = monitor.summary()
        
        self.use_jit = summary["use_jit"]
        self.jit_compile_time = summary.get("jit_compile_time", 0.0)
        self.n_iterations = summary["n_iterations"]
        self.total_time = summary["total_time"]
        self.mean_iteration_time = summary.get("mean_iteration_time", 0.0)
        
        # Calculate throughput
        if self.total_time > 0 and monitor.n_observations > 0:
            self.throughput = monitor.n_observations / self.total_time
        else:
            self.throughput = 0.0
        
        # Store full summary
        self._summary = summary
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerformanceStats(use_jit={self.use_jit}, "
            f"n_iterations={self.n_iterations}, "
            f"total_time={self.total_time:.3f}s, "
            f"throughput={self.throughput:,.0f} obs/sec)"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation
        """
        return self._summary
