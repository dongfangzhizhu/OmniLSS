"""Runtime-layer configuration and backend contracts."""

from .backend import RuntimeBackend
from .errors import ConvergenceError, DistributionError, NumericalError, OmniLSSError, RuntimeExecutionError
from .config import DeterministicPolicy, DTypePolicy, RuntimeTolerancePolicy, SeedManager
from .optimizer import (
    CGOptimizer,
    ConvergenceMonitor,
    ConvergenceStatus,
    ConvergenceThresholds,
    NewtonOptimizer,
    Optimizer,
    OptimizerResult,
    OptimizerTrace,
    OptimizerTraceEntry,
    RSOptimizer,
    TrustRegionOptimizer,
)

__all__ = [
    "DeterministicPolicy",
    "DTypePolicy",
    "RuntimeBackend",
    "RuntimeTolerancePolicy",
    "SeedManager",
    "CGOptimizer",
    "ConvergenceMonitor",
    "ConvergenceStatus",
    "ConvergenceThresholds",
    "NewtonOptimizer",
    "Optimizer",
    "OptimizerResult",
    "OptimizerTrace",
    "OptimizerTraceEntry",
    "RSOptimizer",
    "TrustRegionOptimizer",
    "ProfileEvent",
    "RuntimeProfiler",
    "to_collapsed_stacks",
    "write_collapsed_stacks",
    "OmniLSSError",
    "NumericalError",
    "ConvergenceError",
    "DistributionError",
    "RuntimeExecutionError",
    "RuntimeLogEvent",
    "StructuredRuntimeLogger",
    "build_runtime_event",
]

from .profiling import ProfileEvent, RuntimeProfiler, to_collapsed_stacks, write_collapsed_stacks

from .observability import RuntimeLogEvent, StructuredRuntimeLogger, build_runtime_event
