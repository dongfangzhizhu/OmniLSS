# Phase 8.5: removed the misplaced optimizer wrapper (runtime/optimizer/base.py);
# the real optimizer protocol lives in omnilss.optimizer_protocol.
from .convergence import ConvergenceMonitor, ConvergenceStatus, ConvergenceThresholds
from .optimizer_trace import OptimizerTrace, OptimizerTraceEntry
from ...optimizer_protocol import RSOptimizerAdapter


class RSOptimizer:
    """Small adapter for invoking the existing RS runner contract."""

    def __init__(self):
        self.adapter = RSOptimizerAdapter()

    def optimize(self, context):
        model = context["runner"]("RS")
        state = self.adapter.step({"model": model}, self.adapter.initialize({}))
        return type(
            "OptimizerResult",
            (),
            {
                "converged": state.converged,
                "iterations": state.iteration,
                "diagnostics": state.diagnostics,
            },
        )()


__all__ = [
    "ConvergenceMonitor",
    "ConvergenceStatus",
    "ConvergenceThresholds",
    "OptimizerTrace",
    "OptimizerTraceEntry",
    "RSOptimizer",
]
