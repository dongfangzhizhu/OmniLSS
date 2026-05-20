from __future__ import annotations

from dataclasses import dataclass

from omnilss.runtime.optimizer import (
    ConvergenceMonitor,
    ConvergenceThresholds,
    OptimizerTrace,
    OptimizerTraceEntry,
    RSOptimizer,
)


def test_convergence_monitor_supports_required_signals():
    monitor = ConvergenceMonitor(
        ConvergenceThresholds(
            gradient_norm_tol=1e-5,
            deviance_delta_tol=1e-5,
            parameter_delta_tol=1e-5,
            curvature_tol=1e8,
        )
    )
    status = monitor.evaluate(
        gradient_norm=1e-7,
        deviance_delta=1e-7,
        parameter_delta=1e-7,
        condition_number=1e4,
    )
    assert status.converged


def test_optimizer_trace_json_roundtrip(tmp_path):
    trace = OptimizerTrace()
    trace.record(
        OptimizerTraceEntry(
            iteration=1,
            deviance=12.0,
            gradient_norm=0.8,
            step_size=1.0,
            condition_number=100.0,
            runtime_seconds=0.02,
        )
    )
    path = tmp_path / "trace.json"
    trace.to_json(path)
    loaded = OptimizerTrace.from_json(path)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].deviance == 12.0


@dataclass
class _MockModel:
    iter: int
    additional_slots: dict
    coef: dict


def test_rs_optimizer_adapter_runner_contract():
    calls = []

    def _runner(method: str):
        calls.append(method)
        return _MockModel(iter=3, additional_slots={"converged": True, "cycles": 4}, coef={"mu": [1.0]})

    opt = RSOptimizer()
    result = opt.optimize({"runner": _runner})
    assert calls == ["RS"]
    assert result.converged is True
    assert result.iterations == 4
