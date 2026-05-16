from omnilss.fitting import gamlss
from omnilss.optimizer_protocol import (
    OptimizerState,
    Phase0OptimizerProtocol,
    RSOptimizerAdapter,
)


def test_phase0_rs_adapter_implements_protocol_and_diagnostics():
    data = {"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 1.8, 3.1, 3.9]}
    model = gamlss("y ~ x", family="NO", data=data, method="RS")

    adapter = RSOptimizerAdapter()
    assert isinstance(adapter, Phase0OptimizerProtocol)

    st0 = adapter.initialize({})
    assert isinstance(st0, OptimizerState)

    st1 = adapter.step({"model": model}, st0)
    assert "loss" in st1.diagnostics
    assert "gradient_norm" in st1.diagnostics
    assert "step_size" in st1.diagnostics
    assert "condition_number" in st1.diagnostics
