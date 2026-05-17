"""Integration test: Pro client calls Core via gRPC.

Run with Core already started, for example:
    cd omnilss-server && python grpc_main.py &
    cd omnilss-pro && python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

CORE_HOST = os.environ.get("OMNILSS_CORE_HOST", "localhost")
CORE_PORT = int(os.environ.get("OMNILSS_CORE_PORT", "50051"))


def _is_core_running() -> bool:
    try:
        import grpc

        channel = grpc.insecure_channel(f"{CORE_HOST}:{CORE_PORT}")
        grpc.channel_ready_future(channel).result(timeout=2)
        channel.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_core_running(),
    reason=f"OmniLSS Core not running at {CORE_HOST}:{CORE_PORT}",
)


def test_fit_predict_and_sample() -> None:
    from omnilss_pro.client import OmniLSSCoreClient

    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 5, n)
    y = 2 + 3 * x + rng.normal(size=n)

    with OmniLSSCoreClient(CORE_HOST, CORE_PORT) as client:
        result = client.fit("y ~ x", family="NO", data={"y": y, "x": x})
        assert result["model_id"]
        assert result["deviance"] > 0

        params = client.predict(result["model_id"], {"x": np.array([1.0, 2.0, 3.0])})
        assert "mu" in params
        assert len(params["mu"]) == 3

        samples = client.sample(result["model_id"], n=50)
        assert len(samples) == 50
