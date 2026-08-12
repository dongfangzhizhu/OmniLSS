"""Phase 5.7: AutoML resource governance.

- n_bootstrap, candidate-family count and total fit-request count are capped.
- Per-request timeout is handled by the client (default 60s).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from omnilss_pro.automl import (
        MAX_BOOTSTRAP_SAMPLES,
        MAX_CANDIDATE_FAMILIES,
        MAX_TOTAL_FIT_REQUESTS,
        bootstrap_deviance_intervals,
        rank_candidate_families,
    )
except ImportError as exc:  # grpcio etc. unavailable -> skip the module
    pytest.skip(f"omnilss-pro dependencies unavailable: {exc}", allow_module_level=True)


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def batch_fit(self, requests):
        self.calls += len(requests)
        return [
            {"success": True, "deviance": float(i % 10 + 1)} for i in range(len(requests))
        ]


def _data(n: int = 30) -> dict:
    rng = np.random.default_rng(0)
    return {"y": rng.normal(size=n), "x": rng.normal(size=n)}


def test_governance_constants_defined():
    assert MAX_BOOTSTRAP_SAMPLES > 0
    assert MAX_CANDIDATE_FAMILIES > 0
    assert MAX_TOTAL_FIT_REQUESTS > 0


def test_bootstrap_rejects_excessive_samples():
    client = _FakeClient()
    with pytest.raises(ValueError, match="n_bootstrap too large"):
        bootstrap_deviance_intervals(
            client, "y ~ x", _data(), ("NO",), n_bootstrap=MAX_BOOTSTRAP_SAMPLES + 1
        )


def test_bootstrap_rejects_excessive_families():
    client = _FakeClient()
    many = [f"F{i}" for i in range(MAX_CANDIDATE_FAMILIES + 1)]
    with pytest.raises(ValueError, match="too many candidate families"):
        bootstrap_deviance_intervals(
            client, "y ~ x", _data(), many, n_bootstrap=2
        )


def test_bootstrap_rejects_excessive_total_requests():
    client = _FakeClient()
    # families * n_bootstrap over the total cap
    n_fam = MAX_TOTAL_FIT_REQUESTS // MAX_BOOTSTRAP_SAMPLES + 1
    many = [f"F{i}" for i in range(min(n_fam, MAX_CANDIDATE_FAMILIES))]
    with pytest.raises(ValueError, match="total fit requests too large"):
        bootstrap_deviance_intervals(
            client, "y ~ x", _data(), many, n_bootstrap=MAX_BOOTSTRAP_SAMPLES
        )


def test_rank_rejects_excessive_families():
    client = _FakeClient()
    many = [f"F{i}" for i in range(MAX_CANDIDATE_FAMILIES + 1)]
    with pytest.raises(ValueError, match="too many candidate families"):
        rank_candidate_families(client, "y ~ x", _data(), many)


def test_governance_allows_normal_use():
    """Small bootstrap and candidate sets still work."""
    client = _FakeClient()
    result = bootstrap_deviance_intervals(
        client, "y ~ x", _data(), ("NO", "GA"), n_bootstrap=5, random_state=1
    )
    assert result["n_bootstrap"] == 5
    assert set(result["families"]) == {"NO", "GA"}
    assert result["families"]["NO"]["successful_resamples"] == 5
