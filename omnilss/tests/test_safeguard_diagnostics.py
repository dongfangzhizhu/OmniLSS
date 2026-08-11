"""Phase 2.3: structured safeguard diagnostics.

When numerical safeguards fire during RS fitting, the model diagnostics must
record the event type plus the affected parameter/update.  Implemented as
``additional_slots["safeguard_events"]`` — a tuple of lightweight dicts
{type, parameter, ...} collected along the rs_step -> iwls_step ->
compute_working_weights_and_response chain.

Safeguard types:
- eta_clip: eta update clamped to the link's bounds
- hessian_ridge: stabilize_hessian added a ridge to the normal equations
- hessian_floor: second derivative floored (concavity floor)
- nan_replacement: non-finite working response replaced with 0
- step-halving is recorded separately as rs_step_halving_by_param

Alignment: DEVELOPMENT_PLAN Phase 2 task 2.3 — "Hessian floor/eta clip/
step-halving/NaN replacement/fallback solver 触发时，模型 diagnostics 含类型+
受影响更新".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss

KNOWN_TYPES = {"eta_clip", "hessian_ridge", "hessian_floor", "nan_replacement"}


def _events(model) -> tuple[dict, ...]:
    return tuple(model.additional_slots.get("safeguard_events", ()))


def _types(model) -> list[str]:
    return [e["type"] for e in _events(model)]


def test_eta_clip_records_type_and_affected_parameter() -> None:
    """An extreme outlier drives the log-link eta out of bounds -> eta_clip."""
    rng = np.random.default_rng(3)
    y = rng.gamma(3, 2, 100)
    y[0] = 1e10  # extreme outlier forces eta toward log(1e10) >> 15
    model = gamlss("y ~ 1", family="GA", data={"y": y})

    events = [e for e in _events(model) if e["type"] == "eta_clip"]
    assert events, f"expected eta_clip events, got {_types(model)}"
    assert all(e.get("parameter") == "mu" for e in events)
    assert all(e.get("n_clipped", 0) >= 1 for e in events)
    assert all(e.get("total", 0) == len(y) for e in events)


def test_hessian_floor_records_type_and_parameter() -> None:
    """Zero-altered data yields zero/positive second derivatives -> floor."""
    rng = np.random.default_rng(42)
    n = 100
    y = np.where(rng.uniform(size=n) < 0.2, 0.0, rng.gamma(2.0, 2.0, n))
    model = gamlss("y ~ 1", family="ZAGA", data={"y": y})

    events = [e for e in _events(model) if e["type"] == "hessian_floor"]
    assert events, f"expected hessian_floor events, got {_types(model)}"
    assert all(e.get("parameter") in {"mu", "sigma", "nu"} for e in events)
    assert all(e.get("n_floor", 0) >= 1 for e in events)


def test_hessian_ridge_records_type_and_lambda() -> None:
    """Collinear design matrix triggers the stabilized ridge."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    data = {
        "y": 1.0 + 0.5 * x + rng.normal(scale=0.3, size=60),
        "x1": x,
        "x2": x + 1e-8 * rng.normal(size=60),
    }
    model = gamlss("y ~ x1 + x2", family="NO", data=data)

    events = [e for e in _events(model) if e["type"] == "hessian_ridge"]
    assert events, f"expected hessian_ridge events, got {_types(model)}"
    assert all(e.get("parameter") == "mu" for e in events)
    assert all(e.get("lambda", 0.0) > 0.0 for e in events)


def test_step_halving_recorded() -> None:
    """Step-halving diagnostics are exposed per parameter."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=80)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.3, size=80)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    halving = model.additional_slots.get("rs_step_halving_by_param")
    assert halving is not None
    assert isinstance(halving, dict)
    assert set(halving) <= {"mu", "sigma", "nu", "tau"}


def test_safeguard_events_structure() -> None:
    """Every event is a dict with a registered type."""
    rng = np.random.default_rng(5)
    y = rng.gamma(3, 2, 100)
    y[0] = 1e10
    model = gamlss("y ~ 1", family="GA", data={"y": y})

    events = _events(model)
    assert isinstance(events, tuple)
    for event in events:
        assert isinstance(event, dict)
        assert event["type"] in KNOWN_TYPES


def test_stable_fit_records_no_spurious_safeguards() -> None:
    """A well-conditioned fit should not fire safeguards."""
    rng = np.random.default_rng(13)
    n = 120
    x = rng.normal(size=n)
    y = 1.0 + 0.6 * x + rng.normal(scale=0.4, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x})

    assert _types(model) == []
