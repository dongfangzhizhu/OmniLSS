"""Phase 3.7: smoother admission is complete or fail-closed (Property 12).

For any smoother request, the system admits it to Release 1 fitting routes
if and only if parsing / basis / penalty / lambda-EDF / fitting / newdata
prediction / R golden artifact validation are complete; otherwise it raises
UnsupportedSmootherError containing the smoother name and missing capabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.fitting import _has_smooth_terms
from omnilss.smooth_fitting import (
    ADMITTED_SMOOTHERS,
    RELEASE1_CAPABILITIES,
    UnsupportedSmootherError,
    check_smoother_admission,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(61)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(4 * x) + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x, "z": rng.normal(size=n), "grp": rng.integers(0, 5, n)}


@pytest.mark.parametrize("smooth", ["pb(x)", "ps(x)", "cs(x)", "s(x)"])
def test_admitted_smoothers_fit(smooth, data) -> None:
    """pb/ps/cs (and s() defaulting to pb) are admitted to fitting."""
    model = gamlss(f"y ~ {smooth}", family="NO", data=data)
    assert np.isfinite(model.deviance)


UNVERIFIED = [
    "lo(x)",
    "random(grp)",
    "s(x, bs='re')",
    "s(x, bs='ri')",
    "s(x, bs='pcat')",
    "s(x, bs='fp')",
    "s(x, bs='bfp')",
    "s(x, bs='cy')",
    "s(x, bs='pvc')",
    "s(x, bs='pbm')",
    "s(x, bs='pbz')",
    "s(x, bs='pbo')",
    "s(x, bs='pbc')",
    "s(x, bs='pbp')",
    "s(x, bs='scs')",
    "te(x, z)",
    "ti(x, z)",
]


@pytest.mark.parametrize("smooth", UNVERIFIED)
def test_unverified_smoothers_rejected_fail_closed(smooth, data) -> None:
    """Unverified smoothers raise UnsupportedSmootherError with name + gaps."""
    with pytest.raises(UnsupportedSmootherError) as excinfo:
        gamlss(f"y ~ {smooth}", family="NO", data=data)
    err = excinfo.value
    assert err.smoother
    assert err.missing_capabilities
    assert set(err.missing_capabilities) <= set(RELEASE1_CAPABILITIES)
    envelope = err.to_dict()
    assert envelope["code"] == "unsupported_smoother"
    assert envelope["smoother"] == err.smoother


def test_admission_only_contains_validated_smoothers() -> None:
    """Release 1 admits exactly pb/ps/cs."""
    assert ADMITTED_SMOOTHERS == frozenset({"pb", "ps", "cs"})


def test_check_admission_unit() -> None:
    """Unit-level admission check admits pb/ps/cs and rejects others."""
    for admitted in ("pb", "ps", "cs"):
        check_smoother_admission(admitted)  # must not raise
    with pytest.raises(UnsupportedSmootherError):
        check_smoother_admission("lo")


def test_has_smooth_terms_covers_s_and_tensor(data) -> None:
    """s() and te() route through the smooth design path (no silent drop)."""
    assert _has_smooth_terms("y ~ s(x, bs='re')")
    assert _has_smooth_terms("y ~ te(x, z)")
    assert _has_smooth_terms("y ~ fp(x)")
    assert not _has_smooth_terms("y ~ x1 + x2")
