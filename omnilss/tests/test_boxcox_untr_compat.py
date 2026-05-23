from __future__ import annotations

from omnilss import BCCGuntr, BCPEuntr, BCTuntr
from omnilss.distribution_registry import resolve


def test_boxcox_untr_constructors_and_registry() -> None:
    assert BCCGuntr().name == "BCCGUNTR"
    assert BCPEuntr().name == "BCPEUNTR"
    assert BCTuntr().name == "BCTUNTR"
    assert resolve("BCCGUNTR").name == "BCCGUNTR"
    assert resolve("BCPEUNTR").name == "BCPEUNTR"
    assert resolve("BCTUNTR").name == "BCTUNTR"
