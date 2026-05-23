from __future__ import annotations

from omnilss import DBURR12
from omnilss.distribution_registry import resolve


def test_dburr12_constructor_and_registry() -> None:
    fam = DBURR12()
    assert fam.name == "DBURR12"
    assert fam.parameters == ("mu", "sigma", "nu")
    assert resolve("DBURR12").name == "DBURR12"
