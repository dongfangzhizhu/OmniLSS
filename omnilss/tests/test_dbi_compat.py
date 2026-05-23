from __future__ import annotations

from omnilss import DBI
from omnilss.distribution_registry import resolve


def test_dbi_constructor_and_registry() -> None:
    fam = DBI()
    assert fam.name == "DBI"
    assert fam.parameters == ("mu",)
    assert resolve("DBI").name == "DBI"
