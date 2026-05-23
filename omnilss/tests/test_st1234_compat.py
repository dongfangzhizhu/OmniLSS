from __future__ import annotations

from omnilss import ST1, ST2, ST3, ST4
from omnilss.distribution_registry import resolve


def test_st1234_constructors_and_registry_names() -> None:
    assert ST1().name == "ST1"
    assert ST2().name == "ST2"
    assert ST3().name == "ST3"
    assert ST4().name == "ST4"
    assert resolve("ST1").name == "ST1"
    assert resolve("ST2").name == "ST2"
    assert resolve("ST3").name == "ST3"
    assert resolve("ST4").name == "ST4"
