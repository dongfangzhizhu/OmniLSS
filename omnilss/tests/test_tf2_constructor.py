from __future__ import annotations

from omnilss import TF, TF2
from omnilss.distribution_registry import resolve


def test_tf2_constructor_and_registry() -> None:
    tf = TF()
    tf2 = TF2()
    assert tf2.name == "TF2"
    assert tf2.parameters == tf.parameters
    assert resolve("TF2").name == "TF2"
