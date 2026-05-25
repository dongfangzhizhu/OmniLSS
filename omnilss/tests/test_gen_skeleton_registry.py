from __future__ import annotations

from omnilss.distribution_registry import resolve


def test_gen_skeleton_registry_resolves() -> None:
    assert resolve("COUNT_1_31").name == "COUNT_1_31"
    assert resolve("BINOM_2_33").name == "BINOM_2_33"
    assert resolve("CONTRPLUS_2_11").name == "CONTRPLUS_2_11"


def test_gen_skeleton_registry_uses_compat_family_shell() -> None:
    assert resolve("COUNT_1_31").__class__.__name__ == "GenSkeletonCompatFamily"
