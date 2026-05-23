from __future__ import annotations

import pytest

from omnilss.gen_family_builder import build_gen_family_skeleton


def test_count_skeleton_builder() -> None:
    fam = build_gen_family_skeleton("count_1_31")
    assert fam.name == "COUNT_1_31"
    assert fam.parameters == ("mu",)


def test_cont_skeleton_builder() -> None:
    fam = build_gen_family_skeleton("contRplus_2_11")
    assert fam.name == "CONTRPLUS_2_11"
    assert fam.parameters == ("mu", "sigma")


def test_unknown_skeleton() -> None:
    with pytest.raises(ValueError):
        build_gen_family_skeleton("weird")
