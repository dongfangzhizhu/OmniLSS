from __future__ import annotations

import pytest

from omnilss import MULTIN


def test_multin_default_is_mn3() -> None:
    fam = MULTIN()
    assert fam.name == "MN3"
    assert fam.parameters == ("mu", "sigma")


def test_multin_explicit_category_mapping() -> None:
    assert MULTIN(3).name == "MN3"
    assert MULTIN(4).name == "MN4"
    assert MULTIN(5).name == "MN5"


def test_multin_invalid_category_count() -> None:
    with pytest.raises(ValueError, match="supports n_categories"):
        MULTIN(6)
