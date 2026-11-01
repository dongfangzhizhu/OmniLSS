from __future__ import annotations

import numpy as np
import pytest

from omnilss import MULTIN


def test_multin_default_is_mn3() -> None:
    """MULTIN() is the MN3 family behind a compatibility shell.

    The shell keeps its own registry name ("MULTIN") -- that is what
    resolve("MULTIN") and the serialization contract pin -- so the mapping to
    MN3 is asserted behaviourally: same parameters and the same density, not
    the same .name.
    """
    from omnilss.distributions_b7 import MN3

    fam = MULTIN()
    base = MN3()
    assert fam.name == "MULTIN"
    assert fam.parameters == base.parameters
    grid = [0.0, 1.0, 2.0]
    np.testing.assert_allclose(
        fam.d(grid, 0.3, 0.4, 0.2), base.d(grid, 0.3, 0.4, 0.2), rtol=1e-12
    )


def test_multin_explicit_category_mapping() -> None:
    """Each category count must reproduce the matching MN* family's density."""
    from omnilss.distributions_b7 import MN3, MN4, MN5

    for n_categories, base in ((3, MN3()), (4, MN4()), (5, MN5())):
        fam = MULTIN(n_categories)
        assert fam.name == "MULTIN"
        assert fam.parameters == base.parameters
        grid = [0.0, 1.0, 2.0]
        np.testing.assert_allclose(
            fam.d(grid, 0.3, 0.4, 0.2), base.d(grid, 0.3, 0.4, 0.2), rtol=1e-12
        )


def test_multin_invalid_category_count() -> None:
    with pytest.raises(ValueError, match="supports n_categories"):
        MULTIN(6)
