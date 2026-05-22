from __future__ import annotations

import numpy as np

from omnilss.hazardFun import gen_hazard, hazardFun


def test_hazardFun_basic() -> None:
    x = np.array([1.0, 2.0, 3.0])
    pdf = np.array([0.2, 0.1, 0.05])
    cdf = np.array([0.2, 0.5, 0.8])
    out = hazardFun(pdf, cdf, x)
    np.testing.assert_allclose(out.hazard, pdf / (1.0 - cdf))


def test_gen_hazard_alias() -> None:
    x = np.array([0.0])
    out = gen_hazard(pdf=np.array([0.1]), cdf=np.array([0.2]), x=x)
    assert out.hazard.shape == (1,)
