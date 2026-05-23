from __future__ import annotations

import numpy as np

from omnilss.flexDist import flexDist
from omnilss.hazardFun import gen_hazard, hazardFun
from omnilss.momentSK import momentSK, momentSkew


def test_momentsk_and_momentskew() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = momentSK(y)
    assert np.isclose(m.mean, 2.5)
    assert m.variance > 0
    assert np.isclose(momentSkew(y), m.skewness)


def test_flexdist_summary() -> None:
    y = np.linspace(0.0, 1.0, 101)
    out = flexDist(y, centiles=(0.1, 0.5, 0.9))
    assert np.isclose(out.support_min, 0.0)
    assert np.isclose(out.support_max, 1.0)
    assert out.centiles.shape == (3,)


def test_hazard_helpers() -> None:
    x = np.array([0.0, 1.0, 2.0])
    pdf = np.array([0.4, 0.3, 0.2])
    cdf = np.array([0.4, 0.7, 0.9])
    h1 = hazardFun(pdf=pdf, cdf=cdf, x=x)
    h2 = gen_hazard(pdf=pdf, cdf=cdf, x=x)
    assert np.allclose(h1.hazard, h2.hazard)
