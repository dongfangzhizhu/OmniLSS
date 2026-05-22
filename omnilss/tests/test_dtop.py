from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.dtop import dtop
from omnilss.model import GAMLSSModel


def test_dtop_returns_sorted_top_contributors() -> None:
    y = np.array([0.0, 0.5, 5.0, -0.2, 3.0], dtype=np.float64)
    mu = np.zeros_like(y)
    sigma = np.ones_like(y)

    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=y.shape[0],
        y=y,
        fitted_values={"mu": mu, "sigma": sigma},
    )

    out = dtop(model, n=2)

    assert out.indices.shape == (2,)
    assert out.deviance_increment.shape == (2,)
    assert out.deviance_increment[0] >= out.deviance_increment[1]
    # largest |y| should dominate under NO with mu=0,sigma=1
    assert int(out.indices[0]) == 2
