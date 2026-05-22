from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.getSmo import edfAll, getSmo
from omnilss.model import GAMLSSModel


def test_getSmo_and_edfAll() -> None:
    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=3,
        y=np.array([0.0, 1.0, -1.0]),
        fitted_values={"mu": np.zeros(3), "sigma": np.ones(3)},
        lambda_selection_info={
            "mu": [
                {"variable": "x", "smoother": "pb", "edf": 2.5, "lambda_": 0.1},
                {"variable": "z", "smoother": "cs", "edf": 1.5, "lambda_": 0.2},
            ],
            "sigma": [{"variable": "x", "smoother": "pb", "edf": 1.0, "lambda_": 0.4}],
        },
    )

    mu = getSmo(model, parameter="mu")
    assert len(mu) == 2
    assert mu[0].smoother == "pb"
    edf = edfAll(model)
    assert edf["mu"] == 4.0
    assert edf["sigma"] == 1.0
