from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.model import GAMLSSModel
from omnilss.plot_smoothers import plotDF, plotLambda


def _model() -> GAMLSSModel:
    return GAMLSSModel(
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


def test_plotDF_returns_edf_rows() -> None:
    rows = plotDF(_model(), parameter="mu")
    assert len(rows) == 2
    assert rows[0].variable == "x"
    assert rows[0].smoother == "pb"
    assert rows[0].value == 2.5


def test_plotLambda_returns_lambda_rows() -> None:
    rows = plotLambda(_model(), parameter="sigma")
    assert len(rows) == 1
    assert rows[0].parameter == "sigma"
    assert rows[0].value == 0.4
