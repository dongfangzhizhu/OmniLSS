from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.getZmatrix import getZmatrix
from omnilss.model import GAMLSSModel


def test_getZmatrix_returns_parameter_matrix() -> None:
    x_mu = np.column_stack([np.ones(3), np.array([0.0, 1.0, 2.0])])
    x_sigma = np.ones((3, 1))
    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=3.0,
        g_dev=0.0,
        n=3,
        y=np.array([0.0, 1.0, -1.0]),
        fitted_values={"mu": np.zeros(3), "sigma": np.ones(3)},
        design_matrices={"mu": x_mu, "sigma": x_sigma},
    )

    got = getZmatrix(model, parameter="mu")
    np.testing.assert_allclose(got, x_mu)
