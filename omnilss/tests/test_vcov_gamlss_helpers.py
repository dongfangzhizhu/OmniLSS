from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.model import GAMLSSModel
from omnilss.vcov_gamlss import get_K, rvcov


def _build_model() -> GAMLSSModel:
    y = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    X = np.column_stack([np.ones(3), np.array([0.0, 1.0, 2.0])])
    coef_mu = np.array([0.1, 0.2], dtype=np.float64)
    coef_sigma = np.array([0.0], dtype=np.float64)
    return GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=3.0,
        g_dev=0.0,
        n=3,
        y=y,
        fitted_values={"mu": np.array([0.1, 0.3, 0.5]), "sigma": np.ones(3)},
        coefficients={"mu": coef_mu, "sigma": coef_sigma},
        design_matrices={"mu": X, "sigma": np.ones((3, 1))},
        weights=np.ones(3),
    )


def test_get_K_matches_flattened_coeff_count() -> None:
    model = _build_model()
    assert get_K(model) == 3


def test_rvcov_returns_square_matrix() -> None:
    model = _build_model()
    cov = rvcov(model)
    assert cov.shape == (3, 3)
