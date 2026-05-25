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


def test_dtop_uses_weighted_deviance_for_ranking() -> None:
    y = np.array([0.5, 2.0, 1.5], dtype=np.float64)
    mu = np.zeros_like(y)
    sigma = np.ones_like(y)
    w = np.array([10.0, 0.1, 1.0], dtype=np.float64)

    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=y.shape[0],
        y=y,
        fitted_values={"mu": mu, "sigma": sigma},
        weights=w,
    )

    out = dtop(model, n=1)
    # Unweighted max would be index=1 (y=2.0), weighted max is index=0 due to large weight.
    assert int(out.indices[0]) == 0
    np.testing.assert_allclose(out.weighted_deviance_increment[0], out.deviance_increment[0] * w[out.indices[0]], rtol=1e-8, atol=1e-8)


def test_dtop_rejects_weight_shape_mismatch() -> None:
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=y.shape[0],
        y=y,
        fitted_values={"mu": np.zeros_like(y), "sigma": np.ones_like(y)},
        weights=np.array([1.0, 2.0], dtype=np.float64),
    )

    try:
        dtop(model, n=2)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "weights must have same shape" in str(exc)
