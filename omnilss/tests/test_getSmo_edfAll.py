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
    assert mu[0].index == 0
    edf = edfAll(model)
    assert edf["mu"] == 4.0
    assert edf["sigma"] == 1.0


def test_edf_summary_and_finite_validation() -> None:
    from omnilss.getSmo import edf_summary

    model = GAMLSSModel(
        par=("mu",),
        family=NO(),
        df_fit=1.0,
        g_dev=0.0,
        n=2,
        y=np.array([0.0, 1.0]),
        fitted_values={"mu": np.array([0.0, 0.0])},
        lambda_selection_info={"mu": [{"variable": "x", "smoother": "pb", "edf": 2.0, "lambda_": 0.1}]},
    )
    summary = edf_summary(model)
    assert summary["total"] == 2.0
    assert summary["share"]["mu"] == 1.0

    bad_model = GAMLSSModel(
        par=("mu",),
        family=NO(),
        df_fit=1.0,
        g_dev=0.0,
        n=2,
        y=np.array([0.0, 1.0]),
        fitted_values={"mu": np.array([0.0, 0.0])},
        lambda_selection_info={"mu": [{"variable": "x", "smoother": "pb", "edf": float("nan"), "lambda_": 0.1}]},
    )
    try:
        edfAll(bad_model)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "must be finite" in str(exc)
