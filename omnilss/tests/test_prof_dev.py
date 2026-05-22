from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.model import GAMLSSModel
from omnilss.prof_dev import prof_dev, prof_term


def test_prof_dev_finds_minimum() -> None:
    out = prof_dev([0.0, 1.0, 2.0], [3.0, 1.0, 2.0])
    assert out.minimum_index == 1
    assert out.grid.shape == (3,)


def test_prof_term_validates_model() -> None:
    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=3,
        y=np.array([0.0, 0.1, -0.1]),
        fitted_values={"mu": np.zeros(3), "sigma": np.ones(3)},
    )
    out = prof_term(model, [1, 2], [2.0, 1.0])
    assert out.minimum_index == 1
