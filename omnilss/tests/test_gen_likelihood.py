from __future__ import annotations

import numpy as np

from omnilss.distributions import NO
from omnilss.gen_likelihood import gen_likelihood
from omnilss.model import GAMLSSModel


def test_gen_likelihood_no_family_roundtrip() -> None:
    y = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    mu = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sigma = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    model = GAMLSSModel(
        par=("mu", "sigma"),
        family=NO(),
        df_fit=2.0,
        g_dev=0.0,
        n=3,
        y=y,
        fitted_values={"mu": mu, "sigma": sigma},
    )

    result = gen_likelihood(model)

    assert result.family == "NO"
    assert result.log_likelihood.shape == (3,)
    assert result.deviance_increment.shape == (3,)
    np.testing.assert_allclose(result.deviance_increment, -2.0 * result.log_likelihood, rtol=1e-6, atol=1e-6)
