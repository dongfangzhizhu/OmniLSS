"""Compatibility tests for algorithms.cg_fit wrapper."""

import numpy as np


def test_cg_wrapper_accepts_legacy_kwargs() -> None:
    """Legacy cg_fit kwargs should be accepted and produce a model."""
    from omnilss.algorithms import cg_fit

    np.random.seed(42)
    n = 80
    x = np.linspace(0, 5, n)
    y = 1.0 + 2.0 * x + np.random.normal(0, 0.5, n)
    data = {"y": y, "x": x}

    model = cg_fit(
        formula="y ~ x",
        sigma_formula="~ 1",
        family="NO",
        data=data,
        mu_step=0.7,
        sigma_step=0.7,
        max_outer_iter=10,
        outer_tol=1e-3,
        verbose=False,
    )

    assert model is not None
    assert model.additional_slots.get("method") == "CG"
    assert "cg_iterations" in model.additional_slots


def test_cg_wrapper_accepts_nu_tau_formulas() -> None:
    """Wrapper should accept nu/tau kwargs and fail with domain validation only."""
    from omnilss.algorithms import cg_fit
    import pytest

    np.random.seed(7)
    n = 60
    x = np.random.randn(n)
    y = np.random.randn(n)
    data = {"y": y, "x": x}

    with pytest.raises(ValueError, match="does not support parameter"):
        cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            nu_formula="~ 1",
            tau_formula="~ 1",
            family="NO",
            data=data,
            max_outer_iter=5,
            outer_tol=1e-2,
            verbose=False,
        )


def test_public_cg_fit_is_not_l_bfgs_deprecated_alias() -> None:
    """Package-level cg_fit should be the true CG formula wrapper, not old L-BFGS alias."""
    from omnilss.algorithms import cg_fit
    import warnings

    x = np.linspace(0.0, 1.0, 30)
    y = 1.0 + x + np.linspace(-0.02, 0.02, 30)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        model = cg_fit(
            formula="y ~ x",
            sigma_formula="~ 1",
            family="NO",
            data={"y": y, "x": x},
            max_outer_iter=10,
            outer_tol=1e-3,
            verbose=False,
        )

    assert model.additional_slots.get("method") == "CG"
    assert not any("L-BFGS" in str(record.message) for record in records)
